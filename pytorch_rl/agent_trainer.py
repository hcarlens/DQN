import statistics

from typing import Optional, Tuple, Dict, Union

import numpy as np
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from pytorch_rl.utils import RingBuffer
from pytorch_rl.base_agent import BaseAgent

class Trainer:
    def __init__(self,
                 env,
                 agent: BaseAgent,
                 memory_buffer,
                 timestep_to_start_learning: int = 1000,
                 batch_size: int = 32,
                 buffer_length: int = 50000,
                 max_num_steps: int = 10000,
                 random_seed: Optional[int] = None,
                 train_every_n_steps: int=1,
                 write_to_tensorboard: bool = True,
                 test_every_n_steps: int = 1000,
                 log_every_n_steps: int = 20, # note: this is actually every n training steps, not env steps
                 hparams: dict = {}):
        self.env = env
        self.agent: BaseAgent = agent
        self.memory_buffer = memory_buffer

        self.max_num_steps = max_num_steps
        self.batch_size = batch_size
        self.buffer_length = buffer_length
        self.timestep_to_start_learning = timestep_to_start_learning
        self.train_every_n_steps = train_every_n_steps
        self.test_every_n_steps = test_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        self.write_to_tensorboard = write_to_tensorboard
        
        self.global_step = 0
        self.test_steps = 0
        self.global_episode = 0
        self.num_test_episodes = 0
        self.backward_passes = 0
        self.last_test_timestep = 0
        self.episode_lengths = RingBuffer(100)
        self.backward_pass_durations = RingBuffer(100)
        self.recent_actions = RingBuffer(1000)
        self.loss_values = RingBuffer(100)
        self.episode_cuml_rewards = RingBuffer(100)
        self.test_episode_rewards = RingBuffer(100)

        self.most_recent_episode_final_info = {}

        self.max_episode_reward = None

        self.hparams = hparams
        self.random_seed = random_seed

        if random_seed is not None:
            self.seed(random_seed)

        if self.write_to_tensorboard:
            self.initialise_tensorboard_writer()

        # check whether we're dealing with a parametric env (if so, we'll need action masking)
        if hasattr(self.env, 'parametric') and self.env.parametric is True:
            self.parametric_env = True
            self.actions = np.arange(self.env.action_space.n)
        else:
            self.parametric_env = False

        logging.info('Trainer initialised.')

    def reward_to_go(self, rewards):
        """
        Takes in a list of rewards and returns an array of 'rewards to go' (remaining return)
        from https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
        # todo: rewrite using a cumulative sum in numpy
        """
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def initialise_tensorboard_writer(self):
        if hasattr(self.env, 'name'):
            env_name = self.env.name  # for custom-made envs
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'spec') and hasattr(self.env.unwrapped.spec, 'id'):
            env_name = self.env.unwrapped.spec.id  # for gym envs
        else:
            env_name = 'UNKNOWN_ENV'
        self.writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y_%m_%d'),
                                                         datetime.now().strftime('%H_%M_%S_') + env_name + '_' + self.agent.name))

    def seed(self, random_seed):
        """ Seed all the things (see https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/) """
        self.agent.seed(random_seed)
        self.env.seed(random_seed)
        self.env.action_space.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        logging.info(f'Random seed set: {self.random_seed}')

    def on_episode_start(self):
        self.episode_start_time = time.perf_counter()
        self.episode_start_timestep = self.global_step
        self.agent.set_train_mode() # ensure we're in train mode so we're exploring!

    def on_episode_end(self):
        """ Write to tensorboard and perform some admin after each episode """

        elapsed_time = time.perf_counter() - self.episode_start_time
        elapsed_steps = self.global_step - self.episode_start_timestep
        self.global_episode += 1

        most_recent_reward = self.episode_cuml_rewards.last()
        if (self.max_episode_reward is None or most_recent_reward > self.max_episode_reward) and most_recent_reward is not np.nan:
            self.max_episode_reward = most_recent_reward

        if self.write_to_tensorboard:
            self.writer.add_scalar('Speed/seconds_per_episode', elapsed_time, self.global_step)
            self.writer.add_scalar('Speed/steps_per_second', elapsed_steps/elapsed_time, self.global_step)
            self.writer.add_scalar('Progress/Episodes', self.global_episode, global_step=self.global_step)
            self.writer.add_scalar('Progress/Test_episodes', self.num_test_episodes, global_step=self.global_step)
            self.writer.add_scalar('Progress/Steps_per_episode', elapsed_steps, global_step=self.global_step)
            self.writer.add_scalar('Progress/Buffer_length', self.memory_buffer.current_length, global_step=self.global_step)
            self.writer.add_scalar('Reward/Max',  self.max_episode_reward, global_step=self.global_step)
            self.writer.add_scalar('Reward/Last', most_recent_reward, global_step=self.global_step)
            self.writer.add_scalar('Reward/100_ep_avg', self.episode_cuml_rewards.mean(), global_step=self.global_step)
            self.writer.add_histogram('Recent_actions', self.recent_actions.values, global_step=self.global_step)
            self.writer.add_scalar('Loss/100_step_average', self.loss_values.mean(), global_step=self.global_step)
            if hasattr(self.agent, 'epsilon'):
                self.writer.add_scalar('Progress/Epsilon',  self.agent.epsilon, global_step=self.global_step)

        if self.global_step > self.last_test_timestep + self.test_every_n_steps:
            self.run_test_episode()

    def on_train_end(self):
        """ Various things to do when training ends """
        if self.write_to_tensorboard and hasattr(self.writer, 'add_hparams'):
            # save the hyperparams if we have the correct version of tensorboard
            self.hparams['seed'] = self.random_seed
            # self.hparams['start_epsilon'] = self.agent.start_epsilon
            # self.hparams['end_epsilon'] = self.agent.end_epsilon
            # self.hparams['epsilon_decay_steps'] = self.agent.epsilon_decay_steps
            self.hparams['bs'] = self.batch_size
            self.hparams['timestep_to_start_learning'] = self.timestep_to_start_learning
            # self.hparams['target_update_steps'] = self.agent.target_update_steps
            self.hparams['train_every_n_steps'] = self.train_every_n_steps
            self.writer.add_hparams(hparam_dict=self.hparams, metric_dict={'max_episode_reward': self.max_episode_reward, 'last_100_mean_reward': self.episode_cuml_rewards.mean(),
                                                                           'last_100_min_reward': self.episode_cuml_rewards.min(), 'last_100_max_reward': self.episode_cuml_rewards.max(),})
            logging.info('Wrote hyperparams.')

    def process_env_observation(self, env_observation: Union[np.array, Dict]) -> Tuple[np.array, Optional[np.array]]:
        """ Normalises the environment observation across parametric and non-parametric environments """
        if self.parametric_env:
            observation = env_observation['observation']
            action_mask = env_observation['action_mask']
        else:
            observation = env_observation
            action_mask = None
        return observation, action_mask

    def run_test_episode(self):
        """
        Run a test episode with the policy as it is.
        No training, no randomness.
        Todo: add checkpointing when we have a good test episode.
        """
        logging.debug('Running a test episode')
        test_ep_start_time = time.perf_counter()

        env_observation = self.env.reset()
        self.agent.set_eval_mode() # ensure we're using the optimal policy and not exploring!
        observation, action_mask = self.process_env_observation(env_observation)
        current_episode_actions = []
        current_episode_rewards = []

        for t in range(self.max_num_steps):
            action = self.agent.act(observation, action_mask)

            env_observation, reward, done, info = self.env.step(action)
            observation, action_mask = self.process_env_observation(env_observation)
            current_episode_actions.append(action)
            current_episode_rewards.append(reward)

            self.test_steps += 1
            if done or t == self.max_num_steps - 1:
                break

        self.test_episode_rewards.add(sum(current_episode_rewards))
        test_ep_duration = time.perf_counter() - test_ep_start_time
        if self.write_to_tensorboard:
            self.writer.add_scalar('Test/Reward', sum(current_episode_rewards), self.global_step)
            self.writer.add_scalar('Test/Max_reward', self.test_episode_rewards.all_time_max(), self.global_step)
            self.writer.add_scalar('Speed/Test_episode_duration', test_ep_duration, global_step=self.global_step)
            self.writer.add_histogram('Test/Actions', np.array(current_episode_actions), global_step=self.global_step)

            for k, v in info.items():
                # print all numbers in the final info dict to tensorboard
                if type(v) in (int, float) and self.write_to_tensorboard:
                    self.writer.add_scalar(f'Test_info/{k}', v, global_step=self.global_step)
                if type(v) in (tuple, np.ndarray) and self.write_to_tensorboard:
                    for i, v_i in enumerate(v):
                        self.writer.add_scalar(f'Test_info/{k}_{i}', v_i, global_step=self.global_step)

        self.last_test_timestep = self.global_step
        self.num_test_episodes += 1

        logging.info(f'Test episode done. Reward: {sum(current_episode_actions)}. ')

    def run(self, num_episodes):

        info = {}

        # run through episodes
        for e in range(num_episodes):
            logging.debug(f'Starting episode {e}')
            self.on_episode_start()

            env_observation = self.env.reset()
            current_episode_cuml_reward = 0

            observation, action_mask = self.process_env_observation(env_observation)

            for t in range(self.max_num_steps):
                logging.debug('New step. ')

                # randomness is in the agent now!
                action = self.agent.act(observation, action_mask)

                next_env_observation, reward, done, info = self.env.step(action)
                next_observation, next_action_mask = self.process_env_observation(next_env_observation)

                current_episode_cuml_reward += reward

                self.recent_actions.add(action)

                # add memory to buffer
                memory = (observation, action, reward, next_observation, next_action_mask, done)
                self.memory_buffer.add_memory(memory)

                if self.global_step > self.timestep_to_start_learning and self.global_step % self.train_every_n_steps == 0:
                    backward_pass_start_time = time.perf_counter()
                    # sample a minibatch of experiences and update q-network
                    logging.debug('Sampling from replay buffer...')
                    minibatch = self.memory_buffer.sample_minibatch(
                        self.batch_size)
                    logging.debug('Sampled batch')

                    batch_info = self.agent.fit_batch(minibatch)
                    loss = batch_info['loss']
                    logging.debug('Fitted batch')

                    self.backward_passes += 1
                    backward_pass_duration = time.perf_counter() - backward_pass_start_time
                    self.backward_pass_durations.add(backward_pass_duration)
                    if self.write_to_tensorboard and self.backward_passes % self.log_every_n_steps == 0:
                        logging.debug('Printing to tensorboard...')
                        self.writer.add_scalar('Loss/Last', loss, global_step=self.global_step)
                        if hasattr(self.agent, 'num_target_updates'):
                            self.writer.add_scalar('Progress/target_net_updates', self.agent.num_target_net_updates,
                                                   global_step=self.global_step)
                        self.writer.add_scalar('Progress/backward_passes', self.backward_passes,
                                               global_step=self.global_step)
                        self.writer.add_scalar('Speed/Backward_pass_duration_mean', self.backward_pass_durations.mean(),
                                               global_step=self.global_step)
                        self.writer.add_scalar('Speed/Backward_pass_duration_min', self.backward_pass_durations.min(),
                                               global_step=self.global_step)
                        self.writer.add_scalar('Speed/Backward_pass_duration_max', self.backward_pass_durations.max(),
                                               global_step=self.global_step)
                        if 'q_values' in batch_info:
                            q_values = batch_info['q_values']
                            self.writer.add_scalar('Q/median', statistics.median(q_values), global_step=self.global_step)
                            self.writer.add_scalar('Q/min', min(q_values), global_step=self.global_step)
                            self.writer.add_scalar('Q/max', max(q_values), global_step=self.global_step)
                            self.writer.add_scalar('Q/mean', sum(q_values)/len(q_values), global_step=self.global_step)
                            self.writer.add_histogram('Q_values', q_values, global_step=self.global_step)
                        if 'state_values' in batch_info:
                            self.writer.add_histogram('state_values', batch_info['state_values'], global_step=self.global_step)
                            # todo: print state value samples every few steps
                        if 'action_advantages' in batch_info:
                            self.writer.add_histogram('action_advantages', batch_info['action_advantages'], global_step=self.global_step)
                            # todo: print action:advantage samples every few steps
                        if 'policy_loss' in batch_info:
                            self.writer.add_scalar('Policy/loss', batch_info['policy_loss'], global_step=self.global_step)
                            self.writer.add_scalar('Policy/avg_entropy', batch_info['policy_avg_entropy'], global_step=self.global_step)
                        logging.debug('Done printing to tensorboard...')
                    self.loss_values.add(loss)

                observation = next_observation
                action_mask = next_action_mask
                self.global_step += 1
                if done or t == self.max_num_steps - 1:
                    break

            self.episode_cuml_rewards.add(current_episode_cuml_reward)
            self.episode_lengths.add(t)
            self.on_episode_end()
            self.most_recent_episode_final_info = info

            for k, v in info.items():
                logging.debug('Printing info')
                # print all numbers in the final info dict to tensorboard
                if type(v) in (int, float, np.float32, np.float64) and self.write_to_tensorboard:
                    self.writer.add_scalar(f'info/{k}', v, global_step=self.global_step)
                if type(v) in (tuple, np.ndarray) and self.write_to_tensorboard:
                    for i, v_i in enumerate(v):
                        self.writer.add_scalar(f'info/{k}_{i}', v_i, global_step=self.global_step)
                logging.debug('Printed info')

            if self.loss_values:
                loss_min, loss_mean, loss_max = self.loss_values.min(), self.loss_values.mean(), self.loss_values.max()
            else:
                loss_min, loss_mean, loss_max = np.nan, np.nan, np.nan

            if e % 1 == 0:
                logging.info(
                    f"Ep {e}; {self.episode_cuml_rewards.last()} reward. 100 ep ravg: {np.floor(self.episode_cuml_rewards.mean())}. Loss: {loss_min:.2f}|{loss_mean:.2f}|{loss_max:.2f}"
                )
                logging.info( f"Most recent ep info: {info}")

        self.on_train_end()