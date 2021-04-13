import statistics

import numpy as np
import torch
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from pytorch_rl.utils import RingBuffer

class Trainer:
    def __init__(self,
                 env,
                 agent,
                 memory_buffer,
                 start_epsilon,
                 timestep_to_start_learning=1000,
                 batch_size=32,
                 epsilon_decay_rate=0.999,
                 buffer_length=50000,
                 target_update_steps=1000,
                 max_num_steps=200,
                 end_epsilon=0.01,
                 random_seed=None,
                 train_every_n_steps: int=1,
                 name='DQN',
                 write_to_tensorboard=True):
        self.env = env
        self.epsilon = start_epsilon
        self.agent = agent
        self.memory_buffer = memory_buffer

        self.max_num_steps = max_num_steps
        self.end_epsilon = end_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.batch_size = batch_size
        self.buffer_length = buffer_length
        self.target_update_steps = target_update_steps
        self.timestep_to_start_learning = timestep_to_start_learning
        self.train_every_n_steps = train_every_n_steps

        self.name = name
        self.write_to_tensorboard = write_to_tensorboard
        
        self.global_step = 0
        self.backward_passes = 0
        self.episode_lengths = RingBuffer(100)
        self.loss_values = RingBuffer(100)
        self.episode_cuml_rewards = RingBuffer(100)
        self.most_recent_episode_final_info = {}

        if self.write_to_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y_%m_%d'), datetime.now().strftime('%H_%M_%S_') + env.name + '_' + name))

        if random_seed is not None:
            self.seed(random_seed)

        print('Trainer initialised.')

    def seed(self, random_seed):
            # seed all the things
            self.agent.seed(random_seed)
            self.env.seed(random_seed)
            self.env.action_space.seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

    def on_episode_start(self):
        self.episode_start_time = time.perf_counter()
        self.episode_start_timestep = self.global_step

    def on_episode_end(self):
        elapsed_time = time.perf_counter() - self.episode_start_time
        elapsed_steps = self.global_step - self.episode_start_timestep

        if self.write_to_tensorboard:
            self.writer.add_scalar('Speed/seconds_per_episode', elapsed_time, self.global_step)
            self.writer.add_scalar('Speed/steps_per_second', elapsed_steps/elapsed_time, self.global_step)
            self.writer.add_scalar('Buffer_length', self.memory_buffer.current_length, global_step=self.global_step)
            self.writer.add_scalar('Epsilon',  self.epsilon, global_step=self.global_step)
            self.writer.add_scalar('running_average_reward_100_trials', self.episode_cuml_rewards.mean(), global_step=self.global_step)
            self.writer.add_scalar('running_average_loss_100_steps', self.loss_values.mean(), global_step=self.global_step)

    def run(self, num_episodes):
        # run through episodes
        for e in range(num_episodes):
            logging.debug(f'Starting episode {e}')
            self.on_episode_start()

            observation = self.env.reset()
            current_episode_actions = []
            current_episode_rewards = []

            for t in range(self.max_num_steps):
                # set the target network weights to be the same as the q-network ones every so often
                if self.global_step % self.target_update_steps == 0:
                    self.agent.update_target_network()
                    print('Target network updated. ')

                # with probability epsilon, choose a random action
                # otherwise use Q-network to pick action
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(observation).item()

                next_observation, reward, done, info = self.env.step(action)
                current_episode_actions.append(action)
                current_episode_rewards.append(reward)

                # add memory to buffer
                memory = (observation, action, reward, next_observation, done)
                self.memory_buffer.add_memory(memory)

                if self.global_step > self.timestep_to_start_learning and self.global_step % self.train_every_n_steps == 0:
                    # sample a minibatch of experiences and update q-network
                    minibatch = self.memory_buffer.sample_minibatch(
                        self.batch_size)

                    loss, q_values = self.agent.fit_batch(minibatch)
                    self.backward_passes += 1
                    if self.write_to_tensorboard:
                        self.writer.add_scalar('Loss', loss, global_step=self.global_step)
                        self.writer.add_scalar('Q/median', statistics.median(q_values), global_step=self.global_step)
                        self.writer.add_scalar('Q/min', min(q_values), global_step=self.global_step)
                        self.writer.add_scalar('Q/max', max(q_values), global_step=self.global_step)
                        self.writer.add_scalar('Q/mean', sum(q_values)/len(q_values), global_step=self.global_step)
                        self.writer.add_histogram('Q_values', q_values, global_step=self.global_step)
                        self.writer.add_scalar('Backward_passes', self.backward_passes, global_step=self.global_step)
                    self.loss_values.add(loss)

                observation = next_observation
                self.global_step += 1
                if done or t == self.max_num_steps - 1:
                    break

            self.on_episode_end()
            if self.write_to_tensorboard:
                self.writer.add_scalar('Episode_timesteps', t + 1, global_step=self.global_step)
                self.writer.add_scalar('Episode_reward', sum(current_episode_rewards), global_step=self.global_step)
            self.episode_lengths.add(t)
            self.episode_cuml_rewards.add(sum(current_episode_rewards))
            self.most_recent_episode_final_info = info

            for k, v in info.items():
                # print all numbers in the final info dict to tensorboard
                if type(v) in (int, float) and self.write_to_tensorboard:
                    self.writer.add_scalar(f'info/{k}', v, global_step=self.global_step)

            if self.loss_values:
                loss_min, loss_mean, loss_max = self.loss_values.min(), self.loss_values.mean(), self.loss_values.max()
            else:
                loss_min, loss_mean, loss_max = np.nan, np.nan, np.nan

            if e % 1 == 0:
                logging.info(
                    f"Ep {e}; {self.episode_cuml_rewards.last()} reward. 100 ep ravg: {np.floor(self.episode_cuml_rewards.mean())}. Eps {self.epsilon:.2f}. Loss: {loss_min:.2f}|{loss_mean:.2f}|{loss_max:.2f}"
                )
                logging.info( f"Most recent ep info: {info}")

            # decrease epsilon value
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate,
                               self.end_epsilon)
