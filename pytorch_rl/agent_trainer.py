import numpy as np
import torch
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    """
    # todo: add deterministic mode
    """

    def __init__(self,
                 env,
                 agent,
                 memory_buffer,
                 start_epsilon,
                 timestep_to_start_learning=1000,
                 batch_size=32,
                 epsilon_decay_rate=0.999,
                 learning_rate=0.00025,
                 buffer_length=50000,
                 target_update_steps=1000,
                 max_num_steps=200,
                 end_epsilon=0.01,
                 random_seed=None,
                 name='DQN'):
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

        self.name = name

        
        self.total_steps = 0
        self.episode_lengths = []
        self.loss_values = []

        self.writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y_%m_%d'), datetime.now().strftime('%H_%M_%S_') + name))

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

    def run(self, num_episodes):
        # run through episodes
        for e in range(num_episodes):
            observation = self.env.reset()
            current_actions = []

            for t in range(self.max_num_steps):
                # set the target network weights to be the same as the q-network ones every so often
                if self.total_steps % self.target_update_steps == 0:
                    self.agent.update_target_network()
                    print('Target network updated. ')

                # with probability epsilon, choose a random action
                # otherwise use Q-network to pick action
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(observation).item()

                next_observation, reward, done, info = self.env.step(action)
                current_actions.append(action)

                # add memory to buffer
                memory = (observation, action, reward, next_observation, done)
                self.memory_buffer.add_memory(memory)

                if self.total_steps > self.timestep_to_start_learning:
                    # sample a minibatch of experiences and update q-network
                    minibatch = self.memory_buffer.sample_minibatch(
                        self.batch_size)

                    loss, q_values = self.agent.fit_batch(minibatch)
                    self.writer.add_scalar('Loss', loss, global_step=self.total_steps)
                    self.writer.add_histogram('Q_values', q_values, global_step=self.total_steps)
                    self.loss_values.append(loss)


                observation = next_observation
                self.total_steps += 1
                if done or t == self.max_num_steps - 1:
                    break

            self.writer.add_scalar('Episode_timesteps', t, global_step=self.total_steps)
            self.writer.add_scalar('Buffer_length', self.memory_buffer.current_length, global_step=self.total_steps)  
            self.episode_lengths.append(t)
            self.writer.add_scalar('Action_zero_pct',  sum(current_actions) / len(current_actions), global_step=self.total_steps)  
            self.writer.add_scalar('Epsilon',  self.epsilon, global_step=self.total_steps)  
            self.writer.add_scalar('running_average_100_trials', np.mean(self.episode_lengths[-100:]), global_step=self.total_steps)


            if self.loss_values:
                loss_min, loss_mean, loss_max = min(self.loss_values[-100:]), np.mean(self.loss_values[-100:]), max(self.loss_values[-100:])
            else:
                loss_min, loss_mean, loss_max = np.inf, np.inf, np.inf

            if e % 10 == 0:
                print(
                    f"Ep {e}; {t+1} steps. 100 ep ravg: {np.floor(np.average(self.episode_lengths[-100:]))}. Eps {self.epsilon:.2f}. Loss: {loss_min:.2f}|{loss_mean:.2f}|{loss_max:.2f}"
                )

            # decrease epsilon value
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate,
                               self.end_epsilon)
