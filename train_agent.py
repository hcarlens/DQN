# external imports
import numpy as np
import pandas as pd
import gym
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import logging

from q_network import QNetwork
from dqn_agent import DQNAgent
from memory_buffer import MemoryBuffer

# make Jupyter notebook appear wider
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

# set hyperparameters (mostly taken from DQN paper)
num_episodes = 500
max_num_steps = 200
start_epsilon = 1
end_epsilon = 0.01
epsilon_decay_rate = 0.99
discount_rate = 0.99
batch_size = 32
optimizer_learning_rate = 0.00025
buffer_length = 50000
target_update_steps = 1000
timestep_to_start_learning = 1000

# observed standard deviations on an earlier training run
stddevs = [0.686056, 0.792005, 0.075029, 0.414541]

def build_model():
    # define Q-network (two layers with 32 neurons each)
    # 4 inputs (one for each scalar observable) +1 to represent the action
    # todo: change to 4 inputs; 5 outputs to require fewer forward passes
    return QNetwork(5, 128, optimizer_learning_rate)

# todo: remove references to global variables
# todo: add deterministic mode
class Trainer:
    def __init__(self, env, agent, memory_buffer, epsilon, obs_normalisation=[1,1,1,1], logdir='/logs'):
        self.env = env
        self.total_steps = 0
        self.episode_lengths = []
        self.epsilon = epsilon
        self.agent = agent
        self.memory_buffer = memory_buffer
        
        self.obs_normalisation = obs_normalisation
        print('Trainer initialised')
        
    def run(self, num_episodes):
        # run through episodes
        for e in range(num_episodes):
            observation = self.env.reset()
            observation = np.divide(observation, self.obs_normalisation)

            for t in range(max_num_steps):
                # set the target network weights to be the same as the q-network ones every so often
                if self.total_steps % target_update_steps == 0:
                    self.agent.update_target_network()
                                
                # with probability epsilon, choose a random action
                # otherwise use Q-network to pick action 
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(observation)

                next_observation, reward, done, info = env.step(action)

                # add memory to buffer
                memory = (observation, action, reward, next_observation, done)
                self.memory_buffer.add_memory(memory)
                
                if self.total_steps > timestep_to_start_learning:
                    # sample a minibatch of experiences and update q-network
                    minibatch = self.memory_buffer.sample_minibatch(batch_size)
                    self.agent.fit_batch(minibatch)

                observation = next_observation
                self.total_steps += 1
                if done or t == max_num_steps - 1:
                    break 

            self.episode_lengths.append(t)

            if e % 10 == 0:
                print("Episode {} finished after {} timesteps. 100 ep running avg {}. Epsilon {}.".format(e, t+1, np.floor(np.average(self.episode_lengths[-100:])), self.epsilon))

            # decrease epsilon value
            self.epsilon = max(self.epsilon * epsilon_decay_rate, end_epsilon)

# todo: add tensorboard logging
# todo: checkpoint/save models
# todo: automatically stop when a certain benchmark is reached
# initialise environment, required objects, and some variables
env = gym.make('CartPole-v0')


dqn_agent = DQNAgent(network_generator=build_model, discount_rate=discount_rate)

trainer = Trainer(env=env, agent=dqn_agent, memory_buffer=MemoryBuffer(buffer_length), epsilon=start_epsilon, obs_normalisation=stddevs)
trainer.run(num_episodes)