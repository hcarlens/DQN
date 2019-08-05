import gym
import matplotlib.pyplot as plt
import torch

from pytorch_rl.dqn_agent import DQNAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.memory_buffer import MemoryBuffer
import random

RANDOM_SEED = 1

env = gym.make('CartPole-v0')

dqn_agent = DQNAgent(learning_rate=0.00025, 
                     discount_rate=0.99,
                     num_inputs=4,
                     num_neurons=64,
                     num_outputs=2,#
                     random_seed=RANDOM_SEED
                    )

trainer = Trainer(
    env=env, 
    agent=dqn_agent, 
    memory_buffer=MemoryBuffer(buffer_length=50000), 
    start_epsilon=1, 
    timestep_to_start_learning=1000,
    batch_size=32,
    target_update_steps=1000,
    epsilon_decay_rate=0.99,
    random_seed=RANDOM_SEED
)
trainer.run(num_episodes=100)

plt.plot(trainer.episode_lengths)
plt.show()