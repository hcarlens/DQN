import gym
import matplotlib.pyplot as plt
import torch

from dqn import QNetwork
from dqn_agent import DQNAgent
from agent_trainer import Trainer
from memory_buffer import MemoryBuffer
import random

torch.manual_seed(0)
random.seed(0)
env = gym.make('CartPole-v0')
env.seed(0)

dqn_agent = DQNAgent(learning_rate=0.00025, 
                     discount_rate=0.9,
                     num_inputs=4,
                     num_neurons=32,
                     num_outputs=2
                    )

for p in dqn_agent.q_network.parameters():
 print(p[0])

trainer = Trainer(
    env=env, 
    agent=dqn_agent, 
    memory_buffer=MemoryBuffer(buffer_length=50000), 
    start_epsilon=1, 
    timestep_to_start_learning=1000,
    batch_size=32,
    target_update_steps=1000,
    epsilon_decay_rate=0.99
)
trainer.run(num_episodes=10)

plt.plot(trainer.episode_lengths)
plt.show()