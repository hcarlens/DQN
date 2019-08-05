import gym
import matplotlib.pyplot as plt
import torch

from pytorch_rl.dqn_agent import DQNAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.memory_buffer import MemoryBuffer
import pytorch_rl.utils as utils
from pytorch_rl.utils import loss_functions, optimisers
import random

RANDOM_SEED = 1

env = gym.make('CartPole-v0')

loss_fn = loss_functions['mse']
optimiser = optimisers['adam']

dqn_agent = DQNAgent(learning_rate=0.00025, 
                     discount_rate=0.99,
                     num_inputs=4,
                     num_neurons=64,
                     num_outputs=2,#
                     random_seed=RANDOM_SEED,
                     loss_fn=loss_fn,
                     optimiser=optimiser
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