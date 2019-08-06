import gym
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from pytorch_rl.dqn_agent import DQNAgent, CNNQNetwork
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.memory_buffer import MemoryBuffer
import pytorch_rl.utils as utils
from pytorch_rl.utils import loss_functions, optimisers
import random

def atari_pre_processor(observation):
    """
    Preprocess atari frames to make things easier.
    observation is 210 x 160 x 3
    we crop to 80 x 80, take only one colour channel
    """
    cropped_single_channel_observation = observation[35:-15:2, ::2, 0].copy()

    # scale
    cropped_single_channel_observation = cropped_single_channel_observation / 255.

    # add a channel dimension to the front since our CNN is expecting it
    return np.expand_dims(cropped_single_channel_observation, 0)

RANDOM_SEED = 1

env = gym.make('Pong-v0')

loss_fn = loss_functions['mse']
optimiser = optimisers['adam']

dqn_agent = DQNAgent(learning_rate=0.001, 
                     discount_rate=0.99,
                     num_inputs=4,
                     num_outputs=3,
                     random_seed=RANDOM_SEED,
                     loss_fn=loss_fn,
                     optimiser=optimiser,
                     q_network_producer=CNNQNetwork
                    )

trainer = Trainer(
    env=env, 
    agent=dqn_agent, 
    memory_buffer=MemoryBuffer(buffer_length=50000), 
    start_epsilon=1, 
    timestep_to_start_learning=100000,
    max_num_steps=10000, 
    batch_size=32,
    target_update_steps=10000,
    epsilon_decay_rate=0.999,
    random_seed=RANDOM_SEED,
    observation_pre_processor=atari_pre_processor
)
trainer.run(num_episodes=1000)

# save agent model
torch.save(dqn_agent.q_network.state_dict(), os.path.join('models', trainer.model_name))

plt.plot(trainer.episode_rewards)
plt.show()