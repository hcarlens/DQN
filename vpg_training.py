import logging

import gym
import torch
import random
import numpy as np

from pytorch_rl.vpg_agent import VPGAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.memory_buffer import MemoryBuffer
from pytorch_rl.utils import loss_functions, optimisers, create_sequential_model
from pytorch_rl.sanity_envs import SanityEnvV4, SanityEnvV5
import argparse


parser = argparse.ArgumentParser(description='Train a DQN agent on CartPole.')
parser.add_argument('--seed',  type=int, default=None, help='Random seed. ')
parser.add_argument('--num_episodes',  type=int, default=50000, help='Number of episodes to train for. ')
parser.add_argument('--loss_fn',  type=str, default='mse', help='Loss function. ')
parser.add_argument('--optimiser',  type=str, default='adam', help='Loss function. ')
parser.add_argument('--lr',  type=float, default=0.01, help='Learning rate. ')
parser.add_argument('--discount',  type=float, default=0.99, help='Discount rate. ')
parser.add_argument('--batch_size',  type=int, default=1000, help='Batch size. ')
parser.add_argument('--layers_spec',  type=str, default='32', help='Spec for the main net. E.g. "32_32" for two layers with 32 neurons. ')
parser.add_argument('--final_layer_neurons',  type=int, default=64, help='Number of neurons in the final layer. ')
parser.add_argument('--buffer_length',  type=int, default=100, help='Maximum number of memories stored in the memory buffer before overwriting. ')
parser.add_argument('--timestep_to_start_learning',  type=int, default=20, help='Timestep at which we start updating the agent. ')
parser.add_argument('--target_update_steps',  type=int, default=10, help='Frequency at which to update the target network. ')
parser.add_argument('--test_every_n_steps',  type=int, default=20, help='Frequency at which to run a test episode. ')
parser.add_argument('--train_every_n_steps',  type=int, default=1, help='Frequency at which to run a backward pass. ')
parser.add_argument('--epsilon_decay_rate',  type=float, default=0.5, help='Rate at which epsilon (random action probability) decays. ')
parser.add_argument('--cuda',  action='store_true', default=False, help='Pass this flag to train on GPU instead of CPU. ')
parser.add_argument('--gymenv',  type=str, default='CartPole-v0', help='Choose an OpenAI Gym env to run the agent on. ')

logging.getLogger().setLevel(logging.INFO)

args = parser.parse_args()

env = gym.make(args.gymenv)

loss_fn = loss_functions[args.loss_fn]
optimiser = optimisers[args.optimiser]

# set seeds
if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

layers_spec = tuple(int(x) for x in args.layers_spec.split('_'))
net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=layers_spec, num_outputs=env.action_space.n,
                                  dropout_rate=0, activation_function='relu', final_activation=False)

agent = VPGAgent(learning_rate=args.lr,
                     discount_rate=args.discount,
                     policy_net=net,
                     random_seed=args.seed,
                     loss_fn=loss_fn,
                     optimiser=optimiser,
                     cuda=args.cuda,
)

trainer = Trainer(
    env=env, 
    agent=agent,
    memory_buffer=MemoryBuffer(buffer_length=args.buffer_length), 
    timestep_to_start_learning=args.timestep_to_start_learning,
    batch_size=args.batch_size,
    random_seed=args.seed,
    train_every_n_steps=args.train_every_n_steps,
    test_every_n_steps=args.test_every_n_steps,
    hparams={'lr': args.lr, 'dr': args.discount, 'layers': args.layers_spec, 'final_layer': args.final_layer_neurons,
             'cuda': args.cuda, 'optimiser': args.optimiser, 'loss_fn': args.loss_fn, 'buffer_length': args.buffer_length,
        }
)
trainer.run(num_episodes=args.num_episodes)