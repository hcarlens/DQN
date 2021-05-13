import logging

import gym
import torch
import random
import numpy as np

from pytorch_rl.dqn_agent import DQNAgent
from pytorch_rl.actor_critic_agent import ActorCriticAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.memory_buffer import MemoryBuffer
from pytorch_rl.utils import loss_functions, optimisers, create_sequential_model
from pytorch_rl.sanity_envs import SanityEnvV4, SanityEnvV5
import argparse


parser = argparse.ArgumentParser(description='Train an agent on CartPole.')
parser.add_argument('--seed',  type=int, default=None, help='Random seed. ')
parser.add_argument('--agent',  type=str, default='DQN', help='Type of agent. Currently supported: DQN, AC. ')
parser.add_argument('--num_episodes',  type=int, default=5000, help='Number of episodes to train for. ')
parser.add_argument('--loss_fn',  type=str, default='mse', help='Loss function. ')
parser.add_argument('--optimiser',  type=str, default='adam', help='Loss function. ')
parser.add_argument('--lr',  type=float, default=0.001, help='Learning rate. ')
parser.add_argument('--discount',  type=float, default=0.99, help='Discount rate. ')
parser.add_argument('--batch_size',  type=int, default=64, help='Batch size. ')
parser.add_argument('--layers_spec',  type=str, default='64', help='Spec for the main net. E.g. "32_32" for two layers with 32 neurons. ')
parser.add_argument('--final_layer_neurons',  type=int, default=64, help='Number of neurons in the final layer. ')
parser.add_argument('--buffer_length',  type=int, default=10000, help='Maximum number of memories stored in the memory buffer before overwriting. ')
parser.add_argument('--timestep_to_start_learning',  type=int, default=2000, help='Timestep at which we start updating the agent. ')
parser.add_argument('--target_update_steps',  type=int, default=300, help='Frequency at which to update the target network. ')
parser.add_argument('--test_every_n_steps',  type=int, default=1000, help='Frequency at which to run a test episode. ')
parser.add_argument('--train_every_n_steps',  type=int, default=4, help='Frequency at which to run a backward pass. ')
parser.add_argument('--epsilon_decay_rate',  type=float, default=0.95, help='Rate at which epsilon (random action probability) decays. ')
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

if args.agent == 'DQN':
    agent = DQNAgent(learning_rate=args.lr,
                         discount_rate=args.discount,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=layers_spec,
                         final_layer_neurons=args.final_layer_neurons,
                         num_outputs=env.action_space.n,
                         random_seed=args.seed,
                         loss_fn=loss_fn,
                         optimiser=optimiser,
                         cuda=args.cuda,
                        target_update_steps = args.target_update_steps,
    )
elif args.agent == 'AC':
    agent = ActorCriticAgent(
                     action_space_dim=env.action_space.n,
                     observation_space_dim=env.observation_space.shape[0],
                     policy_learning_rate=args.lr,
                     value_learning_rate=args.lr,
                     discount_rate=args.discount,
                     policy_net_layers_spec=layers_spec,
                     value_net_layers_spec=layers_spec,
                     random_seed=args.seed,
                     loss_fn=loss_fn,
                     optimiser=optimiser,
                     cuda=args.cuda,
    )
else:
    raise ValueError(f'Unknown agent specified: {args.agent}')

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
             'cuda': args.cuda, 'optimiser': args.optimiser, 'loss_fn': args.loss_fn, 'buffer_length': args.buffer_length,}
)
trainer.run(num_episodes=args.num_episodes)