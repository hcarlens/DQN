import logging

import gym

from pytorch_rl.dqn_agent import DQNAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.memory_buffer import MemoryBuffer
from pytorch_rl.utils import loss_functions, optimisers, create_sequential_model
import argparse


parser = argparse.ArgumentParser(description='Train a DQN agent on CartPole.')
parser.add_argument('--seed',  type=int, default=None, help='Random seed. ')
parser.add_argument('--num_episodes',  type=int, default=500, help='Number of episodes to train for. ')
parser.add_argument('--loss_fn',  type=str, default='mse', help='Loss function. ')
parser.add_argument('--optimiser',  type=str, default='adam', help='Loss function. ')
parser.add_argument('--lr',  type=float, default=0.00025, help='Learning rate. ')
parser.add_argument('--discount',  type=float, default=0.99, help='Discount rate. ')
parser.add_argument('--batch_size',  type=int, default=32, help='Batch size. ')
parser.add_argument('--hidden_neurons_0',  type=int, default=32, help='Number of neurons in the first hidden layer. ')
parser.add_argument('--hidden_neurons_1',  type=int, default=32, help='Number of neurons in the second hidden layer. ')
parser.add_argument('--final_layer_neurons',  type=int, default=32, help='Number of neurons in the final layer. ')
parser.add_argument('--buffer_length',  type=int, default=50000, help='Maximum number of memories stored in the memory buffer before overwriting. ')
parser.add_argument('--timestep_to_start_learning',  type=int, default=1000, help='Timestep at which we start updating the agent. ')
parser.add_argument('--target_update_steps',  type=int, default=1000, help='Frequency at which to update the target network. ')
parser.add_argument('--epsilon_decay_rate',  type=float, default=0.99, help='Rate at which epsilon (random action probability) decays. ')
parser.add_argument('--cuda',  action='store_true', default=False, help='Pass this flag to train on GPU instead of CPU. ')
parser.add_argument('--gymenv',  type=str, default='CartPole-v0', help='Choose an OpenAI Gym env to run the agent on. ')

logging.getLogger().setLevel(logging.INFO)

args = parser.parse_args()

env = gym.make(args.gymenv)

loss_fn = loss_functions[args.loss_fn]
optimiser = optimisers[args.optimiser]


dqn_net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=(32,), num_outputs=16,
                                  dropout_rate=0, activation_function='relu', final_activation=True)


dqn_agent = DQNAgent(learning_rate=args.lr, 
                     discount_rate=args.discount,
                     main_net=dqn_net,
                     final_layer_neurons=16,
                     num_outputs=2,
                     random_seed=args.seed,
                     loss_fn=loss_fn,
                     optimiser=optimiser,
                     cuda=args.cuda
                    )

trainer = Trainer(
    env=env, 
    agent=dqn_agent, 
    memory_buffer=MemoryBuffer(buffer_length=args.buffer_length), 
    start_epsilon=1, 
    timestep_to_start_learning=args.timestep_to_start_learning,
    batch_size=args.batch_size,
    target_update_steps=args.target_update_steps,
    epsilon_decay_rate=args.epsilon_decay_rate,
    random_seed=args.seed
)
trainer.run(num_episodes=args.num_episodes)