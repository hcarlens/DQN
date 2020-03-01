# DQN

This repository features a PyTorch implementation of DeepMind's DQN, a value-function-based reinforcement learning (RL) agent as first described in [this paper](https://arxiv.org/abs/1312.5602).

In addition to the base DQN implementation, the [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf) and [Duelling Network Architectures](https://arxiv.org/pdf/1511.06581.pdf) have been implemented. The agent can take in any PyTorch optimiser/loss function, and supports multiple types of gradient clipping.



## Installation

You can install pytorch_rl by running `pip install .` in the repository folder. This will also install PyTorch, OpenAI Gym, NumPy, and matplotlib. 

In order to run the notebooks or tests, you'll need to install a few extra packages, by running `pip install -r requirements.txt`

The agent trainer logs loss values, predicted Q-values, episode lengths, and other useful debugging metrics to Tensorboard. In order to view these, you'll need to install tensorboard: `pip install tensorboard`

## Use

Once installed, you can set an agent training on the [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem (learning to balance a pole vertically on a cart that moves horizontally) with this command: `python run_training.py --num_episodes=500 --seed=1`. After training is complete, you'll get see a chart of episode lengths - if this reaches 200 it means our agent has learnt to solve CartPole! 

The training script accepts a number of arguments. For example  `python run_training.py --batch_size=64 --hidden_neurons=256` runs the same script with a larger batch size of 64 and 256 neurons in the hidden layer. To see all supported arguments, run `python run_training.py --help`. 