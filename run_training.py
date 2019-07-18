from dqn import QNetwork
from dqn_agent import DQNAgent
from agent_trainer import Trainer
from memory_buffer import MemoryBuffer
import gym

env = gym.make('CartPole-v0')


def build_model(learning_rate=0.00025):
    # define Q-network
    # 4 inputs (one for each scalar observable) +1 to represent the action
    # todo: change to 4 inputs; 5 outputs to require fewer forward passes
    return QNetwork(5, 128, learning_rate)

dqn_agent = DQNAgent(network_generator=build_model, discount_rate=0.99)

trainer = Trainer(
    env=env, 
    agent=dqn_agent, 
    memory_buffer=MemoryBuffer(buffer_length=50000), 
    epsilon=1, 
    obs_normalisation=[0.686056, 0.792005, 0.075029, 0.414541]
)

trainer.run(num_episodes=500)