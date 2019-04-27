#%%
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import gym


"""
TODO:
- train a simple DQN agent
- add Rainbow improvements
- add seed for network initialisation
- move to LL TF or Keras custom model

"""


class ReplayMemory:
    """ Vanilla replay memory"""

    def __init__(self, size=1000):
        """
        Use a list to store memories for fast sampling, and keep track of where we are in the list
        so we replace the oldest memories first.
        """
        self.memories = []
        self.max_size = size
        self.current_size = 0
        self.counter = 0

    def add_memory(self, memory):
        if len(self.memories) < self.max_size:
            self.memories.append(memory)
        else:
            self.memories[self.counter] = memory
            self.counter += 1

    def sample(self, num_samples):
        return random.sample(self.memories, min(num_samples, len(self.memories)))


class NeuralNet:
    """ Simple NN """

    def __init__(
        self,
        observation_space_dim,
        action_space_dim,
        layer_neurons,
        optimizer_learning_rate,
    ):
        self.model = keras.Sequential(
            [Dense(128, activation=tf.nn.relu, input_shape=(5,)), Dense(1)]
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_learning_rate)
        self.model.compile(loss="mse", optimizer=optimizer)

    def predict(self, input):
        return self.model.predict(input)

    def get_weights(self):
        pass

    def set_weights(self, new_weights):
        pass


class DQN:
    """ DQN agent """

    def __init__(
        self,
        environment: gym.Env,
        target_update_fequency: int = 1000,
        epsilon: float = 1e-2,
        minibatch_size: int = 32,
        optimizer_learning_rate: float = 3e-4,
        layer_neurons: int = 16,
    ):

        if isinstance(environment.action_space, gym.spaces.Discrete):
            self.action_space = environment.action_space
        else:
            raise TypeError("DQN only supports discrete action spaces!")

        self.observation_space = environment.observation_space

        self.policy_network = NeuralNet(
            self.observation_space.shape,
            self.action_space.n,
            layer_neurons,
            optimizer_learning_rate,
        )
        self.target_network = NeuralNet(
            self.observation_space.shape,
            self.action_space.n,
            layer_neurons,
            optimizer_learning_rate,
        )
        self.memory = ReplayMemory()
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size

    def train(self):
        minibatch = self.memory.sample(self.minibatch_size)
        print(minibatch)
        pass

    def act(self, observation):
        return self.policy_network.predict(observation)

    def update_target_network(self):
        """ Update the target network with a copy of the policy network """
        self.target_network.set_weights(self.policy_network.get_weights())


def training_loop():
    environment = gym.make("CartPole-v0")

    # reproducibility :)
    environment.seed(0)

    agent = DQN(environment)
    num_episodes = 1000
    max_num_steps = 200

    for e in range(num_episodes):
        observation = environment.reset()
        for t in range(max_num_steps):
            action = agent.act(observation)
            
            next_observation, reward, done, info = environment.step(action)
            memory = (observation, action, reward, next_observation, done)
            agent.memory.add_memory(memory)


training_loop()