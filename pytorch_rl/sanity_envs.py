"""
Various toy environments for testing our agents.
Based on https://andyljones.com/posts/rl-debugging.html
Todo:
 - Add stochastic reward envs (E.g. multi-armed bandits)
 - Add delayed reward envs
"""

import numpy as np
import gym
from gym import spaces

class SanityEnvV0(gym.Env):
    """
    One action, zero observation, one timestep long, +1 reward every timestep: This isolates the value network.
    If my agent can't learn that the value of the only observation it ever sees it 1,
    there's a problem with the value loss calculation or the optimizer.
    """

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self.state = np.zeros(1)
        self.name = 'SanityV0'

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = 1
        done = True

        self.num_steps += 1

        return self.state, reward, done, {}

    def reset(self):
        """ Reset the environment. """
        self.num_steps = 0
        return self.state


class SanityEnvV1(gym.Env):
    """
    One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time:
    If my agent can learn the value in V0 but not this one - meaning it can learn a constant reward but not a predictable one!
    - it must be that backpropagation through my network is broken.
    """

    def __init__(self):
        super().__init__()

        self.state = None
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self.name = 'SanityV1'

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = self.state.item()
        done = True

        self.num_steps += 1

        return self.state, reward, done, {}

    def reset(self):
        """ Reset the environment. """
        self.state = (np.random.randint(0, 2, size=(1,)) * 2) - 1 # random +1/-1 reward
        self.num_steps = 0
        return self.state


class SanityEnvV2(gym.Env):
    """
    One action, zero-then-one observation, two timesteps long, +1 reward at the end.
    If my agent can learn the value in V1 but not this one, it must be that my reward discounting is broken.
    """

    def __init__(self):
        super().__init__()

        self.state = np.zeros(1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self.name = 'SanityV2'

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = 0 if self.num_steps == 0 else 1
        done = True if self.num_steps == 1 else 0

        self.num_steps += 1
        self.state = np.array([1])

        return self.state, reward, done, {}

    def reset(self):
        """ Reset the environment. """
        self.state = np.zeros(1)
        self.num_steps = 0
        return self.state

class SanityEnvV3(gym.Env):
    """
    Two actions, zero observation, one timestep long, action-dependent +1/-1 reward:
    The first env to exercise the policy! If my agent can't learn to pick the better action,
    there's something wrong with either my advantage calculations, my policy loss or my policy update.
    That's three things, but it's easy to work out by hand the expected values for each one and check that
        the values produced by your actual code line up with them.
    """

    def __init__(self):
        super().__init__()

        self.state = np.zeros(1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.name = 'SanityV3'

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = 1 if action == 0 else -1
        done = True

        self.num_steps += 1

        return self.state, reward, done, {}

    def reset(self):
        """ Reset the environment. """
        self.state = np.zeros(1)
        self.num_steps = 0
        return self.state


class SanityEnvV4(gym.Env):
    """
    Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent +1/-1 reward:
    Now we've got a dependence on both obs and action.
    The policy and value networks interact here, so there's a couple of things to verify:
    that the policy network learns to pick the right action in each of the two states,
    and that the value network learns that the value of each state is +1.
    If everything's worked up until now, then if - for example - the value network fails to learn here,
    it likely means your batching process is feeding the value network stale experience.
    """

    def __init__(self):
        """ Initialise a Toy Selection Environment. """
        super().__init__()

        self.state = np.zeros(1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.name = 'SanityV4'

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = 1 if action == self.state else 0
        done = True

        self.num_steps += 1

        return self.state, reward, done, {}

    def reset(self):
        """ Reset the environment. """
        self.state = np.random.randint(low=0, high=2, size=(1,))
        self.num_steps = 0
        return self.state
