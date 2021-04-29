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
        super().__init__()

        self.state = np.zeros(1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.name = 'SanityV4'

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = 1 if action == self.state else -1
        done = True

        self.num_steps += 1

        return self.state.copy(), reward, done, {}

    def reset(self):
        """ Reset the environment. """
        self.state = np.random.randint(low=0, high=2, size=(1,))
        self.num_steps = 0
        return self.state.copy()

class SanityEnvV5(gym.Env):
    """
    Two actions, constantly incrementing observation from 0 to 1, 10 timesteps, +1 reward if action 1 is chosen at timestep 5.
    -1 reward for picking action 1 at any other point.
    """

    def __init__(self, max_num_steps: int = 1, correct_timestep = 0, terminate_on_penalty = False):
        super().__init__()

        self.state = np.zeros(1, dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.name = 'SanityV5'
        self.max_num_steps = max_num_steps
        self.correct_timestep = correct_timestep
        self.terminate_on_penalty = terminate_on_penalty


    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        reward = 0
        if action == 1 and self.num_steps == self.correct_timestep:
            # reward if action is taken at the right time
            reward = 1
        elif action == 1:
            # penalty if action is taken at the wrong time
            reward = -1

        done = True if (self.num_steps >= self.max_num_steps
                        or self.terminate_on_penalty and reward == -1
                        ) else False

        self.state += 0.1
        self.num_steps += 1

        return self.state.copy(), reward, done, {}


    def reset(self):
        """ Reset the environment. """
        self.state = np.zeros(1, dtype=np.float64)
        self.num_steps = 0
        return self.state.copy()


class SanityEnvV6(gym.Env):
    """
    Parametric env, with action masking!
    Four actions, constantly incrementing observation from 0 to 0.3, 3 timesteps, +1 reward if action 1 is chosen at timestep 2.
    -1 reward for picking action 1 at any other point.
    Error raised if invalid action (per action mask) is chosen.
    """

    parametric = True

    def __init__(self, max_num_steps: int = 3, correct_timestep = 2, terminate_on_penalty = False):
        super().__init__()

        self.state = np.zeros(1, dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.name = 'SanityV6'
        self.max_num_steps = max_num_steps
        self.correct_timestep = correct_timestep
        self.terminate_on_penalty = terminate_on_penalty
        self.action_mask = np.ones(4)

    def randomise_action_mask(self):
        """ Set a random action mask for this timestep"""
        self.action_mask = np.random.randint(0, 2, 4)
        if self.action_mask.max() == 0:
            # if we don't have any valid actions, randomly pick one
            self.action_mask[np.random.randint(self.action_space.n)] = 1

    def step(self, action: int):
        """ Implement action, update environment, and return state/reward/terminated flag """

        if self.action_mask[action] == 0:
            raise ValueError(f'Selected invalid action {action} when action mask is {self.action_mask}. ')

        reward = 0
        if action == 1 and self.num_steps == self.correct_timestep:
            # reward if action is taken at the right time
            reward = 1
        elif action == 1:
            # penalty if action is taken at the wrong time
            reward = -1

        done = True if (self.num_steps >= self.max_num_steps
                        or self.terminate_on_penalty and reward == -1
                        ) else False

        self.state += 0.1
        self.num_steps += 1

        self.randomise_action_mask()
        return {'observation': self.state.copy(), 'action_mask': self.action_mask.copy()}, reward, done, {}


    def reset(self):
        """ Reset the environment. """
        self.state = np.zeros(1, dtype=np.float64)
        self.num_steps = 0
        self.randomise_action_mask()
        return {'observation': self.state.copy(), 'action_mask': self.action_mask.copy()}

#todo: add stochastic env (e.g. multi-armed bandit)