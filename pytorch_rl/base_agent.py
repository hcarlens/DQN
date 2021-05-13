from abc import abstractmethod
import numpy as np
from typing import Tuple

class BaseAgent:

    def __init__(self):
        self.name = ''

    @abstractmethod
    def act(self, observation: np.array, action_mask: np.array = None):
        """ Act"""
        pass

    def set_eval_mode(self):
        self.eval_mode = True

    def set_train_mode(self):
        self.eval_mode = False

    @abstractmethod
    def fit_batch(self, minibatch: Tuple[Tuple]):
        """ Update policy/value function based on a minibatch of (observation, action, reward, next_observation, done) """
        pass
