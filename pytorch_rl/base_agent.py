from abc import abstractmethod
import numpy as np

class BaseAgent:

    @abstractmethod
    def act(self, observation: np.array, action_mask: np.array = None):
        """ Act"""
        pass