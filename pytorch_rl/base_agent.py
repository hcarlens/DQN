from abc import abstractmethod
import numpy as np

class BaseAgent:

    @abstractmethod
    def act(self, observation: np.array, action_mask: np.array = None):
        """ Act"""
        pass

    def set_eval_mode(self):
        self.eval_mode = True


    def set_train_mode(self):
        self.eval_mode = False
