import torch
import math

loss_functions = {'mse': torch.nn.MSELoss, 'huber': torch.nn.SmoothL1Loss}

optimisers = {
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD
}


class RunningStats():
    """
    Calculate running sample mean and standard deviation in an efficient and numerically stable way.
    Based on Welford's method, as outlined in https://www.johndcook.com/blog/standard_deviation/. 
    """

    def __init__(self):
        self.clear()

    def clear(self):
        # observation counter
        self.n = 0

        # mean values
        self.old_m = None
        self.old_s = None

        # variance values
        self.old_s = None
        self.new_s = None

    def push(self, x: float):
        self.n += 1

        if self.n == 1:
            # first observation: mean=x, variance=0
            self.old_m = x
            self.new_m = x
            self.old_s = 0.
        else:
            # calculate new running mean and variance
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            # reset old values ready for next iteration
            self.old_m = self.new_m
            self.old_s = self.new_s

    def __len__(self):
        return self.n

    @property
    def mean(self):
        return self.new_m if self.n > 0 else 0.0

    @property
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    @property
    def standard_deviation(self):
        return math.sqrt(self.variance)