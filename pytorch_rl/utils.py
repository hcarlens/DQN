import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple

loss_functions = {'mse': torch.nn.MSELoss, 'huber': torch.nn.SmoothL1Loss}

optimisers = {
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD
}

def load_activation_function(activation_function_name: str):
    if activation_function_name == 'tanh':
        return nn.Tanh()
    elif activation_function_name == 'relu':
        return nn.ReLU()
    elif activation_function_name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('activation_function must be one of "tanh", "relu", or "sigmoid". ')

def create_sequential_model(num_inputs: int, layers_spec: Tuple[int, ...], num_outputs: int, dropout_rate: float,
                            activation_function: str, batch_norm: bool = False, final_activation: bool = False):
    """
    Create a PyTorch model from a spec.
    Returning a sequential, since we can't just have a list of layers.
    See https://discuss.pytorch.org/t/runtimeerror-expected-object-of-backend-cpu-but-got-backend-cuda-for-argument-4-mat1/54745/2
    and https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
    """

    dropout = nn.Dropout(p=dropout_rate)

    activation = load_activation_function(activation_function)

    # create input layer
    input_layer = nn.Linear(num_inputs, layers_spec[0])

    # create dynamic list of middle layers, followed by activation function and dropout
    if batch_norm:  # disable dropout when we're using batch norm
        layers = [
            (nn.Linear(layers_spec[i], layers_spec[i + 1]), nn.BatchNorm1d(num_features=layers_spec[i + 1]), activation)
            for i in range(len(layers_spec) - 1)]
    else:
        layers = [(nn.Linear(layers_spec[i], layers_spec[i + 1]), activation, dropout) for i in
                  range(len(layers_spec) - 1)]
    layers = [item for layer in layers for item in layer]

    # create output layer
    output_layer = nn.Linear(layers_spec[-1], num_outputs)

    # stitch all the layers together into a sequential module
    modules = [input_layer, activation] + layers + [output_layer]

    if final_activation:
        modules += [activation]

    return nn.Sequential(*modules)

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

class RingBuffer():
    """ Simple ring buffer of floats, for 'last n running average' stats """
    def __init__(self, size: int):
        self.size = size
        self.array = np.zeros(size)
        self._n_items = 0
    
    def add(self, item: float):
        self.array[self._n_items % self.size] = item
        self._n_items += 1
    
    def mean(self):
        """ Get mean of all values """
        if self._n_items == 0:
            return np.nan
        return np.mean(self.array[:self._n_items])

    def min(self):
        """ Get minimum value """
        if self._n_items == 0:
            return np.nan
        return np.min(self.array[:self._n_items])

    def max(self):
        """ Get maximum value """
        if self._n_items == 0:
            return np.nan
        return np.max(self.array[:self._n_items])

    def last(self):
        """ Get most recently added value """
        if self._n_items == 0:
            return np.nan
        return self.array[(self._n_items - 1) % self.size]
