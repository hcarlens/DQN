""" Loosely based on the Stanford CS231n course: https://cs231n.github.io/ """

from abc import ABC, abstractmethod
import random
import numpy as np
import math


def mse_loss(y_true, y_pred):
    """ returns (loss, grad_loss) """
    return np.power(y_true - y_pred, 2), 2 * (y_pred - y_true)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Gate(ABC):
    def __init__(self, param_value=None):
        self.param_value = param_value if param_value else np.random.random()
        self.input_value = None
        self.param_gradient = None
        self.input_gradient = None

    @abstractmethod
    def forward_pass(self, input_value):
        pass

    @abstractmethod
    def backward_pass(self, output_gradient):
        pass

    def gradient_update(self, alpha):
        self.param_value -= alpha * self.param_gradient


class SigmoidGate(Gate):
    def forward_pass(self, input_value):
        self.input_value = input_value
        return sigmoid(input_value)

    def backward_pass(self, output_gradient):
        self.input_gradient = output_gradient * \
        sigmoid(x) * (1 - sigmoid(x))
        return self.input_gradient


class MSELossGate(Gate):
    def forward_pass(self, input_value_1, input_value_2):
        self.input_value_1 = input_value_1
        self.input_value_2 = input_value_2
        return np.mean(np.power(input_value_1 - input_value_2, 2))

    def backward_pass(self, output_gradient=1):
        """ returns (dL/dY_hat, dL/dY) """
        return output_gradient * (2 * self.input_value_1 -
                                  2 * self.input_value_2), (
                                      2 * self.input_value_2 -
                                      2 * self.input_value_1)


class Layer(Gate):
    def __init__(self, num_inputs: int, num_neurons: int):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weight_values = np.random.rand(num_neurons, num_inputs)
        self.bias_values = np.random.rand(num_neurons)
        self.input_values = None
        self.weight_gradients = None
        self.bias_gradients = None
        self.pre_activation_gradients = None
        self.input_gradients = None
        self.output_values = None

    def forward_pass(self, input_values):
        print(len(input_values))
        print(len(self.num_inputs))
        assert len(input_values) == self.num_inputs
        self.input_values = input_values
        self.pre_activation_values = self.weight_values @ self.input_values + self.bias_values
        self.output = sigmoid(self.pre_activation_values)
        return self.output

    def backward_pass(self, output_gradients):
        self.pre_activation_gradients = output_gradients * \
        sigmoid(self.pre_activation_values) * \
        (1 - sigmoid(self.pre_activation_values))
        self.bias_gradients = self.pre_activation_gradients
        self.weight_gradients = self.bias_gradients.reshape(
            self.num_neurons, 1) @ self.input_values.reshape(
                1, self.num_inputs)
        self.input_gradients = None  # TODO: pass gradients back

    def gradient_update(self, alpha):
        self.weight_values -= alpha * self.weight_gradients
        self.bias_values -= alpha * self.bias_gradients


if __name__ == '__main__':

    learning_rate = 0.1
    num_iterations = 10000
    threshold = 2
    num_inputs = 2
    num_neurons = 2
    np.random.seed(0)
    layer = Layer(num_inputs=num_inputs, num_neurons=num_neurons)
    loss_gate = MSELossGate()

    # loop through iterations
    for i in range(num_iterations + 1):
        # give some inputs
        x = np.random.random_integers(low=0, high=1, size=num_inputs)
        y_true = np.array([1, 0]) if (x[0] == 1 and x[1] == 0) else np.array(
            [0, 0])
        # make a prediction
        y_pred = layer.forward_pass(x)
        # measure loss & gradient of loss
        loss = loss_gate.forward_pass(y_pred, y_true)
        grad_loss = loss_gate.backward_pass()[0]
        # backprop
        layer.backward_pass(grad_loss)
        # gradient updates
        layer.gradient_update(learning_rate)
        # print loss periodically
        if i % 100 == 0:
            print(
                f'Iteration {i}. X: {x}, Y: {y_true}, Y^: {y_pred}. Loss: {loss}. W: {layer.weight_values}. B: {layer.bias_values}. '
            )
    for x0 in [0, 1]:
        for x1 in [0, 1]:
            x = np.array([x0, x1])
            y_true = np.array([1, 0]) if (x[0] == 1
                                          and x[1] == 0) else np.array([0, 0])
            y_pred = layer.forward_pass(x)
            print(f'X: {x}, Y: {y_true}, Y^: {y_pred}. ')