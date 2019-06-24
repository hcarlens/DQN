from dqn_agent import DQNAgent
from nn import Layer, MSELossGate
import logging

class QNetwork:
    """
    Neural net as Q-function approximator. 
    TODO:
    - add support for deep networks
    - add support for multiple outputs
    """
    def __init__(self, num_inputs, num_neurons, learning_rate):
        self.layer = Layer(num_inputs=num_inputs, num_neurons=num_neurons)
        self.loss_gate = MSELossGate()
        self.learning_rate = learning_rate
    
    def predict(self, input_values):
        return self.layer.forward_pass(input_values)
    
    def fit(self, input_values, output_values, verbose=None):
        """
        Takes a batch of inputs and outputs and updates the neural net's parameters.
        TODO: implement!
        """
        if verbose is not None:
            logging.warning("Verbose parameter passed to fit method - this does not do anything and is only implemented for Keras compatibility. ")
        raise NotImplementedError("Fit function isn't implemented yet!")
        
        #TODO: fix this!
        logging.warning("Current implementation only fits on first value of batch (for testing purposes)!")
        input_value = input_values[0]
        output_value = output_values[0]
        
        predicted_output = self.layer.forward_pass(input_value)
        loss = self.loss_gate.forward_pass(predicted_output, output_value)

        grad_loss = self.loss_gate.backward_pass()[0]
        # backprop
        self.layer.backward_pass(grad_loss)
        self.layer.gradient_update(self.learning_rate)
    
    def get_weights(self):
        return self.layer.weight_values.copy(), self.layer.bias_values.copy()
    
    def set_weights(self, new_values):
        self.layer.weight_values = new_values[0]
        self.layer.bias_values = new_values[1]