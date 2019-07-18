import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork:
    """
    Neural net as Q-function approximator. 
    TODO:
    - add support for deep networks
    - add support for multiple outputs (i.e. mask to avoid double forward pass)
    - duelling DQN
    """
    def __init__(self, num_inputs, num_neurons, learning_rate):
        self.learning_rate = learning_rate
        self.network = nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 1)   
            )
        self.loss_fn = nn.MSELoss(reduction='sum')
    
    def predict(self, input_values):
        predicted_value = self.network(torch.tensor(input_values, dtype=torch.float))
        return predicted_value
    
    def fit(self, input_values, output_values, verbose=None):
        """
        Takes a batch of inputs and outputs and updates the neural net's parameters.
        """
     
        # calculate predicted Q-values and loss
        y_pred = self.network(torch.tensor(input_values, dtype=torch.float))
        loss = self.loss_fn(y_pred, torch.tensor(np.expand_dims(output_values, axis=1), dtype=torch.float))

        
        # Zero the gradients before running the backward pass.
        self.network.zero_grad()
        
        loss.backward()
        
        with torch.no_grad():
            for param in self.network.parameters():
                param -= self.learning_rate * param.grad

    def get_weights(self):
        return self.network.state_dict()
    
    def set_weights(self, new_values):
        self.network.load_state_dict(new_values)
        
def build_model(learning_rate=0.00025):
    # define Q-network (two layers with 32 neurons each)
    # 4 inputs (one for each scalar observable) +1 to represent the action
    # todo: change to 4 inputs; 5 outputs to require fewer forward passes
#     return nn.Sequential(
#         nn.Linear(5, 128),
#         nn.ReLU(),
#         nn.Linear(128, 1)   
#     )
    return QNetwork(5, 128, learning_rate)