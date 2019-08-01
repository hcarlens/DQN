import torch
import torch.nn as nn


class QNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_neurons, num_outputs, duelling=True):
        super().__init__()
        self.linear1 = torch.nn.Linear(num_inputs, num_neurons)
        self.duelling = duelling
        if self.duelling:
            self.linear2_value = torch.nn.Linear(num_neurons, num_outputs)
            self.linear2_advantage = torch.nn.Linear(num_neurons, num_outputs)
        else:
            self.linear2 = torch.nn.Linear(num_neurons, num_outputs)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        if self.duelling:
            values = self.linear2_value(h_relu)
            advantages = self.linear2_advantage(h_relu)
            qs = values + advantages - torch.max(advantages, dim=-1, keepdim=True)[0]
        else:
            qs = self.linear2(h_relu)
        return qs

class DQNAgent:
    def __init__(self, learning_rate, discount_rate, num_inputs, num_neurons,
                 num_outputs, random_seed=None, duelling=True, loss_type='mse', optimiser='adam'):
        
        if random_seed is not None:
            self.seed(random_seed)

        self.discount_rate = discount_rate
        self.q_network = QNetwork(num_inputs, num_neurons, num_outputs, duelling=duelling)
        self.target_network = QNetwork(num_inputs, num_neurons, num_outputs, duelling=duelling)
        self.update_target_network()

        if loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = torch.nn.SmoothL1Loss()

        if optimiser == 'adam':
            self.optimiser = torch.optim.Adam(self.q_network.parameters(), learning_rate)
        elif optimiser == 'rmsprop':
            self.optimiser = torch.optim.RMSprop(self.q_network.parameters(), learning_rate)
        elif optimiser == 'sgd':
            self.optimiser = torch.optim.SGD(self.q_network.parameters(), learning_rate)
        elif optimiser == 'sgd_momentum':
            self.optimiser = torch.optim.SGD(self.q_network.parameters(), learning_rate, momentum=0.9)

    def seed(self, random_seed):
        # seed pytorch
        torch.manual_seed(random_seed)

    def fit_batch(self, minibatch):
        """
        minibatch is a list of (observation, action, reward, next_observation, done) tuples
        """
        # turn list of tuples into tuples of multiple values
        observations, actions, rewards, next_observations, terminal_indicators = [
            *zip(*minibatch)
        ]

        observations = torch.tensor(observations, dtype=torch.float)
        actions = torch.tensor(list(actions), dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_observations = torch.tensor(next_observations, dtype=torch.float)

        # NOTE: for some reason we can use these as indices when uint8, but not when long
        terminal_indicators = torch.tensor(terminal_indicators,
                                           dtype=torch.uint8)

        # work out perceived value of next states
        batch_size = len(observations)
        next_state_values = torch.zeros(batch_size)

        # value is non-zero only if the current state isn't terminal
        next_state_values[1 - terminal_indicators] = self.target_network(
            next_observations[1 - terminal_indicators]).max(1)[0].detach()

        expected_state_action_values = rewards + self.discount_rate * next_state_values

        predicted_state_action_values = self.q_network(observations).gather(
            1, actions.unsqueeze(1))

        loss = self.loss_fn(expected_state_action_values,
                     predicted_state_action_values.squeeze())

        # optimise the model
        self.optimiser.zero_grad()
        loss.backward()

        # for param in self.q_network.parameters():
        #     param.grad.data.clamp_(-1,1)
        self.optimiser.step()

        return loss.detach().numpy(), predicted_state_action_values.detach()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, observation):
        q_values = self.q_network(torch.tensor(observation, dtype=torch.float))
        # print(f"q values: {q_values}")
        return q_values.max(0)[1].detach().numpy()