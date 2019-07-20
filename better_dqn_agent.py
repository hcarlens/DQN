import torch


class DQNAgent:
    def __init__(self, learning_rate, discount_rate, num_inputs, num_neurons, num_outputs):
        self.discount_rate = discount_rate
        self.q_network = torch.nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_outputs)   
            )
        self.target_network = torch.nn.Sequential(
            nn.Linear(num_inputs, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_outputs)   
            )
        self.update_target_network()
        self.loss_fn = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def fit_batch(self, minibatch):
        """
        minibatch is a list of (observation, action, reward, next_observation, done) tuples
        """
        # turn list of tuples into tuples of multiple values
        observations, actions, rewards, next_observations, terminal_indicators = [*zip(*minibatch)]

        observations = torch.tensor(observations, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_observations = torch.tensor(next_observations, dtype=torch.float)
        terminal_indicators = torch.tensor(terminal_indicators, dtype=torch.long)

        # work out perceived value of next states
        batch_size = len(observations)
        next_state_values = torch.zeros(batch_size)

        # value is non-zero only if the current state isn't terminal
        next_state_values[1 - terminal_indicators] = self.target_network(next_observations[1 - terminal_indicators]).max(1)[0].detach()

        expected_state_action_values = rewards + self.discount_rate * next_state_values

        predicted_state_action_values = self.q_network(observations).gather(1, actions)

        self.loss_fn(expected_state_action_values, predicted_state_action_values.squeeze())


        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def act(self, observation):
        action = self.q_network(observation).max(1)[0].detach().numpy()
        return action