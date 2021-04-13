import torch
import torch.nn as nn

class QNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_neurons_0, num_neurons_1, num_outputs, duelling=True):
        super().__init__()
        self.linear0 = torch.nn.Linear(num_inputs, num_neurons_0)
        self.linear1 = torch.nn.Linear(num_neurons_0, num_neurons_1)
        self.duelling = duelling
        if self.duelling:
            self.linear2_value = torch.nn.Linear(num_neurons_1, num_outputs)
            self.linear2_advantage = torch.nn.Linear(num_neurons_1, num_outputs)
        else:
            self.linear2 = torch.nn.Linear(num_neurons_1, num_outputs)

    def forward(self, x):
        h0_relu = self.linear0(x).clamp(min=0)
        h1_relu = self.linear1(h0_relu).clamp(min=0)
        if self.duelling:
            values = self.linear2_value(h1_relu)
            advantages = self.linear2_advantage(h1_relu)
            qs = values + advantages - torch.max(advantages, dim=-1, keepdim=True)[0]
        else:
            qs = self.linear2(h1_relu)
        return qs

class DQNAgent:
    def __init__(self, learning_rate, discount_rate, num_inputs, num_neurons_0, num_neurons_1,
                 num_outputs, loss_fn, optimiser, random_seed=None, duelling=True,
                 gradient_clipping_value=None, gradient_clipping_threshold=None, gradient_clipping_norm=None, cuda=False):
        
        if random_seed is not None:
            self.seed(random_seed)

        self.discount_rate = discount_rate
        self.q_network = QNetwork(num_inputs, num_neurons_0, num_neurons_1, num_outputs, duelling=duelling)
        self.target_network = QNetwork(num_inputs, num_neurons_0, num_neurons_1, num_outputs, duelling=duelling)
        self.update_target_network()
        self.loss_fn = loss_fn()
        self.optimiser = optimiser(params=self.q_network.parameters(), lr= learning_rate)
        self.gradient_clipping_value = gradient_clipping_value
        self.gradient_clipping_norm = gradient_clipping_norm
        self.gradient_clipping_threshold = gradient_clipping_threshold
        if cuda:
            self.q_network.cuda()
            self.target_network.cuda()
            self.device = 'cuda'
        else:
            self.device: str = 'cpu'


    def seed(self, random_seed):
        # seed pytorch
        torch.manual_seed(random_seed)

    def fit_batch(self, minibatch):
        """ minibatch is a list of (observation, action, reward, next_observation, done) tuples """
        # turn list of tuples into tuples of multiple values
        observations, actions, rewards, next_observations, terminal_indicators = [
            *zip(*minibatch)
        ]

        observations = torch.tensor(observations, dtype=torch.float, device=self.device)
        actions = torch.tensor(list(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float, device=self.device)

        non_terminal_states = ~torch.tensor(terminal_indicators,
                                           dtype=torch.bool, device=self.device)

        # work out perceived value of next states
        batch_size = len(observations)
        next_state_values = torch.zeros(batch_size, device=self.device)

        # value is non-zero only if the current state isn't terminal
        # ! TODO: complement outside; rename to non_terminal_states
        if non_terminal_states.sum() > 0:
            next_state_values[non_terminal_states] = self.target_network(
                next_observations[non_terminal_states]).max(1)[0].detach()

        expected_state_action_values = rewards + self.discount_rate * next_state_values

        predicted_state_action_values = self.q_network(observations).gather(
            1, actions.unsqueeze(1))

        loss = self.loss_fn(expected_state_action_values,
                     predicted_state_action_values.squeeze())

        # optimise the model
        self.optimiser.zero_grad()
        loss.backward()

        # clip the gradients if we need to
        if self.gradient_clipping_value is not None:
            torch.nn.utils.clip_grad_value_(self.q_network.parameters(), self.gradient_clipping_threshold)
        if self.gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clipping_threshold)

        self.optimiser.step()

        return loss.detach().cpu().numpy(), predicted_state_action_values.detach()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, observation):
        q_values = self.q_network(torch.tensor(observation, dtype=torch.float, device=self.device))
        # print(f"q values: {q_values}")
        return q_values.max(0)[1].detach().cpu().numpy()