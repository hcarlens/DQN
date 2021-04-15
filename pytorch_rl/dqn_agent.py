import torch
import logging

logging.getLogger().setLevel(logging.INFO)

class QNetwork(torch.nn.Module):
    def __init__(self, main_net: torch.nn.Module, final_layer_neurons: int, num_outputs: int, duelling: bool=True):
        super().__init__()
        self.main_net = main_net
        self.duelling = duelling
        if self.duelling:
            self.linear_value = torch.nn.Linear(final_layer_neurons, 1) # single value per state
            self.linear_advantage = torch.nn.Linear(final_layer_neurons, num_outputs) # single advantage per state-action
        else:
            self.linear = torch.nn.Linear(final_layer_neurons, num_outputs)

    def forward(self, x):
        main_net_output = self.main_net(x)
        if self.duelling:
            self.values = self.linear_value(main_net_output)
            self.advantages = self.linear_advantage(main_net_output)
            qs = self.values + self.advantages - torch.max(self.advantages, dim=-1, keepdim=True)[0]
        else:
            qs = self.linear(main_net_output)
        return qs

class DQNAgent:
    def __init__(self, learning_rate: float, discount_rate: float, main_net: torch.nn.Module, final_layer_neurons: int,
                 num_outputs: int, loss_fn, optimiser, random_seed: int = None, duelling: bool = True,
                 gradient_clipping_value=None, gradient_clipping_threshold=None, gradient_clipping_norm=None,
                 cuda: bool = False):
        
        if random_seed is not None:
            self.seed(random_seed)

        self.discount_rate = discount_rate
        self.q_network = QNetwork(main_net, final_layer_neurons, num_outputs, duelling=duelling)
        self.target_network = QNetwork(main_net, final_layer_neurons, num_outputs, duelling=duelling)
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
        info_dict = {'loss': loss.detach().cpu().numpy(), 'q_values': predicted_state_action_values.detach()}
        if hasattr(self.q_network, 'values'):
            info_dict['state_values'] = self.q_network.values
        if hasattr(self.q_network, 'advantages'):
            info_dict['action_advantages'] = self.q_network.advantages

        return info_dict


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, observation):
        q_values = self.q_network(torch.tensor(observation, dtype=torch.float, device=self.device))
        # print(f"q values: {q_values}")
        return q_values.max(0)[1].detach().cpu().numpy()