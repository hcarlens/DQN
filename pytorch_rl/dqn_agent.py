import torch
import random
import logging
import numpy as np
import copy
from pytorch_rl.base_agent import BaseAgent
from pytorch_rl.utils import loss_functions, optimisers, create_sequential_model

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

class DQNAgent(BaseAgent):
    def __init__(self, learning_rate: float, discount_rate: float, main_net: torch.nn.Module, final_layer_neurons: int,
                 target_update_steps: int, num_outputs: int, loss_fn, optimiser, random_seed: int = None, duelling: bool = True,
                 gradient_clipping_value=None, gradient_clipping_threshold=None, gradient_clipping_norm=None,
                 cuda: bool = False, train_mode: bool = False, start_epsilon: float = 1, end_epsilon: float = 0.01,
                 epsilon_decay_steps: int = 10000, eval_mode: bool = False):

        if random_seed is not None:
            self.seed(random_seed)

        self.discount_rate = discount_rate
        self.q_network = QNetwork(main_net, final_layer_neurons, num_outputs, duelling=duelling)
        self.target_network = QNetwork(copy.deepcopy(main_net), final_layer_neurons, num_outputs, duelling=duelling)
        self.loss_fn = loss_fn()
        self.optimiser = optimiser(params=self.q_network.parameters(), lr= learning_rate)
        self.gradient_clipping_value = gradient_clipping_value
        self.gradient_clipping_norm = gradient_clipping_norm
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.train_mode: bool = train_mode  # if true, use epsilon-greedy exploration

        self.target_update_steps: int = target_update_steps  # how often to update the target net (every n backward passes)
        self.num_backward_passes: int = 0  # counter to know when to update target net
        self.num_target_net_updates: int = 0  # counter to know when to update target net
        self.num_training_steps: int = 0  # counter to know how many training steps we've taken

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.epsilon = start_epsilon

        self.name = 'DQN' # todo: make this more re-usable/ structured

        self.eval_mode = eval_mode # if true, use fully greedy policy (no epsilon-randomness)

        self.possible_actions = np.arange(num_outputs) # used for action sampling and masking

        self.update_target_network()

        if cuda:
            self.q_network.cuda()
            self.target_network.cuda()
            self.device = 'cuda'
        else:
            self.device: str = 'cpu'

    def seed(self, random_seed):
        """ Seed relevant libraries """
        torch.manual_seed(random_seed)
        random.seed(random_seed)

    def fit_batch(self, minibatch):
        """ minibatch is a list of (observation, action, reward, next_observation, done) tuples """
        # turn list of tuples into tuples of multiple values
        logging.debug('Starting to fit batch')
        observations, actions, rewards, next_observations, next_action_masks, terminal_indicators = [
            *zip(*minibatch)
        ]

        logging.debug('Turning memories into tensors')
        observations = torch.tensor(observations, dtype=torch.float, device=self.device)
        actions = torch.tensor(list(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float, device=self.device)
        logging.debug('Turned memories into tensors')

        non_terminal_states = ~torch.tensor(terminal_indicators,
                                           dtype=torch.bool, device=self.device)

        # work out perceived value of next states
        batch_size = len(observations)
        next_state_values = torch.zeros(batch_size, device=self.device)

        # value is non-zero only if the current state isn't terminal
        if non_terminal_states.sum() > 0:
            next_state_values[non_terminal_states] = self.target_network(
                next_observations[non_terminal_states]).max(1)[0].detach()

        expected_state_action_values = rewards + self.discount_rate * next_state_values

        predicted_state_action_values = self.q_network(observations).gather(
            1, actions.unsqueeze(1))

        loss = self.loss_fn(expected_state_action_values,
                     predicted_state_action_values.squeeze())

        logging.debug('Starting backward pass.')
        # optimise the model
        self.optimiser.zero_grad()
        loss.backward()
        logging.debug('Finished backward pass.')

        logging.debug('Clipping gradients.')
        # clip the gradients if we need to
        if self.gradient_clipping_value is not None:
            torch.nn.utils.clip_grad_value_(self.q_network.parameters(), self.gradient_clipping_threshold)
        if self.gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clipping_threshold)
        logging.debug('Finished clipping gradients.')

        self.optimiser.step()
        info_dict = {'loss': loss.detach().cpu().numpy(), 'q_values': predicted_state_action_values.detach()}
        if hasattr(self.q_network, 'values'):
            info_dict['state_values'] = self.q_network.values
        if hasattr(self.q_network, 'advantages'):
            info_dict['action_advantages'] = self.q_network.advantages

        self.num_backward_passes += 1
        logging.debug('Finished fitting batch')

        if self.num_backward_passes % self.target_update_steps == 0:
            # set the target network weights to be the same as the q-network ones every so often
            logging.debug('Updating target net...')
            self.update_target_network()
            logging.debug('Target net updated.')

        return info_dict


    def update_target_network(self):
        """ Bring the target network params in line with the main network params """
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.num_target_net_updates += 1

    def update_epsilon(self):
        """ Bring the target network params in line with the main network params """

        epsilon_progress = self.num_training_steps / self.epsilon_decay_steps
        if epsilon_progress > 1:
            self.epsilon = self.end_epsilon
        else: #  linear interpolation
            self.epsilon = self.start_epsilon * (1 - epsilon_progress) + self.end_epsilon * (epsilon_progress)

    def act(self, observation: np.array, action_mask: np.array = None):
        """ Act, using greedy policy (eval_mode = True) or epsilon-greedy (eval_mode = False) """
        if not self.eval_mode:
            self.update_epsilon()
            self.num_training_steps += 1

        if self.eval_mode or random.uniform(0, 1) > self.epsilon:  # deterministic/greedy actions
            logging.debug('Taking greedy action. ')
            if action_mask is not None:
                # we need to implement action masking
                mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                q_values = self.q_network(torch.tensor(observation, dtype=torch.float, device=self.device))

                # find the index of the max q value in the masked array of q_values
                max_masked_action_index = q_values[mask].max(0)[1]

                # find the equivalent index in the unmasked array
                max_unmasked_action_index = torch.arange(len(q_values))[mask][max_masked_action_index]

                return max_unmasked_action_index.item()
            else:
                # straightforward max over q-values
                q_values = self.q_network(torch.tensor(observation, dtype=torch.float, device=self.device))
                return q_values.max(0)[1].detach().cpu().numpy().item()
        else:  # random actions
            logging.debug('Taking random action. ')
            if action_mask is not None:
                return np.random.choice(self.possible_actions[action_mask == 1]).item()
            else:
                return np.random.choice(self.possible_actions).item()