import torch
import logging
import numpy as np

from typing import Tuple

from torch.distributions.categorical import Categorical

from pytorch_rl.base_agent import BaseAgent
from pytorch_rl.utils import loss_functions, optimisers, create_sequential_model

logging.getLogger().setLevel(logging.INFO)


class ActorCriticAgent(BaseAgent):
    """
    Simple single-threaded actor/critic agent.
    Based loosely on OpenAI's simple pg example + Sutton and Barto.
        https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
    Unlike the simple pg example, this agent uses a bootstrapped value function and so is fully on-line.

    This probably shouldn't really work, since we're sampling off-policy from the memory buffer without any corrections.
    But it seems to do ok...
     """
    def __init__(self, discount_rate: float,
                 action_space_dim: int,
                 observation_space_dim: int,
                 policy_net_layers_spec: Tuple,
                 value_net_layers_spec: Tuple,
                 loss_fn, optimiser, random_seed: int = None, cuda: bool = False,
                 learning_rate: float = None,
                 policy_learning_rate: float = None, value_learning_rate: float = None,
            ):
        """ `learning_rate` is a default for the policy and value learning rates, if those aren't specified."""
        if random_seed is not None:
            self.seed(random_seed)

        self.name = 'AC'

        # fallback to default ( easier compatibility with DQN)
        policy_learning_rate = policy_learning_rate or learning_rate
        value_learning_rate = value_learning_rate or learning_rate

        self.discount_rate = discount_rate
        self.policy_net = create_sequential_model(num_inputs=observation_space_dim, layers_spec=policy_net_layers_spec,
                                             num_outputs=action_space_dim, dropout_rate=0, activation_function='relu', final_activation=False)
        self.value_net = create_sequential_model(num_inputs=observation_space_dim, layers_spec=value_net_layers_spec,
                                                  num_outputs=1, dropout_rate=0, activation_function='relu', final_activation=False)
        self.loss_fn = loss_fn()
        self.policy_optimiser = optimiser(params=self.policy_net.parameters(), lr=policy_learning_rate)
        self.value_optimiser = optimiser(params=self.value_net.parameters(), lr=value_learning_rate)
        if cuda:
            self.policy_net.cuda()
            self.value_net.cuda()
            self.device = 'cuda'
        else:
            self.device: str = 'cpu'

    def seed(self, random_seed):
        # seed pytorch
        torch.manual_seed(random_seed)

    def get_policy(self, observations: np.array):
        """ Turn an observation into a policy action distribution """
        logits = self.policy_net(observations)
        return Categorical(logits=logits)

    def compute_loss(self, observations: torch.tensor, actions: torch.tensor, weights: torch.tensor):
        """ Policy loss """
        logp = self.get_policy(observations).log_prob(actions)
        return -(logp * weights).mean()

    def compute_avg_entropy(self, observations: torch.tensor):
        """ Policy loss """
        return self.get_policy(observations).entropy()

    def act(self, observation: np.array, action_mask: np.array = None):
        """
        take in an observation, and return an action
        todo: add action masking.
        """
        if action_mask is not None:
            raise ValueError(f'This agent doesn\'t support action masking.')
        return self.get_policy(torch.as_tensor(observation, dtype=torch.float32)).sample().item()

    def fit_batch(self, minibatch: Tuple[Tuple]):
        """
            Based on DQN agent fitting, and adapted for state-value functions.
            1. Manipulate data into the right form.
            2. Calculate TD errors.
            3. Gradient descent.
            minibatch is a list of (observation, action, reward, next_observation, done) tuples
        """

        logging.debug('Starting to fit batch')
        # turn list of tuples into tuples of multiple values
        observations, actions, rewards, next_observations, next_action_masks, terminal_indicators = [
            *zip(*minibatch)
        ]

        logging.debug('Turning memories into tensors')
        # todo: review memory buffer implementation to improve efficiency and reduce type conversions
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
            next_state_values[non_terminal_states] = self.value_net(next_observations[non_terminal_states]).squeeze().detach()

        # todo: confirm that there is no gradient coming from this part
        target_state_values = rewards + self.discount_rate * next_state_values

        predicted_state_values = self.value_net(observations)

        deltas = (target_state_values - predicted_state_values.squeeze()).detach() # TD error for policy update

        value_loss = self.loss_fn(target_state_values, predicted_state_values.squeeze())

        logging.debug('Starting value net backward pass.')
        self.value_optimiser.zero_grad()
        value_loss.backward()
        self.value_optimiser.step()
        logging.debug('Finished value net backward pass.')


        policy_loss = self.compute_loss(observations, actions, deltas)

        logging.debug('Starting policy net backward pass.')
        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()
        logging.debug('Finished policy net backward pass.')



        info_dict = {
            'value_loss': value_loss.detach().cpu().numpy(),
            'loss': value_loss.detach().cpu().numpy(), # backwards compatibility
            'policy_loss': policy_loss.detach().cpu().numpy(),
            'policy_avg_entropy': self.compute_avg_entropy(observations).detach().cpu().numpy().mean().item()
        }

        return info_dict



