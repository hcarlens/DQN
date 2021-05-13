import torch
import logging
import numpy as np

from torch.distributions.categorical import Categorical

from pytorch_rl.base_agent import BaseAgent

logging.getLogger().setLevel(logging.INFO)


class VPGAgent(BaseAgent):
    """
    Vanilla Policy Gradient agent.
    This is a Monte Carlo agent and isn't supported by the normal agent trainer. (since it requires full trajectories)
    Based on OpenAI's simple pg example:
        https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
     """
    def __init__(self, learning_rate: float, discount_rate: float, policy_net: torch.nn.Module,
                 optimiser, random_seed: int = None, cuda: bool = False):
        
        if random_seed is not None:
            self.seed(random_seed)

        self.name = 'VPG' # todo: make this more re-usable

        self.discount_rate = discount_rate
        self.policy_net = policy_net
        self.optimiser = optimiser(params=self.policy_net.parameters(), lr= learning_rate)
        if cuda:
            self.policy_net.cuda()
            self.device = 'cuda'
        else:
            self.device: str = 'cpu'

    def seed(self, random_seed):
        # seed pytorch
        torch.manual_seed(random_seed)

    def get_policy(self, observations: np.array):
        """ Turn an observation into a policy distribution """
        logits = self.policy_net(observations)
        return Categorical(logits=logits)

    def compute_loss(self, observations: torch.tensor, actions: torch.tensor, weights: torch.tensor):
        logp = self.get_policy(observations).log_prob(actions)
        return -(logp * weights).mean()

    def act(self, observation: np.array, action_mask: np.array = None):
        """
        take in an observation, and return an action
        todo: add action masking.
        """
        if action_mask is not None:
            raise ValueError(f'This agent doesn\'t support action masking.')
        return self.get_policy(torch.as_tensor(observation, dtype=torch.float32)).sample().item()


    def fit_batch(self, observations, actions, weights):
        logging.debug('Starting to fit batch')

        self.optimiser.zero_grad()
        loss = self.compute_loss(torch.as_tensor(observations, dtype=torch.float32),
                                       torch.as_tensor(actions, dtype=torch.int32),
                                       torch.as_tensor(weights, dtype=torch.int32))

        logging.debug('Starting backward pass.')
        # optimise the model
        loss.backward()
        logging.debug('Finished backward pass.')

        self.optimiser.step()
        info_dict = {'loss': loss.detach().cpu().numpy().item()}

        logging.debug('Finished fitting batch')
        return info_dict

