""" Test the q network convergence. """
import numpy as np
import torch
import random

from pytorch_rl.dqn_agent import QNetwork
from pytorch_rl.utils import create_sequential_model, loss_functions, optimisers
RANDOM_SEED = 0

def seed_everything():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def test_simple_q_network():
    seed_everything()

    training_steps = 1000
    learning_rate = 0.01

    main_net = create_sequential_model(layers_spec=(64, ), num_inputs=1, num_outputs=64, activation_function='relu', final_activation=True, dropout_rate=0)
    q_network = QNetwork(main_net=main_net, final_layer_neurons=64, num_outputs=2, duelling=False)

    loss_fn = loss_functions['mse']()
    optimiser = optimisers['adam'](params = q_network.parameters(), lr=learning_rate)


    examples = ((np.array([0]), np.array([0, -1])),
                (np.array([0.1]), np.array([0, -1])),
                (np.array([0.2]), np.array([0, 1])),
                (np.array([0.3]), np.array([0, -1])),
                (np.array([0.4]), np.array([0, -1])),
                )

    # x =
    x = torch.from_numpy(np.vstack([x_ for x_, y_ in examples]).astype(np.float32))
    y = torch.from_numpy(np.vstack([y_ for x_, y_ in examples]).astype(np.float32))

    for i in range(training_steps):
        y_pred = q_network(x)
        loss = loss_fn(y_pred, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss < 1e-3, 'Plain Q net doesn\'t converge!'

def test_dueling_q_network():
    seed_everything()
    training_steps = 500
    learning_rate = 0.05

    main_net = create_sequential_model(layers_spec=(64, ), num_inputs=1, num_outputs=64, activation_function='relu', final_activation=True, dropout_rate=0)
    q_network = QNetwork(main_net=main_net, final_layer_neurons=64, num_outputs=2, duelling=True)

    loss_fn = loss_functions['mse']()
    optimiser = optimisers['adam'](params = q_network.parameters(), lr=learning_rate)


    examples = ((np.array([0]), np.array([0, -1])),
                (np.array([0.1]), np.array([0, -1])),
                (np.array([0.2]), np.array([0, 1])),
                (np.array([0.3]), np.array([0, -1])),
                (np.array([0.4]), np.array([0, -1])),
                )

    # x =
    x = torch.from_numpy(np.vstack([x_ for x_, y_ in examples]).astype(np.float32))
    y = torch.from_numpy(np.vstack([y_ for x_, y_ in examples]).astype(np.float32))

    for i in range(training_steps):
        y_pred = q_network(x)
        loss = loss_fn(y_pred, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss < 1e-3, 'Dueling Q net doesn\'t converge!'


def test_q_network_with_actions():
    seed_everything()
    training_steps = 500
    learning_rate = 0.05

    main_net = create_sequential_model(layers_spec=(64, ), num_inputs=1, num_outputs=64, activation_function='relu', final_activation=True, dropout_rate=0)
    q_network = QNetwork(main_net=main_net, final_layer_neurons=64, num_outputs=2, duelling=True)

    loss_fn = loss_functions['mse']()
    optimiser = optimisers['adam'](params = q_network.parameters(), lr=learning_rate)


    examples = ((np.array([0]), np.array([0, -1])),
                (np.array([0.1]), np.array([0, -1])),
                (np.array([0.2]), np.array([0, 1])),
                (np.array([0.3]), np.array([0, -1])),
                (np.array([0.4]), np.array([0, -1])),
                )

    x = torch.from_numpy(np.vstack([x_ for x_, y_ in examples]).astype(np.float32))
    y = torch.from_numpy(np.vstack([y_ for x_, y_ in examples]).astype(np.float32))

    for i in range(training_steps):
        # pick some pretend 'actions' to optimise like we will in training
        random_actions = torch.from_numpy(np.random.randint(0, 2, 5))

        y_pred = q_network(x)
        action_preds = y_pred.gather(1, random_actions.unsqueeze(1))

        targets = y.gather(1, random_actions.unsqueeze(1))

        loss = loss_fn(action_preds, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss < 1e-3, 'Q net with actions doesn\'t converge!'
