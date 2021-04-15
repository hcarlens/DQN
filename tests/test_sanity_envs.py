from pytorch_rl.dqn_agent import DQNAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.sanity_envs import (SanityEnvV0, SanityEnvV1, SanityEnvV2,
                                                  SanityEnvV3, SanityEnvV4)
from pytorch_rl.memory_buffer import MemoryBuffer
from pytorch_rl.utils import loss_functions, optimisers, create_sequential_model
import torch
import random
import numpy as np

RANDOM_SEED = 0

import logging
logging.getLogger().setLevel(logging.INFO)

def seed_everything():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def test_sanity_env_v0():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV0()

    dqn_net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=(8,), num_outputs=8,
                                  dropout_rate=0, activation_function='relu', final_activation=True)
    dqn_agent = DQNAgent(learning_rate=5e-3,
                        discount_rate=0.99,
                        main_net=dqn_net,
                        final_layer_neurons=8,
                        num_outputs=env.action_space.n,
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        gradient_clipping_value=True,
                        gradient_clipping_threshold=1
                        )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        start_epsilon=1,
        timestep_to_start_learning=20,
        batch_size=16,
        target_update_steps=10,
        epsilon_decay_rate=0.5,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 0'

def test_sanity_env_v1():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV1()

    dqn_net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=(8,), num_outputs=8,
                                      dropout_rate=0, activation_function='relu', final_activation=True)
    dqn_agent = DQNAgent(learning_rate=5e-3,
                        discount_rate=0.99,
                        main_net=dqn_net,
                        final_layer_neurons=8,
                        num_outputs=env.action_space.n,
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        gradient_clipping_value=True,
                        gradient_clipping_threshold=1
                        )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        start_epsilon=1,
        timestep_to_start_learning=20,
        batch_size=16,
        target_update_steps=10,
        epsilon_decay_rate=0.5,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 1'

def test_sanity_env_v2():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV2()

    dqn_net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=(8,), num_outputs=8,
                                      dropout_rate=0, activation_function='relu', final_activation=True)

    dqn_agent = DQNAgent(learning_rate=5e-2,
                        discount_rate=0.99,
                        main_net=dqn_net,
                        final_layer_neurons=8,
                        num_outputs=env.action_space.n,
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        gradient_clipping_value=True,
                        gradient_clipping_threshold=1
                        )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        start_epsilon=1,
        timestep_to_start_learning=20,
        batch_size=16,
        target_update_steps=10,
        epsilon_decay_rate=0.5,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 2'

def test_sanity_env_v3():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV3()

    dqn_net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=(8,), num_outputs=8,
                                      dropout_rate=0, activation_function='relu', final_activation=True)

    dqn_agent = DQNAgent(learning_rate=5e-2,
                        discount_rate=0.99,
                        main_net=dqn_net,
                        final_layer_neurons=8,
                        num_outputs=env.action_space.n,
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        start_epsilon=1,
        timestep_to_start_learning=20,
        batch_size=16,
        target_update_steps=10,
        epsilon_decay_rate=0.5,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 3'

def test_sanity_env_v4():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV4()
    dqn_net = create_sequential_model(num_inputs=env.observation_space.shape[0], layers_spec=(8,), num_outputs=8,
                                      dropout_rate=0, activation_function='relu', final_activation=True)

    dqn_agent = DQNAgent(learning_rate=5e-2,
                        discount_rate=0.99,
                        main_net=dqn_net,
                        final_layer_neurons=8,
                        num_outputs=env.action_space.n,
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        start_epsilon=1,
        timestep_to_start_learning=20,
        batch_size=16,
        target_update_steps=10,
        epsilon_decay_rate=0.5,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 4'
