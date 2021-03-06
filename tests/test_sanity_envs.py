from pytorch_rl.dqn_agent import DQNAgent
from pytorch_rl.agent_trainer import Trainer
from pytorch_rl.sanity_envs import (SanityEnvV0, SanityEnvV1, SanityEnvV2,
                                                  SanityEnvV3, SanityEnvV4,
                                    SanityEnvV5, SanityEnvV6)
from pytorch_rl.memory_buffer import MemoryBuffer
from pytorch_rl.utils import loss_functions, optimisers
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

    dqn_agent = DQNAgent(learning_rate=5e-3,
                        discount_rate=0.99,
                        final_layer_neurons=8,
                        action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(8,),
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        gradient_clipping_value=True,
                        gradient_clipping_threshold=1,
                        target_update_steps = 10,
    )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
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

    dqn_agent = DQNAgent(learning_rate=5e-3,
                        discount_rate=0.99,
                        final_layer_neurons=8,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(8,),
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        gradient_clipping_value=True,
                        gradient_clipping_threshold=1,
                        target_update_steps=10,
                         )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
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

    dqn_agent = DQNAgent(learning_rate=5e-3,
                        discount_rate=0.99,
                        final_layer_neurons=8,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(8,),
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        gradient_clipping_value=True,
                        gradient_clipping_threshold=1,
                        target_update_steps=10,
                         )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=350)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 2'

def test_sanity_env_v3():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV3()

    dqn_agent = DQNAgent(learning_rate=5e-2,
                        discount_rate=0.99,
                        final_layer_neurons=8,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(8,),
                         random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        target_update_steps=10,
                         )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 3'

def test_sanity_env_v4():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV4()

    dqn_agent = DQNAgent(learning_rate=5e-2,
                        discount_rate=0.99,
                        final_layer_neurons=8,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(8,),
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        target_update_steps=10,
                         epsilon_decay_steps=100
                         )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=300)

    assert trainer.loss_values.max() < 1e-5, 'Loss is too high on sanity env 4'

def test_sanity_env_v5():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV5(10, correct_timestep=4)

    dqn_agent = DQNAgent(learning_rate=5e-3,
                        discount_rate=0.99,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(64,),
                         final_layer_neurons=64,
                        num_outputs=env.action_space.n,
                        random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        target_update_steps=10,
                         )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=1,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=500)

    assert trainer.loss_values.max() < 1e-2, f'Loss is too high on {env.name}'

def test_sanity_env_v6():
    seed_everything()
    loss_fn = loss_functions['mse']
    optimiser = optimisers['adam']
    env = SanityEnvV6(max_num_steps=3, correct_timestep=2)

    dqn_agent = DQNAgent(learning_rate=1e-2,
                        discount_rate=0.99,
                        final_layer_neurons=64,
                         action_space_dim=env.action_space.n,
                         observation_space_dim=env.observation_space.shape[0],
                         value_net_layer_spec=(64,),
                         random_seed=RANDOM_SEED,
                        loss_fn=loss_fn,
                        optimiser=optimiser,
                        cuda=False,
                        target_update_steps=10,
                         )

    trainer = Trainer(
        env=env,
        agent=dqn_agent,
        memory_buffer=MemoryBuffer(buffer_length=100),
        timestep_to_start_learning=20,
        batch_size=16,
        random_seed=RANDOM_SEED,
        max_num_steps=1000000,
        train_every_n_steps=2,
        write_to_tensorboard=False
    )
    trainer.run(num_episodes=500)

    assert trainer.loss_values.max() < 1e-5, f'Loss is too high on {env.name}'
