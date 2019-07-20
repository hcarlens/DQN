import numpy as np
import random

random.seed(0)


class Trainer:
    """
    # todo: add deterministic mode
    """
    def __init__(self, env, agent, memory_buffer, start_epsilon, timestep_to_start_learning = 1000):
        self.env = env
        self.total_steps = 0
        self.episode_lengths = []
        self.buffer_sizes = []
        self.zero_action_pcts = []
        self.epsilon = start_epsilon
        self.agent = agent
        self.memory_buffer = memory_buffer
        
        self.num_episodes = 500
        self.max_num_steps = 200
        self.end_epsilon = 0.01
        self.epsilon_decay_rate = 0.99
        self.batch_size = 32
        self.optimizer_learning_rate = 0.00025
        self.buffer_length = 50000
        self.target_update_steps = 1000
        self.timestep_to_start_learning = timestep_to_start_learning
        
        print('Trainer initialised')
        
    def run(self, num_episodes):
        # run through episodes
        for e in range(num_episodes):
            observation = self.env.reset()
            current_actions = []

            for t in range(self.max_num_steps):
                # set the target network weights to be the same as the q-network ones every so often
                if self.total_steps % self.target_update_steps == 0:
                    self.agent.update_target_network()
                    print('Target network updated. ')
                                
                # with probability epsilon, choose a random action
                # otherwise use Q-network to pick action 
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(observation).item()

                next_observation, reward, done, info = self.env.step(action)
                current_actions.append(action)

                # add memory to buffer
                memory = (observation, action, reward, next_observation, done)
                self.memory_buffer.add_memory(memory)
                
                if self.total_steps > self.timestep_to_start_learning:
                    # sample a minibatch of experiences and update q-network
                    minibatch = self.memory_buffer.sample_minibatch(self.batch_size)

                    self.agent.fit_batch(minibatch)

                observation = next_observation
                self.total_steps += 1
                if done or t == self.max_num_steps - 1:
                    break 

            self.episode_lengths.append(t)
            self.buffer_sizes.append(self.memory_buffer.current_length)
            self.zero_action_pcts.append(sum(current_actions) / len(current_actions))

            if e % 10 == 0:
                print(f"Episode {e} finished after {t+1} timesteps. 100 ep running avg {np.floor(np.average(self.episode_lengths[-100:]))}. Epsilon {self.epsilon:.2f}. Buffer length: {self.buffer_sizes[-1]}. Zero actions: {self.zero_action_pcts[-1]:.2f}")

            # decrease epsilon value
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.end_epsilon)