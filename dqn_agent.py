import pandas as pd
import numpy as np

class DQNAgent:
    def __init__(self, network_generator, discount_rate):
        self.discount_rate = discount_rate
        self.q_network = network_generator()
        self.target_network = network_generator()
        self.update_target_network()
        
    def fit_batch(self, minibatch):
        """
        minibatch is a list of (observation, action, reward, next_observation, done) tuples
        """
        

        minibatch = pd.DataFrame(minibatch, 
                    columns=['observation', 'action', 'reward', 'next_state', 'done'])

        # get max q values for the next state
        # todo: clean this up! Review and compare against baseline implementations
        # Can probably do this just with tensors rather than going via a pandas dataframe
        # This might be useful: https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Double%20DQN%20Solution.ipynb
        inputs = pd.DataFrame(minibatch.next_state.tolist())
        inputs['zeros'] = 0
        inputs['ones'] = 1
        next_value_0 = self.target_network.predict(inputs[[0,1,2,3,'zeros']])
        next_value_1 = self.target_network.predict(inputs[[0,1,2,3,'ones']])

        # calculate target q values; set future values to 0 for terminal states
        minibatch['next_state_max_q'] = np.maximum(next_value_0, next_value_1)
        minibatch.loc[minibatch.done, 'next_state_max_q'] = 0
        minibatch['target_q_value'] = minibatch.reward + self.discount_rate * minibatch.next_state_max_q

        nn_inputs = pd.DataFrame(minibatch.observation.tolist())
        nn_inputs['action'] = minibatch.action

        q_values =  [np.array(x) for x in minibatch.target_q_value.tolist()]
        self.q_network.fit(np.array(nn_inputs.values), np.array(q_values), verbose=0)
        
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
        
    def act(self,observation):
        inputs = np.array([np.append(observation,0), np.append(observation,1)])
        action = np.argmax(self.q_network.predict(inputs))
        return action