import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import helpers
from helpers import booster_env
import sys

class TQ:
    def __init__(self, table_dim = (5, 2, 3, 2), seed = None, track_cell = None, lr = 0.1, lr_min = 0.001):
        # The following are recommended hyper-parameters.
        
        # Initial learning rate: 0.1
        # Learning rate decay for each episode: 0.998
        # Minimum learning rate: 0.001
        # Initial epsilon for exploration: 0.5
        # Epsilon decay for each episode: 0.99
        
        self.q_table = np.zeros(table_dim)  # The Q table.                             
        self.learning_rate = lr  # Learning rate. 
        self.learning_rate_decay = 0.998  # You may decay the learning rate as the training proceeds.
        self.min_learning_rate = lr_min
        self.epsilon = 0.5  # For the epsilon-greedy exploration. 
        self.epsilon_decay = 0.99  # You may decay the epsilon as the training proceeds.
        self.track_cell = track_cell
        if self.track_cell is not None:
            self.trace = []
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.samples = []
        self.steps = 0
        
        self.restriction = 1

    def select_action(self, state, monthsLastVax, previous_booster):
        """
        This function returns an action for the agent to take.
        Args:
            state: the state in the current step
        Returns:
            action: the action that the agent plans to take in the current step
        """
        
        # Please complete codes for choosing an action given the current state
        """
        Hint: You may use epsilon-greedy for exploration. 
        With probability self.epsilon, choose an action uniformly at random; 
        Otherwise, choose a greedy action based on self.q_table.
        """
        if previous_booster:
            action = 0
        else:
            if monthsLastVax <= 4:
                self.restriction = 1
                action = 0
            else:
                self.restriction = 0
                rand_num = np.random.rand(1)
                if rand_num < self.epsilon:
                    action = np.random.choice([0, 1])
                else:
                    action = np.argmax(self.q_table[state[0], state[1], state[2], :])

        return action
    
    def optimal_action(self, state, monthsLastVax, previous_booster):
        """
        This function returns an action for the agent to take.
        Args:
            state: the state in the current step
        Returns:
            action: the action that the agent plans to take in the current step
        """
        
        # Please complete codes for choosing an action given the current state
        """
        Hint: You may use epsilon-greedy for exploration. 
        With probability self.epsilon, choose an action uniformly at random; 
        Otherwise, choose a greedy action based on self.q_table.
        """
        if previous_booster:
            action = 0
        else:
            if monthsLastVax <= 4:
                self.restriction = 1
                action = 0
            else:
                self.restriction = 0
                action = np.argmax(self.q_table[state[0], state[1], state[2], :])

        return action
    
    def train(self, cur_state, cur_action, reward, next_state):
        """
        This function is used for the update of the Q table
        Args:
            - cur_state: the current state
            - cur_action: the current action
            - reward: the reward received
            - next_state: the next state observed
            - `done=1` means that the agent reaches the terminal state (`next_state=6`) and the episode terminates;
              `done=0` means that the current episode does not terminate;
              `done=-1` means that the current episode reaches the maximum length and terminates.
              We set the maximum length of each episode to be 1000.
        """
        
        self.q_table[cur_state[0], cur_state[1], cur_state[2], cur_action] = ( 
            self.q_table[cur_state[0], cur_state[1], cur_state[2], cur_action] + 
            self.learning_rate * (reward + 0.99 * np.max(self.q_table[next_state[0], next_state[1], next_state[2], :]) - 
            self.q_table[cur_state[0], cur_state[1], cur_state[2], cur_action])
        )
        self.steps = self.steps + 1

        if self.steps % 5000 == 0:
            self.learning_rate = self.learning_rate * self.learning_rate_decay
            if self.learning_rate < self.min_learning_rate:
                self.learning_rate = self.min_learning_rate
            self.epsilon = self.epsilon * self.epsilon_decay

        if (self.track_cell is not None) and (steps % 100 == 0):
            self.add_to_trace()
        
    def add_to_trace(self):
        self.trace.append(self.q_table[self.track_cell])

    
exp_id = int(sys.argv[1]) - 1
reward_type = ["linear", "log", "logprop", "prop"][3]
vax_cost = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 50][exp_id]
if vax_cost >= 0.05:
    lr = 0.1
elif vax_cost >= 0.005:
    lr = 0.01
elif vax_cost >= 0.0005:
    lr = 0.001
else:
    lr = 0.0005
lr_min = lr / 1000

rnn = helpers.rnn(16, 2, 128, 2)
rnn.load_state_dict(torch.load("../v2/rnn/rnn_weights_2_128_2000_1e-04.pth", map_location = torch.device('cpu')))

RNN_data = pd.read_csv("../v2/data/RLdata_for_RNN_01242024.csv").drop(['Unnamed: 0'], axis = 1)
demographics = pd.read_csv("../v2/data/demographics.csv").drop(['Unnamed: 0'], axis = 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
RNN_data.loc[RNN_data['Age.FirstDose'] == '>89', 'Age.FirstDose'] = 90
RNN_data_demographics = RNN_data.drop_duplicates(subset = ["id"])
age = RNN_data_demographics["Age.FirstDose"].astype(int)
age_cat = pd.cut(age, bins = [0, 18, 30, 50, 65, 100], include_lowest = True, right = False)
age_dummies = pd.get_dummies(age_cat).iloc[:, 1:]
age_dummies.reset_index(inplace = True, drop = True)

demographics = pd.concat([demographics["id"], 
                          demographics["Age.FirstDose"],
                          age_dummies,
                          demographics.iloc[:, 2:]], axis = 1)
demographics.columns = ["id", "age", "a18-29", "a30-49", "a50-64", "a65", "imm_baseline", "gender", "African American", "Other",
                        "v5-9", "v10-19", "v20-49", "v50", "c1-2", "c3-4", "c5"]

action_list = RNN_data[["id", "action"]].groupby(by = "id").aggregate(func = lambda x: [np.where(x)[0]])


seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_eval = [1, 1]
repeats = 30
epochs_train_eval = np.tile(np.repeat([False, True], train_eval), repeats)
epochs = len(epochs_train_eval)

agent_TQ = TQ(table_dim = (5, 2, 3, 2), seed = seed, lr = lr, lr_min = lr_min)
n = len(demographics)
epoch_reward_list = np.zeros((epochs, n))
epoch_reward_list[:] = np.nan
for epoch in range(epochs):
    is_train = epochs_train_eval[epoch]
    interval_length = 27
    sample_idx_array = np.random.choice(demographics.shape[0], size = n)
    for i in range(n):
        sample_idx = sample_idx_array[i]
        vaccine_pattern = helpers.generate_vaccine_pattern(0, interval_length, action_list.iloc[sample_idx][0])
        if sum(vaccine_pattern) < 2:
            continue
        row_idx = np.where(vaccine_pattern)[0][1]
        patid, age, age_dummies, imm_baseline, gender, race, visits_dummies, com_dummies = (demographics.iloc[sample_idx, 0], 
                                                                                            demographics.iloc[sample_idx, 1], 
                                                                                            np.array(demographics.iloc[sample_idx, 2:6]), 
                                                                                            demographics.iloc[sample_idx, 6], 
                                                                                            demographics.iloc[sample_idx, 7], 
                                                                                            np.array(demographics.iloc[sample_idx, 8:10]), 
                                                                                            np.array(demographics.iloc[sample_idx, 10:14]), 
                                                                                            np.array(demographics.iloc[sample_idx, 14:]))
        vaccine_hist = vaccine_pattern[:(row_idx + 1)]
        variant = np.array(RNN_data[RNN_data["id"] == patid]["variant"][:row_idx + 1].replace({"none": 0, "delta": 1, "omicron": 2}))
        env = booster_env(rnn, 128, age, imm_baseline, gender, race, visits_dummies, com_dummies, variant, vaccine_hist, age_dummies, vax_cost, reward_type)
        tq_state = env.tq_state
        done = False
        episodic_reward = 0
        previous_booster = False
        for t in range(row_idx, interval_length):
            if t == row_idx:
                action = int(vaccine_hist[row_idx])
            else:
                if is_train:
                    action = int(agent_TQ.select_action(tq_state, env.monthsLastVax, previous_booster))
                else:
                    action = int(agent_TQ.optimal_action(tq_state, env.monthsLastVax, previous_booster))
                if action == 1:
                    previous_booster = True
            
            next_state, next_tq_state, reward, done = env.step(action)
            episodic_reward += reward / (interval_length - row_idx)
            
            if is_train:
                if t != row_idx:
                    agent_TQ.train(tq_state, action, reward, next_tq_state)
            
            tq_state = next_tq_state.copy()
            if env.nextMonthSevereInf == 1:
                break
        
        if is_train == False:
            epoch_reward_list[epoch, i] = episodic_reward
        
        #if (i + 1) % 100 == 0:
            #print("epoch: {}, {} / {}".format(epoch + 1, i + 1, n))
    #print("epoch {} ends, average episodic reward {}".format(epoch + 1, round(np.mean(epoch_reward_list[epoch, :]).item(), 4)))
    np.save("tabular_q/epoch_reward{}_vc{}.npy".format(reward_type, vax_cost), epoch_reward_list)