import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import pandas as pd

class random_agent:
    def __init__(self):
        pass
    
    def optimal_action(self):
        return(np.random.choice([0, 1], size = 1))

"""
A simple Q-network class
"""
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Args:
            input_dim (int): state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer dimension (fully connected layer)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Returns a Q value
        Args:
            state (torch.Tensor): state, 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


"""
Agent class that implements the DQN algorithm
"""
class DQN:
    def __init__(self, state_dim = 22, seed = None, hidden = 256, lr = 1e-5, device = "cpu"):
        self.device = device
        self.output_dim = 2  # Output dimension of Q network, i.e., the number of possible actions
        self.dqn = QNetwork(state_dim, self.output_dim, hidden).to(self.device)  # Q network
        self.dqn_target = QNetwork(state_dim, self.output_dim, hidden).to(self.device)  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.batch_size = 64  # Batch size
        self.gamma = 0.99  # Discount factor
        self.eps = 0  # epsilon-greedy for exploration
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr = lr)  # optimizer for training
        self.replay_memory_buffer = deque(maxlen = 1000000)  # replay buffer
        self.steps = 0
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
    
    def select_action(self, state):
        """
        Returns an action for the agent to take during training process
        Args:
            state: a numpy array with size 4
        Returns:
            action: an integer, 0 or 1
        """
        
        # Please complete codes for choosing an action given the current state
        """
        Hint: You may use epsilon-greedy for exploration. 
        With probability self.eps, choose an action uniformly at random; 
        Otherwise, choose a greedy action based on the output of the Q network (self.dqn).
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        rand_num = np.random.rand(1)
        if rand_num < self.eps:
            action = np.random.choice(np.arange(2))
        else:
            Q_s = self.dqn.forward( torch.tensor(state, dtype = torch.float32).to(self.device) ).detach()
            action = torch.argmax(Q_s)
        ### END SOLUTION
        return int(action)

    def train(self):
        """
        Train the Q network
        Args:
            s0: current state, a numpy array with size 4
            a0: current action, 0 or 1
            r: reward
            s1: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        #self.add_to_replay_memory(s0, a0, r, s1, done)
        
        if self.steps % 5000 == 0:
            self.update_epsilon()
            self.target_update()
            
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        
        """
        state_batch: torch.Tensor with shape (self.batch_size, 4), a mini-batch of current states
        action_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of current actions
        reward_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of rewards
        next_state_batch: torch.Tensor with shape (self.batch_size, 4), a mini-batch of next states
        done_list: torch.Tensor with shape (self.batch_size, 1), a mini-batch of 0-1 integers, 
                   where 1 means the episode terminates for that sample;
                         0 means the episode does not terminate for that sample.
        """
        mini_batch = self.get_random_sample_from_replay_mem()
        state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
        action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).int().to(self.device)
        reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
        next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
        done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)

        # Please complete codes for updating the Q network self.dqn
        """
        Hint: You may use the above tensors: state_batch, action_batch, reward_batch, next_state_batch, done_list
              You may use self.dqn_target as your target Q network
              You may use self.loss_fn (or torch.nn.MSELoss()) as your loss function
              You may use self.optim as your optimizer for training the Q network
        """
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        action_batch = torch.tensor(action_batch, dtype = torch.int64)
        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        mask = (done_list.flatten() == 0)
        non_final_next_states = next_state_batch[mask, :]
        next_state_values = torch.zeros((self.batch_size, 1)).to(self.device)
        next_state_values[mask, :] = self.dqn_target.forward(non_final_next_states).max(1, keepdim = True)[0].detach()

        target_values = reward_batch + self.gamma * next_state_values
        loss = self.loss_fn(state_action_values, target_values)

        self.optim.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        ### END SOLUTION
        self.steps += 1

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        """
        Add samples to replay memory
        Args:
            state: current state, a numpy array with size 4
            action: current action, 0 or 1
            reward: reward
            next_state: next state, a numpy array with size 4
            done: done=True means that the episode terminates and done=False means that the episode does not terminate.
        """
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self):
        """
        Random samples from replay memory without replacement
        Returns a self.batch_size length list of unique elements chosen from the replay buffer.
        Returns:
            random_sample: a list with len=self.batch_size,
                           where each element is a tuple (state, action, reward, next_state, done)
        """
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample
    
    def update_epsilon(self):
        # Decay epsilon
        if self.eps >= 0.01:
            self.eps *= 0.95
    
    def target_update(self):
        # Update the target Q network (self.dqn_target) using the original Q network (self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        

class rnn(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = torch.nn.LSTM(input_size = input_size, 
                                  hidden_size = hidden_size, 
                                  num_layers = num_stacked_layers,
                                  batch_first = True,
                                  dropout = 0.2)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out
    
    def predict(self, x, h0, c0, cutoff = 0.5):
        prob = torch.sigmoid(self.forward(x, h0, c0))
        pred = (prob > cutoff) + 0
        return prob, pred


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.n = X.shape[0]
        self.t = X.shape[1]
        self.p = X.shape[2]
        self.output_size = y.shape[2]
        self.create_seq_mask()
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        X = self.X[index, :]
        y = self.y[index, :]
        seq_mask_X = self.seq_mask_X[index, :]
        seq_mask_y = self.seq_mask_y[index, :]
        return X, y, seq_mask_X, seq_mask_y
    
    def create_seq_mask(self):
        self.seq_mask_X = np.zeros((self.n, self.t, self.p))
        self.seq_mask_y = np.zeros((self.n, self.t, self.output_size))
        for i in range(self.n):
            self.seq_mask_X[i, :self.seq_length[i], :] = 1
            self.seq_mask_y[i, :self.seq_length[i], :] = 1
            

class booster_env():
    def __init__(self, rnn, hidden_size, age, imm_baseline, gender, race, visitsCat, comCat, variant, vaccine_hist, age_dummies = None, vax_cost = 0, reward_type = "log"):
        self.rnn = rnn
        self.hidden_size = hidden_size
        self.age = age
        self.age_dummies = age_dummies
        self.imm_baseline = imm_baseline
        self.gender = gender
        self.race = race + 0
        self.visitsCat = visitsCat
        self.comCat = comCat
        self.nextMonthInf = False
    
        hist_t = vaccine_hist.shape[0]
        variant_dummies = np.array(pd.get_dummies(
            pd.cut(variant, bins = [0, 1, 2, 3], include_lowest = True, right = False)
                                                    ) + 0)[:, 1:]
        self.action_state_to_date = np.hstack([vaccine_hist.reshape(-1, 1), 
                                               np.repeat(self.age, hist_t).reshape(-1, 1), 
                                               np.repeat(self.imm_baseline, hist_t).reshape(-1, 1), 
                                               np.cumsum(vaccine_hist).reshape(-1, 1), 
                                               np.repeat(gender, hist_t).reshape(-1, 1), 
                                               np.tile(self.race, (hist_t, 1)),
                                               np.tile(self.visitsCat, (hist_t, 1)), 
                                               np.tile(self.comCat, (hist_t, 1)), 
                                               variant_dummies]).astype(np.float32)
        
        self.state = self.action_state_to_date[-1, 1:]
        if self.age_dummies is not None:
            self.state = np.concatenate((self.age_dummies, self.state[1:])).astype(np.float32)
        
        self.numVax = np.cumsum(vaccine_hist)[-1]
        self.step_num = hist_t
        
        self.monthsLastVax = 0
        self.monthsLastVax_cat = 0
        self.monthsLastInf = -1

        self.vax_cost = vax_cost
        self.reward_type = reward_type
        
        if len(np.where(age_dummies == True)[0]) == 0:
            self.age_cat = 0
        else:
            self.age_cat = np.where(age_dummies == True)[0][0] + 1
        self.tq_state = [self.age_cat, imm_baseline, self.monthsLastVax_cat]
        #self.tq_state = [self.age_cat, imm_baseline]
    
    def step(self, action):
        done = False
        
        self.step_num += 1
        
        self.numVax = self.numVax + action
        if self.nextMonthInf == True:
            self.monthsLastInf = 0
        else:
            if self.monthsLastInf >= 0:
                self.monthsLastInf += 1

        if action:
            self.monthsLastVax = 1
        else:
            if self.monthsLastVax >= 0:
                self.monthsLastVax += 1
        
        if self.step_num <= 16:
            self.variant = 0
        elif self.step_num <= 22:
            self.variant = 1
        else:
            self.variant = 2
            
        self.monthsLastVax_cat = np.where( np.array(pd.get_dummies(
            pd.cut([self.monthsLastVax], bins = [0, 5, 7, 100], include_lowest = True, right = False)
                                                            ) + 0).reshape(-1) == 1)[0][0]
        self.tq_state[2] = self.monthsLastVax_cat
        
        self.variant_dummies = np.array(pd.get_dummies(
            pd.cut([self.variant], bins = [0, 1, 2, 3], include_lowest = True, right = False)
                                                            ) + 0).reshape(-1)[1:]
        
        self.state = np.concatenate(([self.age, self.imm_baseline, self.numVax, self.gender], 
                                      self.race, self.visitsCat, self.comCat, self.variant_dummies)).astype(np.float32)
        action_state = np.concatenate(([action], self.state)).astype(np.float32)
        self.action_state_to_date = np.vstack((self.action_state_to_date, action_state.reshape(1, -1)))     
        
        if self.age_dummies is not None:
            self.state = np.concatenate((self.age_dummies, [self.imm_baseline, self.numVax, self.gender], 
                                         self.race, self.visitsCat, self.comCat, self.variant_dummies)).astype(np.float32)
        
        with torch.no_grad():
            risk = self.rnn(torch.tensor(self.action_state_to_date).float().reshape((1, self.step_num, self.action_state_to_date.shape[1])), 
                            torch.zeros((self.rnn.num_stacked_layers, 1, self.hidden_size)).float(), 
                            torch.zeros((self.rnn.num_stacked_layers, 1, self.hidden_size)).float())
            risk = risk[0, -1, :]
            risk = 1 / (1 + np.exp(-risk))
            risk_inf = risk[1].item()
            risk_severe_inf = risk[0].item()
            self.nextMonthInf = np.random.choice([False, True], p = [1 - risk_inf, risk_inf], size = 1).item()
            self.nextMonthSevereInf = np.random.choice([False, True], p = [1 - risk_severe_inf, risk_severe_inf], size = 1).item()
            if self.reward_type == "linear":
                reward = - (risk_severe_inf + action * self.vax_cost) * 10000
            elif self.reward_type == "log":
                reward = - np.log(risk_severe_inf + action * self.vax_cost)
            elif self.reward_type == "logprop":
                reward = - np.log(risk_severe_inf * (1 + action * self.vax_cost))
            elif self.reward_type == "prop":
                reward = - risk_severe_inf * (1 + action * self.vax_cost) * 10000
            else:
                raise ValueError("unsupported reward_type")
        
        if self.nextMonthSevereInf:
            done = True
        
        return self.state, self.tq_state, reward, done


def generate_demographics():
    age = np.random.choice(np.arange(91), size = 1).item()
    gender = np.random.choice([0, 1], size = 1).item()
    race = np.random.choice(np.arange(4), size = 1).item()
    imm_baseline = np.random.choice([0, 1], size = 1).item()
    visits = np.random.choice(np.arange(235), size = 1).item()
    age_dummies = age2dummies(age)
    race_dummies = race2dummies(race)
    return age, gender, race, imm_baseline, visits, age_dummies, race_dummies


def generate_vaccine_pattern(num_vax, interval_length, from_real_data = None):
    vaccine_pattern = np.zeros(interval_length)
    if from_real_data is None:
        if num_vax > 0:
            if num_vax <= 2:
                when_to_receive = np.random.choice(np.arange(interval_length), size = num_vax)
            elif num_vax == 3:
                when_to_receive_vax3 = np.random.choice(np.arange(7, interval_length), size = 1)
                when_to_receive_vax12 = np.random.choice(np.arange(when_to_receive_vax3 - 5), size = 2, replace = False)
                when_to_receive = np.concatenate((when_to_receive_vax12, when_to_receive_vax3))
            elif num_vax == 4:
                when_to_receive_vax4 = np.random.choice(np.arange(13, interval_length), size = 1)
                when_to_receive_vax3 = np.random.choice(np.arange(7, when_to_receive_vax4 - 5), size = 1)
                when_to_receive_vax12 = np.random.choice(np.arange(when_to_receive_vax3 - 5), size = 2, replace = False)
                when_to_receive = np.concatenate((when_to_receive_vax12, when_to_receive_vax3, when_to_receive_vax4))
            vaccine_pattern[when_to_receive] = 1
    else:
        vaccine_pattern[from_real_data] = 1
    return vaccine_pattern


def rate_by_month(df):
    grouped_data = df[["id", "inf_next", "sev_inf_next"]].groupby("id")
    n = len(grouped_data)
    inf_mat = np.empty((n, 27))
    sev_inf_mat = np.empty((n, 27))
    inf_mat[:] = np.nan
    sev_inf_mat[:] = np.nan
    for i, (pid, group) in enumerate(grouped_data):
        inf = group["inf_next"].values
        sev_inf = group["sev_inf_next"].values

        where_sev_inf = np.where(sev_inf)[0]
        if len(where_sev_inf) > 0:
            t = where_sev_inf[0] + 1
        else:
            t = len(inf)

        if t == 27:
            inf_mat[i, :] = inf
            sev_inf_mat[i, :] = sev_inf
        else:
            inf_mat[i, :t] = inf[:t]
            sev_inf_mat[i, :t] = sev_inf[:t]
    return np.nanmean(inf_mat, axis = 0), np.nanmean(sev_inf_mat, axis = 0)
    

def infection_transition_by_varaible(real_transition_count, simulated_transition_count, severe, variable, value):
    if severe:
        transition_list = ["00", "01"]
    else:
        transition_list = ["00", "01", "10", "11"]
    real = ( real_transition_count[(real_transition_count[variable] == value).mean(axis = 1) == 1]["01"].sum() / 
            real_transition_count[(real_transition_count[variable] == value).mean(axis = 1) == 1][transition_list].sum().sum() )
    simulated = ( simulated_transition_count[(simulated_transition_count[variable] == value).mean(axis = 1) == 1]["01"].sum() / 
                 simulated_transition_count[(simulated_transition_count[variable] == value).mean(axis = 1) == 1][transition_list].sum().sum() )
    return real, simulated


def convert_to_transition_probs(df):
    inf_this = list(df.inf_this.astype(np.int16))
    inf_next = list(df.inf_next.astype(np.int16))
    sev_inf_this = list(df.sev_inf_this.astype(np.int16))
    sev_inf_next = list(df.sev_inf_next.astype(np.int16))
    inf_this = [str(s) for s in inf_this]
    inf_next = [str(s) for s in inf_next]
    sev_inf_this = [str(s) for s in sev_inf_this]
    sev_inf_next = [str(s) for s in sev_inf_next]

    df['inf_transition'] = list(map(''.join, zip(inf_this, inf_next)))
    df['sev_inf_transition'] = list(map(''.join, zip(sev_inf_this, sev_inf_next)))

    index_cols = df.columns.values.tolist()[:-6]
    inf_transition = df.pivot_table(index = index_cols, columns = 'inf_transition', aggfunc = 'size', fill_value = 0)
    sev_inf_transition = df.pivot_table(index = index_cols, columns = 'sev_inf_transition', aggfunc = 'size', fill_value = 0)
    inf_transition_probs = inf_transition.div(inf_transition.sum(axis = 1), axis = 0).reset_index()
    sev_inf_transition_probs = sev_inf_transition.div(sev_inf_transition.sum(axis = 1), axis = 0).reset_index()
    
    return inf_transition_probs, sev_inf_transition_probs, inf_transition.reset_index(), sev_inf_transition.reset_index()