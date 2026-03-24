import numpy as np
import torch
import random
import pandas as pd
import helpers
import sys
import itertools

settings = int(sys.argv[1]) - 1

hidden_size_list = [64, 128]
num_stacked_layers_list = [1, 2]
lr_list = [1e-4, 1e-5]

hyperpara_list = list(itertools.product(hidden_size_list, num_stacked_layers_list, lr_list))
hyperpara = hyperpara_list[settings]
hidden_size = hyperpara[0]
num_stacked_layers = hyperpara[1]
lr = hyperpara[2]
num_epochs = 2000

rnn = helpers.rnn(16, 2, hidden_size, num_stacked_layers)
rnn.load_state_dict(torch.load("rnn/rnn_weights_{}_{}_{}_{:.0e}.pth".format(num_stacked_layers, hidden_size, num_epochs, lr), map_location = torch.device('cpu')))

RNN_data = pd.read_csv("data/RLdata_for_RNN_01242024.csv").drop(['Unnamed: 0'], axis = 1)
demographics = pd.read_csv("data/demographics.csv").drop(['Unnamed: 0'], axis = 1)

action_list = RNN_data[["id", "action"]].groupby(by = "id").aggregate(func = lambda x: [np.where(x)[0]])

exp = 0
random.seed(exp)
np.random.seed(exp)
torch.manual_seed(exp)
#agent1 = helpers.DQN(seed = seed, hidden = 256, device = "cpu")
n = len(demographics)
interval_length = 27

col_names = ['id', 'action', 'age', 'imm_baseline', 'numVax', 'gender', 'African American', 'Other', 'v5-9', 'v10-19', 'v20-49', 'v50', 
             'c1-2', 'c3-4', 'c5', 'delta', 'omicron', "sev_inf_next", "inf_next", "inf_this", "sev_inf_this"]

simulated_dataset = np.zeros((interval_length * n, len(col_names) - 1))
patids = np.empty(interval_length * n, dtype = "object")
count = 0

sample_idx = np.arange(n)
demographics_subset = np.array(demographics.iloc[sample_idx, :])
for i in range(n):
    patid, age, imm_baseline, gender, race, visitsCat, comCat = demographics_subset[i, 0], demographics_subset[i, 1], demographics_subset[i, 2], demographics_subset[i, 3], demographics_subset[i, 4:6], demographics_subset[i, 6:10], demographics_subset[i, 10:]
    env = helpers.vaccine_env(rnn, hidden_size, age, imm_baseline, gender, race, visitsCat, comCat)
    state = env.state
    done = False
    episodic_reward = 0
    num_vax = np.random.choice(np.arange(5), size = 1).item()
    num_inf = 0
    num_severe_inf = 0
    vaccine_pattern = helpers.generate_vaccine_pattern(num_vax, interval_length, action_list.iloc[sample_idx[i]][0])
    for t in range(interval_length):
        action = vaccine_pattern[t]
        next_state, reward, done = env.step(action)
        episodic_reward += reward
        num_inf = num_inf + env.nextMonthInf
        num_severe_inf = num_severe_inf + env.nextMonthSevereInf
        
        if t == 0:
            inf_this = 0
            severe_inf_this = 0
        else:
            inf_this = inf_next_prev
            severe_inf_this = severe_inf_next_prev
        data_point = np.concatenate((env.action_state_to_date[-1, :], np.array([env.nextMonthSevereInf, env.nextMonthInf, inf_this, severe_inf_this])))
        
        simulated_dataset[count, :] = data_point
        patids[count] = patid
        count = count + 1
        inf_next_prev = env.nextMonthInf
        severe_inf_next_prev = env.nextMonthSevereInf
        
        #agent1.add_to_replay_memory(state, action, reward, next_state, done)
        state = next_state
        if env.nextMonthSevereInf == 1:
            break

    if (i + 1) % 100 == 0:
        print("{} / {}".format(i + 1, n))

simulated_dataset = pd.DataFrame(simulated_dataset[:count, :], columns = col_names[1:])
patids = pd.Series(patids[:count], name = "id")
simulated_dataset = pd.concat((patids, simulated_dataset), axis = 1)
simulated_dataset.to_csv("simulated_data/one_stage_{}_{}_{}_{:.0e}_data_action_{}.csv".format(num_stacked_layers, hidden_size, num_epochs, lr, exp))

seed = exp
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#agent1 = helpers.DQN(seed = seed, hidden = 256, device = "cpu")
n = len(demographics)
interval_length = 27

col_names = ['id', 'action', 'age', 'imm_baseline', 'numVax', 'gender', 'African American', 'Other', 'v5-9', 'v10-19', 'v20-49', 'v50', 
             'c1-2', 'c3-4', 'c5', 'delta', 'omicron', "sev_inf_next", "inf_next", "inf_this", "sev_inf_this"]

simulated_dataset = np.zeros((interval_length * n, len(col_names) - 1))
patids = np.empty(interval_length * n, dtype = "object")
count = 0

sample_idx = np.arange(n)
demographics_subset = np.array(demographics.iloc[sample_idx, :])
for i in range(n):
    patid, age, imm_baseline, gender, race, visitsCat, comCat = demographics_subset[i, 0], demographics_subset[i, 1], demographics_subset[i, 2], demographics_subset[i, 3], demographics_subset[i, 4:6], demographics_subset[i, 6:10], demographics_subset[i, 10:]
    env = helpers.vaccine_env(rnn, hidden_size, age, imm_baseline, gender, race, visitsCat, comCat)
    state = env.state
    done = False
    episodic_reward = 0
    num_vax = np.random.choice(np.arange(5), size = 1).item()
    num_inf = 0
    num_severe_inf = 0
    vaccine_pattern = helpers.generate_vaccine_pattern(num_vax, interval_length)
    for t in range(interval_length):
        action = vaccine_pattern[t]
        next_state, reward, done = env.step(action)
        episodic_reward += reward
        num_inf = num_inf + env.nextMonthInf
        num_severe_inf = num_severe_inf + env.nextMonthSevereInf
        
        if t == 0:
            inf_this = 0
            severe_inf_this = 0
        else:
            inf_this = inf_next_prev
            severe_inf_this = severe_inf_next_prev
        data_point = np.concatenate((env.action_state_to_date[-1, :], np.array([env.nextMonthSevereInf, env.nextMonthInf, inf_this, severe_inf_this])))
        
        simulated_dataset[count, :] = data_point
        patids[count] = patid
        count = count + 1
        inf_next_prev = env.nextMonthInf
        severe_inf_next_prev = env.nextMonthSevereInf
        
        #agent1.add_to_replay_memory(state, action, reward, next_state, done)
        state = next_state
        if env.nextMonthSevereInf == 1:
            break

    if (i + 1) % 100 == 0:
        print("{} / {}".format(i + 1, n))

simulated_dataset = pd.DataFrame(simulated_dataset[:count, :], columns = col_names[1:])
patids = pd.Series(patids[:count], name = "id")
simulated_dataset = pd.concat((patids, simulated_dataset), axis = 1)
simulated_dataset.to_csv("simulated_data/one_stage_{}_{}_{}_{:.0e}_random_action_{}.csv".format(num_stacked_layers, hidden_size, num_epochs, lr, exp))