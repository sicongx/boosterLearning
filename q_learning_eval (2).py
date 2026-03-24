import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import helpers
from helpers import booster_env
import sys


random_mode = ["evaltable", "all", "none", "data"][3]
    
exp_id = int(sys.argv[1]) - 1
reward_type = ["linear", "log", "logprop", "prop"][3]
vax_cost = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100][exp_id]

if random_mode == "evaltable":
    lr_fill = "_1"
    if vax_cost >= 0.0001:
        q_table = np.load("tabular_q/table_reward{}_vc{}{}.npy".format(reward_type, vax_cost, lr_fill))
    else:
        q_table = np.load("tabular_q/table_reward{}_vc{:.0e}{}.npy".format(reward_type, vax_cost, lr_fill))

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

epochs = 5
n = len(demographics)
epoch_reward_list = np.zeros((epochs, n))
epoch_reward_list[:] = np.nan
for epoch in range(epochs): 
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
        
        if random_mode == "all":
            if (row_idx + 1) < interval_length:
                vaccine_pattern[(row_idx + 1):interval_length] = 0
                if (row_idx + 4) < interval_length:
                    vaccine_pattern[np.random.choice(np.arange(row_idx + 4, interval_length), size = 1)] = 1
            
        for t in range(row_idx, interval_length):
            if previous_booster == True:
                action = 0
            else:
                if t == row_idx:
                    action = int(vaccine_hist[row_idx])
                elif env.monthsLastVax <= 4:
                    action = 0
                else:
                    if random_mode == "evaltable":
                        action = np.argmax(q_table[env.age_cat, env.imm_baseline, env.monthsLastVax_cat, :]).item()
                    elif random_mode == "none":
                        action = 0
                    else:
                        action = int(vaccine_pattern[t])
    
                    if action == 1:
                        previous_booster = True
                
            next_state, next_tq_state, reward, done = env.step(action)
            episodic_reward += reward / (interval_length - row_idx)
            tq_state = next_tq_state.copy()
            if env.nextMonthSevereInf == 1:
                break
        epoch_reward_list[epoch, i] = episodic_reward
        
        #if (i + 1) % 100 == 0:
            #print("epoch: {}, {} / {}".format(epoch + 1, i + 1, n))
    #print("epoch {} ends, average episodic reward {}".format(epoch + 1, round(np.mean(epoch_reward_list[epoch, :]).item(), 4)))
    #np.save("tabular_q/random_{}_reward{}_vc{}.npy".format(random_mode, reward_type, vax_cost), epoch_reward_list)
    np.save("tabular_q/{}_reward{}_vc{}.npy".format(random_mode, reward_type, vax_cost), epoch_reward_list)