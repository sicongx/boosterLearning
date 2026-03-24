import sys
sys.path.append('/home/gxma/vaccine_rnn/TestMDP-master/test_func') # download https://github.com/RunzheStat/TestMDP
import pandas as pd
import numpy as np
import random

import importlib
import _core_test_fun as mdp

num_stacked_layers = 2
hidden_size = 128
lr = 1e-4
num_epochs = 2000
exp = 0

file_name = "../v2/simulated_data/one_stage_{}_{}_{}_{:.0e}_data_action_{}.csv".format(num_stacked_layers, hidden_size, num_epochs, lr, exp)
simulated_dataset = pd.read_csv(file_name, index_col = 0)

grouped = simulated_dataset.groupby('id')
XAT_list = []
for pid, group in grouped:
    X = np.array(group[['age', 'imm_baseline', 'numVax', 'sev_inf_next']])
    A = np.array(group['action']).reshape(-1, 1)
    XAT_list.append([X, A, len(A)])
    
T = [entry[2] for entry in XAT_list]

random.seed(test_idx)
np.random.seed(test_idx)

n_sample = 500


XAT_list_complete = [XAT_list[i] for i in np.where(np.array(T) == 27)[0]]
data = [[entry[0][:, 2:4], entry[1]] for entry in XAT_list_complete]

p_list = []
for i in range(100):
    idx = np.random.choice(len(data), size = n_sample)
    p = mdp.test([data[_] for _ in idx], J = 1, paras = [8, 5], n_trees = 200, B = 200)
    p_list.append(p)
    np.savetxt("p_{}_{}.txt".format(test_idx, n_sample), p_list)