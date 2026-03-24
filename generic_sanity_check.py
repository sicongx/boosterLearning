import numpy as np
import pandas as pd
from booster_learning_aligned import BoosterLearning

REAL_DATA_PATH = "data/RLdata_for_RNN_01242024.csv"
DEMOGRAPHICS_PATH = "data/demographics.csv"
RNN_WEIGHTS_PATH = "data/rnn_weights_2_128_2000_1e-04.pth"

learner = BoosterLearning(device="cpu", seed=2024)

real_df = pd.read_csv(REAL_DATA_PATH)
demo_df = pd.read_csv(DEMOGRAPHICS_PATH)

summary = learner.preprocess_covid_for_generic(real_df, demo_df)
print("preprocess summary:", summary)
print("generic setup:", learner.summarize_covid_generic_setup())

learner.load_rnn(RNN_WEIGHTS_PATH, hidden_size=128, num_layers=2, dropout=0.2)

sim_df = learner.simulate_env(n=100, vax_cost=0.04, reward_type="log", policy_mode="observed")
print(sim_df.head())
print(sim_df.shape)

rl_out = learner.tabular_q_learning(vax_cost=0.04, reward_type="log", repeats_train_eval=1)
print("epoch_reward_list shape:", rl_out["epoch_reward_list"].shape)

learned = learner.eval_q_learning("evaltable", vax_cost=0.04, reward_type="log", epochs=1)
observed = learner.eval_q_learning("data", vax_cost=0.04, reward_type="log", epochs=1)
always = learner.eval_q_learning("all", vax_cost=0.04, reward_type="log", epochs=1)
never = learner.eval_q_learning("none", vax_cost=0.04, reward_type="log", epochs=1)

print("learned mean:", np.nanmean(learned))
print("observed mean:", np.nanmean(observed))
print("always mean:", np.nanmean(always))
print("never mean:", np.nanmean(never))