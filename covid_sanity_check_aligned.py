
from __future__ import annotations

import numpy as np
import pandas as pd

from booster_learning_aligned import BoosterLearning


REAL_DATA_PATH = "data/RLdata_for_RNN_01242024.csv"
DEMOGRAPHICS_PATH = "data/demographics.csv"
COVARIATES_PATH = "data/covariates_rnn.npy"
OUTCOMES_PATH = "data/outcomes_rnn.npy"
SEQ_LENGTH_PATH = "data/seq_length.npy"
TRUE_INF_PATH = "data/true_inf_by_month.txt"
TRUE_SEV_INF_PATH = "data/true_sev_inf_by_month.txt"

RNN_WEIGHTS_PATH = "data/rnn_weights_2_128_2000_1e-04.pth"
SEED = 2024
VAX_COST = 0.04


def main():
    learner = BoosterLearning(device="cpu", seed=SEED)

    # faithful COVID reproducibility path
    learner.load_rnn_arrays(
        np.load(COVARIATES_PATH),
        np.load(OUTCOMES_PATH),
        np.load(SEQ_LENGTH_PATH),
    )
    learner.load_covid_data(
        pd.read_csv(REAL_DATA_PATH),
        pd.read_csv(DEMOGRAPHICS_PATH),
    )

    if RNN_WEIGHTS_PATH is None:
        learner.train_rnn(
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            epochs=2000,
            lr=1e-4,
            batch_size=32,
            verbose_every=100,
        )
    else:
        learner.load_rnn(RNN_WEIGHTS_PATH, hidden_size=128, num_layers=2, dropout=0.2)

    rl_out = learner.tabular_q_learning_covid(
        vax_cost=VAX_COST,
        reward_type="log",
        repeats_train_eval=30,
    )

    learned = learner.eval_covid_policy("evaltable", vax_cost=VAX_COST, reward_type="log", epochs=5)
    observed = learner.eval_covid_policy("data", vax_cost=VAX_COST, reward_type="log", epochs=5)
    always = learner.eval_covid_policy("all", vax_cost=VAX_COST, reward_type="log", epochs=5)
    never = learner.eval_covid_policy("none", vax_cost=VAX_COST, reward_type="log", epochs=5)

    print("\n===== Reward summary =====")
    print(f"table  : {np.nanmean(learned):.6f}")
    print(f"data   : {np.nanmean(observed):.6f}")
    print(f"all    : {np.nanmean(always):.6f}")
    print(f"none   : {np.nanmean(never):.6f}")

    print("\n===== Training/eval reward array =====")
    print(rl_out["epoch_reward_list"].shape)

    true_inf = np.loadtxt(TRUE_INF_PATH)
    true_sev = np.loadtxt(TRUE_SEV_INF_PATH)
    print("\n===== Real marginal rates =====")
    print("inf first 5 months:", true_inf[:5])
    print("sev first 5 months:", true_sev[:5])


if __name__ == "__main__":
    main()
