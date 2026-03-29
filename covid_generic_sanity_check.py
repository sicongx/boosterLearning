"""Sanity check / example runner for the generic BoosterLearning interface.

This script shows how to use the generic classes in `booster_learning_aligned.py`
with the COVID-19 booster example from the paper.

What it does
------------
1. Loads the original long-format COVID data used for the RNN / RL workflow.
2. Recreates the original 16-dimensional RNN covariates.
3. Wraps the COVID-specific logic as generic hooks:
   - reward_fn
   - action_constraint_fn
   - transition_fn
   - terminal_fn
   - episode_start_fn
   - policy callables for learned / data / all / none
4. Either loads the original trained RNN weights or optionally refits the RNN.
5. Trains tabular Q-learning through the new generic interface.
6. Evaluates four policies and prints a compact sanity-check summary.

The goal is not to put COVID logic back into the library. Instead, this file keeps
COVID-specific logic outside the library and demonstrates how the generic interface
can reproduce the paper's example workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import helpers
from booster_learning_aligned import MicrosimQLearner, TrajectoryDataset


# -----------------------------------------------------------------------------
# Constants matching the original COVID example
# -----------------------------------------------------------------------------
INTERVAL_LENGTH = 27
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.2
RNN_EPOCHS = 2000
RNN_LR = 1e-4

RNN_COVARIATE_COLS = [
    "action",
    "Age.FirstDose",
    "imm_baseline",
    "numVax",
    "gender",
    "African American",
    "Other",
    "v5-9",
    "v10-19",
    "v20-49",
    "v50",
    "c1-2",
    "c3-4",
    "c5",
    "delta",
    "omicron",
]
RNN_OUTCOME_COLS = ["sev_inf_next", "inf_next"]
RL_STATE_COLS = ["age_cat", "imm_baseline", "months_since_vax_cat"]
TIME_SINCE_ACTION_BINS = [0, 5, 7, 1000001]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def _safe_drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.startswith("Unnamed:")]
    return df.loc[:, cols].copy()


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in COVID long-format data: {missing}")


def load_covid_long_format(rnn_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and transform the original RLdata_for_RNN csv into generic long format.

    Returns
    -------
    long_df : pd.DataFrame
        Full patient-month trajectories for policy learning.
    raw_df : pd.DataFrame
        Cleaned original file, useful for reporting real marginal rates.
    """
    raw = pd.read_csv(rnn_data_path)
    raw = _safe_drop_unnamed(raw)

    if "severe_infection_next" in raw.columns and "sev_inf_next" not in raw.columns:
        raw = raw.rename(columns={"severe_infection_next": "sev_inf_next"})

    required = [
        "id",
        "action",
        "Age.FirstDose",
        "Gender",
        "Race",
        "Visits",
        "windex",
        "imm_baseline",
        "numVax",
        "variant",
        "inf_next",
        "sev_inf_next",
    ]
    _require_columns(raw, required)

    raw = raw.copy()
    raw.loc[raw["Age.FirstDose"] == ">89", "Age.FirstDose"] = 90
    raw["Age.FirstDose"] = raw["Age.FirstDose"].astype(int)

    # Keep the original paper convention: severe infection implies infection.
    severe_idx = raw["sev_inf_next"].astype(int).values == 1
    raw.loc[severe_idx, "inf_next"] = 1

    # Month index within patient.
    raw = raw.sort_values(["id"]).reset_index(drop=True)
    raw["month_index"] = raw.groupby("id").cumcount()

    # Standardize age exactly the way create_rnn_data.ipynb does.
    age_unique = raw[["Age.FirstDose"]].drop_duplicates().sort_values("Age.FirstDose").reset_index(drop=True)
    scaler = StandardScaler()
    age_unique["normalized_age"] = scaler.fit_transform(age_unique[["Age.FirstDose"]])
    age_map = dict(zip(age_unique["Age.FirstDose"], age_unique["normalized_age"]))
    raw["Age.FirstDose_raw"] = raw["Age.FirstDose"].astype(int)
    raw["Age.FirstDose"] = raw["Age.FirstDose_raw"].map(age_map).astype(float)

    # Dummy variables exactly aligned with create_rnn_data.ipynb.
    gender_dummies = (raw["Gender"] == "M").astype(int).rename("gender")

    race_dummies = pd.get_dummies(raw["Race"])
    for col in ["Caucasian", "African American", "Other"]:
        if col not in race_dummies.columns:
            race_dummies[col] = 0
    race_dummies = race_dummies[["Caucasian", "African American", "Other"]]

    variant_dummies = pd.get_dummies(raw["variant"])
    for col in ["none", "delta", "omicron"]:
        if col not in variant_dummies.columns:
            variant_dummies[col] = 0
    variant_dummies = variant_dummies[["none", "delta", "omicron"]]

    visits_dummies = pd.get_dummies(
        pd.cut(raw["Visits"], bins=[0, 5, 10, 20, 50, 1000], include_lowest=True, right=False)
    )
    for col in ["[0, 5)", "[5, 10)", "[10, 20)", "[20, 50)", "[50, 1000)"]:
        if col not in visits_dummies.columns:
            visits_dummies[col] = 0
    visits_dummies = visits_dummies[["[0, 5)", "[5, 10)", "[10, 20)", "[20, 50)", "[50, 1000)"]]
    visits_dummies.columns = ["v0-4", "v5-9", "v10-19", "v20-49", "v50"]

    windex_dummies = pd.get_dummies(
        pd.cut(raw["windex"], bins=[0, 1, 3, 5, 100], include_lowest=True, right=False)
    )
    for col in ["[0, 1)", "[1, 3)", "[3, 5)", "[5, 100)"]:
        if col not in windex_dummies.columns:
            windex_dummies[col] = 0
    windex_dummies = windex_dummies[["[0, 1)", "[1, 3)", "[3, 5)", "[5, 100)"]]
    windex_dummies.columns = ["c0", "c1-2", "c3-4", "c5"]

    long_df = pd.concat(
        [
            raw[["id", "month_index", "action", "Age.FirstDose", "Age.FirstDose_raw", "imm_baseline", "numVax"]].copy(),
            gender_dummies,
            race_dummies[["African American", "Other"]].astype(int),
            visits_dummies[["v5-9", "v10-19", "v20-49", "v50"]].astype(int),
            windex_dummies[["c1-2", "c3-4", "c5"]].astype(int),
            variant_dummies[["delta", "omicron"]].astype(int),
            raw[["sev_inf_next", "inf_next"]].astype(int),
        ],
        axis=1,
    )

    # RL state 1: age category, identical bins as q_learning_table.py.
    age_cat = pd.cut(
        raw["Age.FirstDose_raw"].astype(int),
        bins=[0, 18, 30, 50, 65, 100],
        include_lowest=True,
        right=False,
        labels=False,
    )
    long_df["age_cat"] = age_cat.astype(int)

    # RL state 2: months since last vaccine category from observed history.
    months_since = np.zeros(len(long_df), dtype=int)
    for _, idx in long_df.groupby("id", sort=False).groups.items():
        idx = list(idx)
        months_last = 0
        for pos in idx:
            action = int(long_df.at[pos, "action"])
            if action == 1:
                months_last = 0
            else:
                months_last += 1
            months_since[pos] = months_last
    months_cat = pd.cut(
        months_since,
        bins=TIME_SINCE_ACTION_BINS,
        include_lowest=True,
        right=False,
        labels=False,
    )
    long_df["months_since_vax_cat"] = months_cat.astype(int)

    # Force the RL state mapping to include all 3 categories even if a small subset is loaded.
    extra_rows = []
    present = set(long_df["months_since_vax_cat"].unique().tolist())
    for cat in [0, 1, 2]:
        if cat not in present:
            dummy = long_df.iloc[[0]].copy()
            dummy["id"] = f"__state_map_dummy_{cat}__"
            dummy["month_index"] = 0
            dummy["action"] = 0
            dummy["months_since_vax_cat"] = cat
            dummy["sev_inf_next"] = 0
            dummy["inf_next"] = 0
            extra_rows.append(dummy)
    if extra_rows:
        long_df = pd.concat([long_df] + extra_rows, axis=0, ignore_index=True)

    return long_df, raw


# -----------------------------------------------------------------------------
# Generic hook functions for the COVID example
# -----------------------------------------------------------------------------

def covid_episode_start_fn(ctx: Dict[str, Any]) -> int:
    actions = np.asarray(ctx["actions"], dtype=int)
    pos = np.where(actions == 1)[0]
    if len(pos) < 2:
        return 0
    return int(pos[1])


@dataclass
class CovidHookBundle:
    reward_fn: Callable[[Dict[str, Any]], float]
    action_constraint_fn: Callable[[Dict[str, Any]], Sequence[int]]
    transition_fn: Callable[[Dict[str, Any]], np.ndarray]
    terminal_fn: Callable[[Dict[str, Any]], bool]


def build_covid_hooks(dataset: TrajectoryDataset, vax_cost: float, reward_type: str) -> CovidHookBundle:
    feature_idx = dataset.feature_col_index
    outcome_idx = dataset.outcome_col_index

    required_feature_cols = ["action", "numVax", "delta", "omicron"]
    missing = [c for c in required_feature_cols if c not in feature_idx]
    if missing:
        raise ValueError(f"COVID transition_fn requires columns {missing} in rnn_covariate_cols.")

    def reward_fn(ctx: Dict[str, Any]) -> float:
        risk = np.asarray(ctx["predicted_outcomes"], dtype=float)
        action = int(ctx["action"])
        risk_severe = float(risk[outcome_idx["sev_inf_next"]])
        if reward_type == "linear":
            return -((risk_severe + action * vax_cost) * 10000.0)
        if reward_type == "log":
            return -float(np.log(risk_severe + action * vax_cost))
        if reward_type == "logprop":
            return -float(np.log(risk_severe * (1.0 + action * vax_cost)))
        if reward_type == "prop":
            return -((risk_severe * (1.0 + action * vax_cost)) * 10000.0)
        raise ValueError("unsupported reward_type")

    def action_constraint_fn(ctx: Dict[str, Any]) -> Sequence[int]:
        env = ctx["env"]
        previous_booster = int(np.sum(env.action_history[env.history_len :]) > 0)
        if previous_booster:
            return [0]
        if int(env.time_since_action) <= 4:
            return [0]
        return [0, 1]

    def transition_fn(ctx: Dict[str, Any]) -> np.ndarray:
        env = ctx["env"]
        row = np.asarray(ctx["base_next_row"], dtype=np.float32).copy()
        action = int(ctx["action"])
        next_step = int(ctx["next_step"])

        if action == 1:
            env.time_since_action = 1
        else:
            env.time_since_action += 1

        row[feature_idx["action"]] = float(action)
        row[feature_idx["numVax"]] = float(np.sum(env.action_history))

        step_num = next_step + 1
        if step_num <= 16:
            delta, omicron = 0.0, 0.0
        elif step_num <= 22:
            delta, omicron = 1.0, 0.0
        else:
            delta, omicron = 0.0, 1.0
        row[feature_idx["delta"]] = delta
        row[feature_idx["omicron"]] = omicron
        return row.astype(np.float32)

    def terminal_fn(ctx: Dict[str, Any]) -> bool:
        env = ctx["env"]
        risk = np.asarray(ctx["predicted_outcomes"], dtype=float)
        risk_severe = float(risk[outcome_idx["sev_inf_next"]])
        risk_inf = float(risk[outcome_idx["inf_next"]])
        env.next_month_inf = bool(np.random.choice([False, True], p=[1.0 - risk_inf, risk_inf], size=1).item())
        env.next_month_severe_inf = bool(
            np.random.choice([False, True], p=[1.0 - risk_severe, risk_severe], size=1).item()
        )
        return bool(env.next_month_severe_inf)

    return CovidHookBundle(
        reward_fn=reward_fn,
        action_constraint_fn=action_constraint_fn,
        transition_fn=transition_fn,
        terminal_fn=terminal_fn,
    )


# -----------------------------------------------------------------------------
# Policy wrappers faithful to q_learning_eval.py
# -----------------------------------------------------------------------------

def _previous_booster_from_env(env: Any) -> bool:
    return bool(np.sum(env.action_history[env.history_len :]) > 0)


@dataclass
class CovidPolicies:
    learned: Callable[[Tuple[int, ...], Dict[str, Any]], int]
    data: Callable[[Tuple[int, ...], Dict[str, Any]], int]
    none: Callable[[Tuple[int, ...], Dict[str, Any]], int]
    all: Callable[[Tuple[int, ...], Dict[str, Any]], int]


def build_covid_policies(learner: MicrosimQLearner, seed: int = 2024) -> CovidPolicies:
    rng = np.random.default_rng(seed)

    def data_policy(state: Tuple[int, ...], ctx: Dict[str, Any]) -> int:
        env = ctx["env"]
        patient = ctx["patient"]
        t = int(ctx["t"])
        if t == patient.history_start_idx:
            return int(patient.observed_actions[t])
        if _previous_booster_from_env(env):
            return 0
        if int(env.time_since_action) <= 4:
            return 0
        action = int(patient.observed_actions[t])
        return action

    def none_policy(state: Tuple[int, ...], ctx: Dict[str, Any]) -> int:
        patient = ctx["patient"]
        t = int(ctx["t"])
        if t == patient.history_start_idx:
            return int(patient.observed_actions[t])
        return 0

    def all_policy(state: Tuple[int, ...], ctx: Dict[str, Any]) -> int:
        env = ctx["env"]
        patient = ctx["patient"]
        t = int(ctx["t"])
        if t == patient.history_start_idx:
            return int(patient.observed_actions[t])
        if _previous_booster_from_env(env):
            return 0
        if int(env.time_since_action) <= 4:
            return 0
        if not hasattr(env, "_all_policy_target_t"):
            start = patient.history_start_idx + 4
            if start < patient.rnn_inputs.shape[0]:
                env._all_policy_target_t = int(rng.choice(np.arange(start, patient.rnn_inputs.shape[0]), size=1).item())
            else:
                env._all_policy_target_t = None
        target_t = getattr(env, "_all_policy_target_t")
        return int(target_t is not None and t == target_t)

    def learned_policy(state: Tuple[int, ...], ctx: Dict[str, Any]) -> int:
        env = ctx["env"]
        patient = ctx["patient"]
        t = int(ctx["t"])
        if t == patient.history_start_idx:
            return int(patient.observed_actions[t])
        if learner.q_learner is None:
            raise ValueError("Q-learning has not been trained yet.")
        valid_actions = list(ctx["valid_actions"])
        return int(learner.q_learner.select_action(state, valid_actions=valid_actions, greedy_only=True))

    return CovidPolicies(
        learned=learned_policy,
        data=data_policy,
        none=none_policy,
        all=all_policy,
    )


# -----------------------------------------------------------------------------
# Dataset builders
# -----------------------------------------------------------------------------

def build_policy_dataset(long_df: pd.DataFrame, seed: int) -> TrajectoryDataset:
    return TrajectoryDataset.from_long_format(
        df=long_df,
        patient_id_col="id",
        time_col="month_index",
        action_col="action",
        rnn_covariate_cols=RNN_COVARIATE_COLS,
        rnn_outcome_cols=RNN_OUTCOME_COLS,
        rl_state_cols=RL_STATE_COLS,
        time_since_action_state_col="months_since_vax_cat",
        time_since_action_state_bins=TIME_SINCE_ACTION_BINS,
        reward_outcome_col="sev_inf_next",
        episode_start_fn=covid_episode_start_fn,
        seed=seed,
    )


def build_rnn_training_dataset(long_df: pd.DataFrame, seed: int) -> TrajectoryDataset:
    """Optional helper for refitting the RNN with the original masking logic.

    The original create_rnn_data.ipynb masks each sequence after the first severe infection.
    The generic interface does not expose per-sequence masks directly, so for RNN refitting we
    emulate the same behavior by truncating each patient trajectory at the first severe event.
    """
    trunc_frames = []
    for _, group in long_df.groupby("id", sort=False):
        group = group.sort_values("month_index").copy()
        severe_pos = np.where(group["sev_inf_next"].astype(int).values == 1)[0]
        if len(severe_pos) > 0:
            end = int(severe_pos[0]) + 1
            group = group.iloc[:end, :].copy()
        trunc_frames.append(group)
    trunc_df = pd.concat(trunc_frames, axis=0, ignore_index=True)
    return build_policy_dataset(trunc_df, seed=seed)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SanityCheckOutputs:
    reward_summary: Dict[str, float]
    epoch_reward_shape: Tuple[int, int]
    real_inf_rate_first5: np.ndarray
    real_sev_rate_first5: np.ndarray



def run_sanity_check(
    rnn_data_path: str,
    weights_path: Optional[str],
    reward_type: str,
    vax_cost: float,
    train_repeats: int,
    eval_epochs: int,
    seed: int,
    fit_rnn: bool,
    save_q_table_path: Optional[str],
) -> SanityCheckOutputs:
    set_all_seeds(seed)

    long_df, raw_df = load_covid_long_format(rnn_data_path)
    policy_dataset = build_policy_dataset(long_df, seed=seed)
    hooks = build_covid_hooks(policy_dataset, vax_cost=vax_cost, reward_type=reward_type)

    learner = MicrosimQLearner(
        dataset=policy_dataset,
        seed=seed,
        reward_fn=hooks.reward_fn,
        action_constraint_fn=hooks.action_constraint_fn,
        transition_fn=hooks.transition_fn,
        terminal_fn=hooks.terminal_fn,
    )

    if fit_rnn:
        print("\n===== RNN refit mode =====")
        rnn_train_dataset = build_rnn_training_dataset(long_df, seed=seed)
        rnn_trainer = MicrosimQLearner(dataset=rnn_train_dataset, seed=seed)
        rnn_trainer.fit_sequence_model(
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_NUM_LAYERS,
            dropout=RNN_DROPOUT,
            epochs=RNN_EPOCHS,
            lr=RNN_LR,
            batch_size=32,
            verbose_every=100,
        )
        learner.rnn_model = rnn_trainer.rnn_model
    else:
        if weights_path is None:
            raise ValueError("weights_path must be provided unless --fit-rnn is used.")
        learner.load_sequence_model(
            model_path=weights_path,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_NUM_LAYERS,
            dropout=RNN_DROPOUT,
        )

    print("\n===== Generic preprocessing summary =====")
    summary = policy_dataset.summary()
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n===== Training Q-learning =====")
    fit_out = learner.fit_tabular_q_learning(
        repeats_train_eval=train_repeats,
        gamma=0.99,
        learning_rate=0.01,
        learning_rate_decay=0.998,
        min_learning_rate=1e-5,
        epsilon=0.5,
        epsilon_decay=0.99,
        decay_every=5000,
    )
    if save_q_table_path is not None:
        np.save(save_q_table_path, fit_out["q_table"])

    policies = build_covid_policies(learner, seed=seed)

    set_all_seeds(seed)
    learned = learner.evaluate_policy(policies.learned, epochs=eval_epochs)
    set_all_seeds(seed)
    observed = learner.evaluate_policy(policies.data, epochs=eval_epochs)
    set_all_seeds(seed)
    always = learner.evaluate_policy(policies.all, epochs=eval_epochs)
    set_all_seeds(seed)
    never = learner.evaluate_policy(policies.none, epochs=eval_epochs)

    reward_summary = {
        "table": float(np.nanmean(learned)),
        "data": float(np.nanmean(observed)),
        "all": float(np.nanmean(always)),
        "none": float(np.nanmean(never)),
    }

    real_inf_rate, real_sev_rate = helpers.rate_by_month(raw_df.copy())

    print("\n===== Reward summary =====")
    for name in ["table", "data", "all", "none"]:
        print(f"{name:6s}: {reward_summary[name]:.6f}")

    print("\n===== Training/eval reward array =====")
    print(fit_out["epoch_reward_list"].shape)

    print("\n===== Real marginal rates =====")
    print("inf first 5 months:", real_inf_rate[:5])
    print("sev first 5 months:", real_sev_rate[:5])

    return SanityCheckOutputs(
        reward_summary=reward_summary,
        epoch_reward_shape=fit_out["epoch_reward_list"].shape,
        real_inf_rate_first5=real_inf_rate[:5],
        real_sev_rate_first5=real_sev_rate[:5],
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a COVID sanity check through the generic booster interface.")
    parser.add_argument("--rnn-data", required=True, help="Path to RLdata_for_RNN_01242024.csv")
    parser.add_argument("--weights", default=None, help="Path to pretrained RNN weights (.pth)")
    parser.add_argument("--reward-type", default="prop", choices=["linear", "log", "logprop", "prop"])
    parser.add_argument("--vax-cost", type=float, default=0.04)
    parser.add_argument("--train-repeats", type=int, default=1, help="Use 30 for the paper setting; 1 is a quick sanity check.")
    parser.add_argument("--eval-epochs", type=int, default=1, help="Use 5 for the paper setting; 1 is a quick sanity check.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fit-rnn", action="store_true", help="Refit the RNN instead of loading pretrained weights.")
    parser.add_argument("--save-q-table", default=None, help="Optional path to save the learned Q-table (.npy)")
    return parser.parse_args()


if __name__ == "__main__":

    USE_DEFAULT = True

    if USE_DEFAULT:
        class Args:
            rnn_data = "data/RLdata_for_RNN_01242024.csv"
            weights = "data/rnn_weights_2_128_2000_1e-04.pth"
            reward_type = "prop"
            vax_cost = 0.04
            train_repeats = 1
            eval_epochs = 1
            seed = 2024
            fit_rnn = False
            save_q_table = None

        args = Args()

    else:
        args = parse_args()

    run_sanity_check(
        rnn_data_path=args.rnn_data,
        weights_path=args.weights,
        reward_type=args.reward_type,
        vax_cost=args.vax_cost,
        train_repeats=args.train_repeats,
        eval_epochs=args.eval_epochs,
        seed=args.seed,
        fit_rnn=args.fit_rnn,
        save_q_table_path=args.save_q_table,
    )
