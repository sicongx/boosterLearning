
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

"""
Generic trajectory-based sequential decision learning interface.

This module exposes two user-facing objects:

1. TrajectoryDataset
   - builds padded sequence arrays and per-patient trajectory artifacts
   - stores column mappings and discrete RL state mappings

2. MicrosimQLearner
   - trains/loads a sequence model
   - builds a generic microsimulation environment
   - trains and evaluates tabular Q-learning policies

Users provide long-format trajectory data and optional custom functions for reward, action constraints,
episode start, and fixed-policy evaluation.
"""

State = Tuple[int, ...]
RewardFn = Callable[[Dict[str, Any]], float]
ActionConstraintFn = Callable[[Dict[str, Any]], Sequence[int]]
EpisodeStartFn = Callable[[Dict[str, Any]], int]
TransitionFn = Callable[[Dict[str, Any]], np.ndarray]
TerminalFn = Callable[[Dict[str, Any]], bool]
PolicyFn = Callable[[State, Dict[str, Any]], int]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, seq_length: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = np.asarray(seq_length, dtype=np.int64)
        self.n, self.t, self.p = self.x.shape
        self.output_size = self.y.shape[2]
        self.seq_mask_y = self._make_mask()

    def _make_mask(self) -> torch.Tensor:
        mask = np.zeros((self.n, self.t, self.output_size), dtype=np.float32)
        for i, length in enumerate(self.seq_length):
            mask[i, :length, :] = 1.0
        return torch.tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.seq_mask_y[idx]


class RNNModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class TabularQLearner:
    def __init__(
        self,
        state_levels: Sequence[int],
        action_space: Sequence[int] = (0, 1),
        gamma: float = 0.99,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.998,
        min_learning_rate: float = 1e-5,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.99,
        decay_every: int = 5000,
        seed: int = 2024,
    ) -> None:
        self.state_levels = tuple(int(v) for v in state_levels)
        self.action_space = tuple(int(a) for a in action_space)
        self.action_to_index = {a: i for i, a in enumerate(self.action_space)}
        self.q_table = np.zeros(self.state_levels + (len(self.action_space),), dtype=np.float32)

        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.min_learning_rate = float(min_learning_rate)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.decay_every = int(decay_every)
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        state: State,
        valid_actions: Sequence[int],
        greedy_only: bool = False,
    ) -> int:
        valid_actions = [int(a) for a in valid_actions]
        if len(valid_actions) == 0:
            raise ValueError("valid_actions cannot be empty.")

        valid_idx = [self.action_to_index[a] for a in valid_actions]

        if (not greedy_only) and (self.rng.uniform() < self.epsilon):
            return int(self.rng.choice(valid_actions))

        q_vals = self.q_table[state]
        best_local_idx = valid_idx[int(np.argmax(q_vals[valid_idx]))]
        return int(self.action_space[best_local_idx])

    def update(self, cur_state: State, cur_action: int, reward: float, next_state: State) -> None:
        a_idx = self.action_to_index[int(cur_action)]
        self.q_table[cur_state + (a_idx,)] = (
            self.q_table[cur_state + (a_idx,)]
            + self.learning_rate
            * (
                reward
                + self.gamma * np.max(self.q_table[next_state])
                - self.q_table[cur_state + (a_idx,)]
            )
        )
        self.steps += 1
        if self.steps % self.decay_every == 0:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
            self.epsilon *= self.epsilon_decay



@dataclass
class TrajectoryPatientArtifacts:
    patient_id: Any
    times: np.ndarray
    rnn_inputs: np.ndarray
    rnn_outcomes: np.ndarray
    observed_actions: np.ndarray
    rl_state_raw: pd.DataFrame
    rl_state_idx_seq: np.ndarray
    action_positions: np.ndarray
    history_start_idx: int


@dataclass
class EnvironmentConfig:
    action_col: str
    action_col_idx: int
    rl_state_cols: Sequence[str]
    rl_state_maps: Dict[str, Dict[Any, int]]
    outcome_indices: Dict[str, int]
    reward_outcome_col: str
    reward_outcome_idx: int
    feature_col_index: Dict[str, int]
    action_space: Sequence[int] = (0, 1)
    cumulative_action_col: Optional[str] = None
    time_since_action_col: Optional[str] = None
    time_since_action_state_col: Optional[str] = None
    time_since_action_state_bins: Optional[Sequence[int]] = None
    reward_fn: Optional[RewardFn] = None
    action_constraint_fn: Optional[ActionConstraintFn] = None
    episode_start_fn: Optional[EpisodeStartFn] = None
    transition_fn: Optional[TransitionFn] = None
    terminal_fn: Optional[TerminalFn] = None


@dataclass
class PolicyEvaluationResult:
    rewards: np.ndarray
    q_table: Optional[np.ndarray] = None


class GenericTrajectoryEnv:
    def __init__(
        self,
        rnn_model: RNNModel,
        patient: TrajectoryPatientArtifacts,
        config: EnvironmentConfig,
        device: str = "cpu",
        action_history: Optional[np.ndarray] = None,
    ) -> None:
        self.rnn = rnn_model
        self.patient = patient
        self.config = config
        self.device = device

        self.total_steps = patient.rnn_inputs.shape[0]
        self.history_len = int(patient.history_start_idx + 1)

        if action_history is None:
            self.action_history = patient.observed_actions[: self.history_len].astype(int).copy()
        else:
            self.action_history = np.asarray(action_history, dtype=int).copy()

        self.path_x = patient.rnn_inputs[: self.history_len].copy().astype(np.float32)
        self.current_step = self.history_len - 1

        action_positions = np.where(self.action_history == 1)[0]
        if len(action_positions) == 0:
            self.time_since_action = 10**6
        else:
            self.time_since_action = int(self.current_step - action_positions[-1] + 1)

        self.last_predicted_risk: Optional[np.ndarray] = None
        self.last_reward: Optional[float] = None
        self.done: bool = False
        self._refresh_state_from_last_row()

    def _map_state_value(self, col: str, value: Any) -> int:
        mapping = self.config.rl_state_maps[col]
        if value in mapping:
            return int(mapping[value])

        keys = list(mapping.keys())
        try:
            numeric_keys = np.array([float(k) for k in keys], dtype=float)
            numeric_value = float(value)
            nearest = keys[int(np.argmin(np.abs(numeric_keys - numeric_value)))]
            return int(mapping[nearest])
        except Exception:
            return int(mapping[keys[0]])

    def _refresh_state_from_last_row(self) -> None:
        raw_row = self.patient.rl_state_raw.iloc[self.current_step]
        state_list = []

        for col in self.config.rl_state_cols:
            if col == self.config.time_since_action_state_col:
                if self.config.time_since_action_state_bins is None:
                    mapped_value = self.time_since_action
                else:
                    cut_val = pd.cut(
                        [self.time_since_action],
                        bins=self.config.time_since_action_state_bins,
                        include_lowest=True,
                        right=False,
                        labels=False,
                    )[0]

                    if pd.isna(cut_val):
                        cut_val = len(self.config.time_since_action_state_bins) - 2

                    mapped_value = int(cut_val)
                state_list.append(self._map_state_value(col, mapped_value))

            elif col == self.config.time_since_action_col:
                state_list.append(self._map_state_value(col, self.time_since_action))

            elif col == self.config.cumulative_action_col:
                state_list.append(self._map_state_value(col, int(self.action_history.sum())))

            elif col == self.config.action_col:
                state_list.append(self._map_state_value(col, int(self.action_history[-1])))

            else:
                state_list.append(self._map_state_value(col, raw_row[col]))

        self.tq_state = tuple(state_list)

    def _default_transition(self, base_next_row: np.ndarray, action: int) -> np.ndarray:
        row = base_next_row.copy()
        row[self.config.action_col_idx] = float(action)

        if action == 1:
            self.time_since_action = 1
        else:
            self.time_since_action += 1

        if self.config.cumulative_action_col is not None:
            idx = self.config.feature_col_index[self.config.cumulative_action_col]
            row[idx] = float(self.action_history.sum())

        if self.config.time_since_action_col is not None:
            idx = self.config.feature_col_index[self.config.time_since_action_col]
            row[idx] = float(self.time_since_action)

        return row

    def get_valid_actions(self) -> Sequence[int]:
        if self.config.action_constraint_fn is None:
            return list(self.config.action_space)

        context = {
            "env": self,
            "patient": self.patient,
            "current_step": self.current_step,
            "time_since_action": self.time_since_action,
            "action_history": self.action_history.copy(),
            "state": self.tq_state,
        }
        return list(self.config.action_constraint_fn(context))

    def step(self, action: int) -> Tuple[np.ndarray, State, float, bool]:
        if self.current_step >= self.total_steps - 1:
            self.done = True
            return self.path_x[-1].copy(), self.tq_state, 0.0, True

        next_step = self.current_step + 1
        base_next_row = self.patient.rnn_inputs[next_step].copy().astype(np.float32)

        self.action_history = np.append(self.action_history, int(action))

        if self.config.transition_fn is None:
            next_row = self._default_transition(base_next_row, int(action))
        else:
            next_row = self.config.transition_fn(
                {
                    "env": self,
                    "base_next_row": base_next_row,
                    "action": int(action),
                    "next_step": next_step,
                }
            ).astype(np.float32)

        self.path_x = np.vstack([self.path_x, next_row.reshape(1, -1)])
        self.current_step = next_step
        self._refresh_state_from_last_row()

        with torch.no_grad():
            x = torch.tensor(self.path_x, dtype=torch.float32, device=self.device).unsqueeze(0)
            risk = self.rnn.predict_proba(x)[0, -1, :].detach().cpu().numpy()

        self.last_predicted_risk = risk.copy()

        reward_context = {
            "env": self,
            "patient": self.patient,
            "action": int(action),
            "predicted_outcomes": risk.copy(),
            "reward_outcome_idx": self.config.reward_outcome_idx,
            "reward_outcome_col": self.config.reward_outcome_col,
            "current_step": self.current_step,
            "state": self.tq_state,
        }

        if self.config.reward_fn is None:
            reward = -float(risk[self.config.reward_outcome_idx])
        else:
            reward = float(self.config.reward_fn(reward_context))

        self.last_reward = reward

        if self.config.terminal_fn is None:
            done = False
        else:
            done = bool(
                self.config.terminal_fn(
                    {
                        "env": self,
                        "patient": self.patient,
                        "action": int(action),
                        "predicted_outcomes": risk.copy(),
                        "current_step": self.current_step,
                        "state": self.tq_state,
                    }
                )
            )

        self.done = done
        return self.path_x[-1].copy(), self.tq_state, reward, done


class TrajectoryDataset:
    def __init__(self, seed: int = 2024) -> None:
        self.seed = int(seed)
        self.long_format_df: Optional[pd.DataFrame] = None
        self.patients: List[TrajectoryPatientArtifacts] = []

        self.covariates_rnn: Optional[np.ndarray] = None
        self.outcomes_rnn: Optional[np.ndarray] = None
        self.seq_length: Optional[np.ndarray] = None

        self.patient_id_col: Optional[str] = None
        self.time_col: Optional[str] = None
        self.action_col: Optional[str] = None

        self.rnn_covariate_cols: List[str] = []
        self.rnn_outcome_cols: List[str] = []
        self.rl_state_cols: List[str] = []

        self.feature_col_index: Dict[str, int] = {}
        self.outcome_col_index: Dict[str, int] = {}
        self.rl_state_maps: Dict[str, Dict[Any, int]] = {}

        self.cumulative_action_col: Optional[str] = None
        self.time_since_action_col: Optional[str] = None
        self.time_since_action_state_col: Optional[str] = None
        self.time_since_action_state_bins: Optional[List[int]] = None
        self.reward_outcome_col: Optional[str] = None

    @staticmethod
    def _build_state_maps(df: pd.DataFrame, rl_state_cols: Sequence[str]) -> Dict[str, Dict[Any, int]]:
        state_maps: Dict[str, Dict[Any, int]] = {}
        for col in rl_state_cols:
            vals = pd.Series(df[col]).dropna().unique().tolist()
            try:
                vals = sorted(vals)
            except Exception:
                vals = list(vals)
            state_maps[col] = {v: i for i, v in enumerate(vals)}
        return state_maps

    @staticmethod
    def _default_history_start_idx(actions: np.ndarray) -> int:
        pos = np.where(actions == 1)[0]
        if len(pos) >= 2:
            return int(pos[1])
        if len(pos) == 1:
            return int(pos[0])
        return 0

    @classmethod
    def from_long_format(
        cls,
        df: pd.DataFrame,
        patient_id_col: str,
        time_col: str,
        action_col: str,
        rnn_covariate_cols: Sequence[str],
        rnn_outcome_cols: Sequence[str],
        rl_state_cols: Sequence[str],
        cumulative_action_col: Optional[str] = None,
        time_since_action_col: Optional[str] = None,
        time_since_action_state_col: Optional[str] = None,
        time_since_action_state_bins: Optional[Sequence[int]] = None,
        reward_outcome_col: Optional[str] = None,
        episode_start_fn: Optional[EpisodeStartFn] = None,
        seed: int = 2024,
    ) -> "TrajectoryDataset":
        obj = cls(seed=seed)

        required = {patient_id_col, time_col, action_col, *rnn_covariate_cols, *rnn_outcome_cols, *rl_state_cols}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if action_col not in rnn_covariate_cols:
            raise ValueError("action_col must be included in rnn_covariate_cols.")

        work = df.copy().sort_values([patient_id_col, time_col]).reset_index(drop=True)

        obj.long_format_df = work.copy()
        obj.patient_id_col = patient_id_col
        obj.time_col = time_col
        obj.action_col = action_col
        obj.rnn_covariate_cols = list(rnn_covariate_cols)
        obj.rnn_outcome_cols = list(rnn_outcome_cols)
        obj.rl_state_cols = list(rl_state_cols)
        obj.cumulative_action_col = cumulative_action_col
        obj.time_since_action_col = time_since_action_col
        obj.time_since_action_state_col = time_since_action_state_col
        obj.time_since_action_state_bins = list(time_since_action_state_bins) if time_since_action_state_bins is not None else None
        obj.reward_outcome_col = reward_outcome_col or rnn_outcome_cols[0]

        obj.feature_col_index = {c: i for i, c in enumerate(obj.rnn_covariate_cols)}
        obj.outcome_col_index = {c: i for i, c in enumerate(obj.rnn_outcome_cols)}
        obj.rl_state_maps = cls._build_state_maps(work, obj.rl_state_cols)

        seq_x_list = []
        seq_y_list = []
        seq_len = []

        for patient_id, pat_df in work.groupby(patient_id_col, sort=False):
            pat_df = pat_df.sort_values(time_col).reset_index(drop=True)
            x = pat_df[obj.rnn_covariate_cols].to_numpy(dtype=np.float32)
            y = pat_df[obj.rnn_outcome_cols].to_numpy(dtype=np.float32)
            a = pat_df[action_col].to_numpy(dtype=int)

            rl_state_idx = np.column_stack(
                [pat_df[col].map(obj.rl_state_maps[col]).to_numpy(dtype=int) for col in obj.rl_state_cols]
            )

            if episode_start_fn is None:
                history_start_idx = cls._default_history_start_idx(a)
            else:
                history_start_idx = int(
                    episode_start_fn(
                        {
                            "patient_id": patient_id,
                            "patient_df": pat_df.copy(),
                            "actions": a.copy(),
                        }
                    )
                )

            obj.patients.append(
                TrajectoryPatientArtifacts(
                    patient_id=patient_id,
                    times=pat_df[time_col].to_numpy(),
                    rnn_inputs=x,
                    rnn_outcomes=y,
                    observed_actions=a,
                    rl_state_raw=pat_df[obj.rl_state_cols].copy(),
                    rl_state_idx_seq=rl_state_idx,
                    action_positions=np.where(a == 1)[0].astype(int),
                    history_start_idx=history_start_idx,
                )
            )

            seq_x_list.append(x)
            seq_y_list.append(y)
            seq_len.append(x.shape[0])

        max_len = max(seq_len)
        n = len(seq_x_list)
        p = len(obj.rnn_covariate_cols)
        q = len(obj.rnn_outcome_cols)

        obj.covariates_rnn = np.zeros((n, max_len, p), dtype=np.float32)
        obj.outcomes_rnn = np.zeros((n, max_len, q), dtype=np.float32)
        obj.seq_length = np.asarray(seq_len, dtype=np.int64)

        for i, (x, y) in enumerate(zip(seq_x_list, seq_y_list)):
            obj.covariates_rnn[i, : x.shape[0], :] = x
            obj.outcomes_rnn[i, : y.shape[0], :] = y

        return obj

    def summary(self) -> Dict[str, Any]:
        return {
            "n_patients": len(self.patients),
            "max_seq_len": int(self.covariates_rnn.shape[1]),
            "input_size": int(self.covariates_rnn.shape[2]),
            "output_size": int(self.outcomes_rnn.shape[2]),
            "rl_state_levels": tuple(len(self.rl_state_maps[c]) for c in self.rl_state_cols),
        }


class MicrosimQLearner:
    def __init__(
        self,
        dataset: TrajectoryDataset,
        device: Optional[str] = None,
        seed: int = 2024,
        action_space: Sequence[int] = (0, 1),
        reward_fn: Optional[RewardFn] = None,
        action_constraint_fn: Optional[ActionConstraintFn] = None,
        episode_start_fn: Optional[EpisodeStartFn] = None,
        transition_fn: Optional[TransitionFn] = None,
        terminal_fn: Optional[TerminalFn] = None,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.rnn_model: Optional[RNNModel] = None
        self.loss_history: List[float] = []
        self.q_learner: Optional[TabularQLearner] = None

        self.env_config = EnvironmentConfig(
            action_col=self.dataset.action_col,
            action_col_idx=self.dataset.feature_col_index[self.dataset.action_col],
            rl_state_cols=self.dataset.rl_state_cols,
            rl_state_maps=self.dataset.rl_state_maps,
            outcome_indices=self.dataset.outcome_col_index,
            reward_outcome_col=self.dataset.reward_outcome_col,
            reward_outcome_idx=self.dataset.outcome_col_index[self.dataset.reward_outcome_col],
            feature_col_index=self.dataset.feature_col_index,
            action_space=tuple(action_space),
            cumulative_action_col=self.dataset.cumulative_action_col,
            time_since_action_col=self.dataset.time_since_action_col,
            time_since_action_state_col=self.dataset.time_since_action_state_col,
            time_since_action_state_bins=self.dataset.time_since_action_state_bins,
            reward_fn=reward_fn,
            action_constraint_fn=action_constraint_fn,
            episode_start_fn=episode_start_fn,
            transition_fn=transition_fn,
            terminal_fn=terminal_fn,
        )

    def fit_sequence_model(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 2000,
        lr: float = 1e-4,
        batch_size: int = 32,
        verbose_every: int = 100,
    ) -> Dict[str, Any]:
        dataset = SequenceDataset(
            self.dataset.covariates_rnn,
            self.dataset.outcomes_rnn,
            self.dataset.seq_length,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = RNNModel(
            input_size=self.dataset.covariates_rnn.shape[2],
            output_size=self.dataset.outcomes_rnn.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch, seq_mask_y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                seq_mask_y_batch = seq_mask_y_batch.to(self.device).bool()

                logits = model(x_batch)
                logits_masked = logits[seq_mask_y_batch]
                targets_masked = y_batch[seq_mask_y_batch]

                loss = nn.functional.binary_cross_entropy(
                    torch.sigmoid(logits_masked),
                    targets_masked,
                    reduction="mean",
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch + 1 == epochs):
                print(f"[fit_sequence_model] epoch {epoch + 1:4d}/{epochs} | loss={avg_loss:.6f}")

        self.rnn_model = model
        return {
            "final_loss": self.loss_history[-1],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "lr": lr,
        }

    def load_sequence_model(
        self,
        model_path: str,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        model = RNNModel(
            input_size=self.dataset.covariates_rnn.shape[2],
            output_size=self.dataset.outcomes_rnn.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.rnn_model = model

    def build_env(self, patient_index: int, action_history: Optional[np.ndarray] = None) -> GenericTrajectoryEnv:
        if self.rnn_model is None:
            raise ValueError("Train or load the sequence model first.")
        return GenericTrajectoryEnv(
            rnn_model=self.rnn_model,
            patient=self.dataset.patients[patient_index],
            config=self.env_config,
            device=self.device,
            action_history=action_history,
        )

    def _resolve_policy_action(
        self,
        policy: Union[str, PolicyFn],
        env: GenericTrajectoryEnv,
        patient: TrajectoryPatientArtifacts,
        state: State,
        t: int,
    ) -> int:
        valid_actions = env.get_valid_actions()

        if callable(policy):
            return int(policy(state, {"env": env, "patient": patient, "t": t, "valid_actions": valid_actions}))

        if policy == "observed":
            return int(patient.observed_actions[t])
        if policy == "none":
            return 0
        if policy == "all":
            return max(valid_actions)
        if policy == "learned":
            if self.q_learner is None:
                raise ValueError("Train Q-learning first.")
            return int(self.q_learner.select_action(state, valid_actions=valid_actions, greedy_only=True))

        raise ValueError("Unsupported policy.")

    def simulate(
        self,
        n: Optional[int] = None,
        policy: Union[str, PolicyFn] = "observed",
    ) -> pd.DataFrame:
        n_patients = len(self.dataset.patients) if n is None else min(int(n), len(self.dataset.patients))
        out_rows = []

        for patient_index in range(n_patients):
            patient = self.dataset.patients[patient_index]
            env = self.build_env(patient_index)
            state = tuple(env.tq_state)

            for t in range(patient.history_start_idx, patient.rnn_inputs.shape[0]):
                action = self._resolve_policy_action(policy, env, patient, state, t)
                _, next_state, reward, done = env.step(action)

                row = {
                    "patient_id": patient.patient_id,
                    "t": t,
                    "action": int(action),
                    "reward": float(reward),
                }
                for j, col in enumerate(self.dataset.rl_state_cols):
                    row[f"state_{col}"] = int(next_state[j])

                if env.last_predicted_risk is not None:
                    for outcome_name, idx in self.dataset.outcome_col_index.items():
                        row[f"pred_{outcome_name}"] = float(env.last_predicted_risk[idx])

                out_rows.append(row)
                state = tuple(next_state)

                if done:
                    break

        return pd.DataFrame(out_rows)

    def fit_tabular_q_learning(
        self,
        repeats_train_eval: int = 30,
        gamma: float = 0.99,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.998,
        min_learning_rate: float = 1e-5,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.99,
        decay_every: int = 5000,
    ) -> Dict[str, Any]:
        state_levels = tuple(len(self.dataset.rl_state_maps[c]) for c in self.dataset.rl_state_cols)

        agent = TabularQLearner(
            state_levels=state_levels,
            action_space=self.env_config.action_space,
            gamma=gamma,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            min_learning_rate=min_learning_rate,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            decay_every=decay_every,
            seed=self.seed,
        )

        epochs_train_eval = np.tile(np.repeat([False, True], [1, 1]), repeats_train_eval)
        epoch_reward_list = np.full((len(epochs_train_eval), len(self.dataset.patients)), np.nan, dtype=np.float32)

        for epoch, is_train in enumerate(epochs_train_eval):
            sample_idx_array = np.random.choice(len(self.dataset.patients), size=len(self.dataset.patients))

            for i, sample_idx in enumerate(sample_idx_array):
                patient = self.dataset.patients[int(sample_idx)]
                env = self.build_env(int(sample_idx))
                state = tuple(env.tq_state)
                episodic_reward = 0.0
                horizon = max(1, patient.rnn_inputs.shape[0] - patient.history_start_idx)

                for t in range(patient.history_start_idx, patient.rnn_inputs.shape[0]):
                    if t == patient.history_start_idx:
                        action = int(patient.observed_actions[t])
                    else:
                        valid_actions = env.get_valid_actions()
                        action = agent.select_action(
                            state,
                            valid_actions=valid_actions,
                            greedy_only=(not is_train),
                        )

                    _, next_state, reward, done = env.step(action)
                    episodic_reward += reward / horizon

                    if is_train and t != patient.history_start_idx:
                        agent.update(state, action, reward, next_state)

                    state = tuple(next_state)
                    if done:
                        break

                if not is_train:
                    epoch_reward_list[epoch, i] = episodic_reward

        self.q_learner = agent
        return {"q_table": agent.q_table.copy(), "epoch_reward_list": epoch_reward_list}

    def evaluate_policy(
        self,
        policy: Union[str, PolicyFn],
        epochs: int = 5,
    ) -> np.ndarray:
        out = np.full((epochs, len(self.dataset.patients)), np.nan, dtype=np.float32)

        for epoch in range(epochs):
            sample_idx_array = np.random.choice(len(self.dataset.patients), size=len(self.dataset.patients))

            for i, sample_idx in enumerate(sample_idx_array):
                patient = self.dataset.patients[int(sample_idx)]
                env = self.build_env(int(sample_idx))
                state = tuple(env.tq_state)
                episodic_reward = 0.0
                horizon = max(1, patient.rnn_inputs.shape[0] - patient.history_start_idx)

                for t in range(patient.history_start_idx, patient.rnn_inputs.shape[0]):
                    action = self._resolve_policy_action(policy, env, patient, state, t)
                    _, next_state, reward, done = env.step(action)
                    episodic_reward += reward / horizon
                    state = tuple(next_state)
                    if done:
                        break

                out[epoch, i] = episodic_reward

        return out

