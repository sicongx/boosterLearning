
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

"""
BoosterLearning provides two interfaces:

1. Generic long-format EHR interface:
   - preprocess_long_format(...)
   - train_rnn(...)
   - simulate_env(...)
   - tabular_q_learning(...)
   - eval_q_learning(...)

2. Faithful COVID-specific reproducibility interface:
   - load_covid_data(...)
   - tabular_q_learning_covid(...)
   - eval_covid_policy(...)

The generic interface is designed for extensibility and user-facing simplicity.
The COVID-specific interface is retained to faithfully reproduce the original
JASA COVID booster experiments.
"""

State = Tuple[int, ...]


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
        state_levels: Sequence[int] = (5, 2, 3),
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
        self.q_table = np.zeros(self.state_levels + (2,), dtype=np.float32)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_every = decay_every
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: State, months_last_vax: int, previous_booster: bool, greedy_only: bool = False) -> int:
        if previous_booster:
            return 0
        if months_last_vax <= 4:
            return 0
        if (not greedy_only) and (self.rng.uniform() < self.epsilon):
            return int(self.rng.choice([0, 1]))
        return int(np.argmax(self.q_table[state]))

    def update(self, cur_state: State, cur_action: int, reward: float, next_state: State) -> None:
        self.q_table[cur_state + (cur_action,)] = (
            self.q_table[cur_state + (cur_action,)]
            + self.learning_rate
            * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[cur_state + (cur_action,)])
        )
        self.steps += 1
        if self.steps % self.decay_every == 0:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
            self.epsilon *= self.epsilon_decay


@dataclass
class CovidPatientArtifacts:
    patid: Any
    age_scaled: float
    age_cat: int
    imm_baseline: int
    gender: int
    race: np.ndarray
    visits_cat: np.ndarray
    com_cat: np.ndarray
    variant_history: np.ndarray
    action_positions: np.ndarray

@dataclass
class LongFormatPatientArtifacts:
    patient_id: Any
    times: np.ndarray
    rnn_inputs: np.ndarray
    rnn_outcomes: np.ndarray
    observed_actions: np.ndarray
    rl_state_raw: pd.DataFrame
    rl_state_idx_seq: np.ndarray
    action_positions: np.ndarray
    history_start_idx: int


class CovidBoosterEnv:
    def __init__(
        self,
        rnn_model: RNNModel,
        patient: CovidPatientArtifacts,
        vaccine_hist: np.ndarray,
        vax_cost: float = 0.04,
        reward_type: str = "log",
        device: str = "cpu",
    ) -> None:
        self.rnn = rnn_model
        self.device = device
        self.age_scaled = float(patient.age_scaled)
        self.age_cat = int(patient.age_cat)
        self.imm_baseline = int(patient.imm_baseline)
        self.gender = int(patient.gender)
        self.race = patient.race.astype(np.float32)
        self.visits_cat = patient.visits_cat.astype(np.float32)
        self.com_cat = patient.com_cat.astype(np.float32)
        self.variant_history = patient.variant_history.astype(int)
        self.vax_cost = float(vax_cost)
        self.reward_type = reward_type

        hist_t = int(vaccine_hist.shape[0])
        variant_dummies = np.array(
            pd.get_dummies(pd.cut(self.variant_history[:hist_t], bins=[0, 1, 2, 3], include_lowest=True, right=False)) + 0
        )[:, 1:]
        self.action_state_to_date = np.hstack(
            [
                vaccine_hist.reshape(-1, 1),
                np.repeat(self.age_scaled, hist_t).reshape(-1, 1),
                np.repeat(self.imm_baseline, hist_t).reshape(-1, 1),
                np.cumsum(vaccine_hist).reshape(-1, 1),
                np.repeat(self.gender, hist_t).reshape(-1, 1),
                np.tile(self.race, (hist_t, 1)),
                np.tile(self.visits_cat, (hist_t, 1)),
                np.tile(self.com_cat, (hist_t, 1)),
                variant_dummies,
            ]
        ).astype(np.float32)
        self.num_vax = int(np.cumsum(vaccine_hist)[-1])
        self.step_num = hist_t
        self.months_last_vax = 0
        self.months_last_vax_cat = 0
        self.next_month_inf = False
        self.next_month_severe_inf = False
        self.state = self._build_policy_state(variant_dummies[-1])
        self.tq_state = [self.age_cat, self.imm_baseline, self.months_last_vax_cat]

    def _build_policy_state(self, variant_dummies: np.ndarray) -> np.ndarray:
        age_dummies = np.zeros(4, dtype=np.float32)
        if self.age_cat > 0:
            age_dummies[self.age_cat - 1] = 1.0
        return np.concatenate(
            [
                age_dummies,
                np.array([self.imm_baseline, self.num_vax, self.gender], dtype=np.float32),
                self.race,
                self.visits_cat,
                self.com_cat,
                variant_dummies.astype(np.float32),
            ]
        ).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, State, float, bool]:
        done = False
        self.step_num += 1
        self.num_vax += int(action)
        self.months_last_vax = 1 if int(action) == 1 else self.months_last_vax + 1

        if self.step_num <= 16:
            variant = 0
        elif self.step_num <= 22:
            variant = 1
        else:
            variant = 2

        self.months_last_vax_cat = np.where(
            np.array(pd.get_dummies(pd.cut([self.months_last_vax], bins=[0, 5, 7, 100], include_lowest=True, right=False)) + 0).reshape(-1)
            == 1
        )[0][0]
        self.tq_state[2] = int(self.months_last_vax_cat)
        variant_dummies = np.array(
            pd.get_dummies(pd.cut([variant], bins=[0, 1, 2, 3], include_lowest=True, right=False)) + 0
        ).reshape(-1)[1:]
        action_state = np.concatenate(
            [
                np.array([action, self.age_scaled, self.imm_baseline, self.num_vax, self.gender], dtype=np.float32),
                self.race,
                self.visits_cat,
                self.com_cat,
                variant_dummies.astype(np.float32),
            ]
        ).astype(np.float32)
        self.action_state_to_date = np.vstack((self.action_state_to_date, action_state.reshape(1, -1)))
        self.state = self._build_policy_state(variant_dummies)

        with torch.no_grad():
            x = torch.tensor(self.action_state_to_date, dtype=torch.float32, device=self.device).unsqueeze(0)
            risk = self.rnn.predict_proba(x)[0, -1, :].detach().cpu().numpy()
            risk_severe_inf = float(risk[0])
            risk_inf = float(risk[1])
            self.next_month_inf = bool(np.random.choice([False, True], p=[1 - risk_inf, risk_inf], size=1).item())
            self.next_month_severe_inf = bool(np.random.choice([False, True], p=[1 - risk_severe_inf, risk_severe_inf], size=1).item())

            if self.reward_type == "linear":
                reward = -(risk_severe_inf + action * self.vax_cost) * 10000
            elif self.reward_type == "log":
                reward = -np.log(risk_severe_inf + action * self.vax_cost)
            elif self.reward_type == "logprop":
                reward = -np.log(risk_severe_inf * (1 + action * self.vax_cost))
            elif self.reward_type == "prop":
                reward = -risk_severe_inf * (1 + action * self.vax_cost) * 10000
            else:
                raise ValueError("unsupported reward_type")

        if self.next_month_severe_inf:
            done = True
        return self.state.copy(), tuple(int(x) for x in self.tq_state), float(reward), done


class GenericBoosterEnv:
    def __init__(
        self,
        rnn_model: RNNModel,
        patient: LongFormatPatientArtifacts,
        action_col_idx: int,
        reward_outcome_idx: int,
        outcome_indices: Dict[str, int],
        rl_state_cols: Sequence[str],
        rl_state_maps: Dict[str, Dict[Any, int]],
        vax_cost: float = 0.04,
        reward_type: str = "log",
        device: str = "cpu",
        action_history: Optional[np.ndarray] = None,
        cumulative_action_col: Optional[str] = None,
        months_since_action_col: Optional[str] = None,
        months_since_action_state_col: Optional[str] = None,
        months_since_action_state_bins: Optional[Sequence[int]] = None,
        min_action_gap: int = 4,
    ) -> None:
        self.rnn = rnn_model
        self.device = device
        self.patient = patient
        self.action_col_idx = int(action_col_idx)
        self.reward_outcome_idx = int(reward_outcome_idx)
        self.outcome_indices = outcome_indices
        self.rl_state_cols = list(rl_state_cols)
        self.rl_state_maps = rl_state_maps
        self.vax_cost = float(vax_cost)
        self.reward_type = reward_type
        self.cumulative_action_col = cumulative_action_col
        self.months_since_action_col = months_since_action_col
        self.months_since_action_state_col = months_since_action_state_col
        self.months_since_action_state_bins = (
            list(months_since_action_state_bins)
            if months_since_action_state_bins is not None
            else None
        )
        self.min_action_gap = int(min_action_gap)

        self.total_steps = patient.rnn_inputs.shape[0]
        self.history_len = int(patient.history_start_idx + 1)

        if action_history is None:
            self.action_history = patient.observed_actions[: self.history_len].astype(int).copy()
        else:
            self.action_history = np.asarray(action_history, dtype=int).copy()

        self.path_x = patient.rnn_inputs[: self.history_len].copy().astype(np.float32)
        self.current_step = self.history_len - 1
        self.previous_extra_action = False

        action_positions = np.where(self.action_history == 1)[0]
        if len(action_positions) == 0:
            self.months_since_action = 100
        else:
            self.months_since_action = int(self.current_step - action_positions[-1] + 1)

        self.next_month_inf = False
        self.next_month_severe_inf = False
        self._refresh_state_from_last_row()

    def _map_state_value(self, col: str, value: Any) -> int:
        mapping = self.rl_state_maps[col]
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
        last_idx = self.current_step
        raw_row = self.patient.rl_state_raw.iloc[last_idx]
        state_list = []

        for col in self.rl_state_cols:
            if col == self.months_since_action_state_col:
                if self.months_since_action_state_bins is None:
                    mapped_value = self.months_since_action
                else:
                    mapped_value = int(
                        pd.cut(
                            [self.months_since_action],
                            bins=self.months_since_action_state_bins,
                            include_lowest=True,
                            right=False,
                            labels=False,
                        )[0]
                    )
                state_list.append(self._map_state_value(col, mapped_value))

            elif col == self.months_since_action_col:
                state_list.append(self._map_state_value(col, self.months_since_action))

            elif col == self.cumulative_action_col:
                state_list.append(self._map_state_value(col, int(self.action_history.sum())))

            elif col == "action":
                state_list.append(self._map_state_value(col, int(self.action_history[-1])))

            else:
                state_list.append(self._map_state_value(col, raw_row[col]))

        self.tq_state = tuple(state_list)

    def _update_next_row_features(self, base_next_row: np.ndarray, action: int, next_step: int) -> np.ndarray:
        row = base_next_row.copy()

        row[self.action_col_idx] = float(action)

        if action == 1:
            self.months_since_action = 1
        else:
            self.months_since_action += 1

        if self.cumulative_action_col is not None:
            cum_idx = self.feature_col_index[self.cumulative_action_col]
            row[cum_idx] = float(self.action_history.sum())

        if self.months_since_action_col is not None:
            m_idx = self.feature_col_index[self.months_since_action_col]
            row[m_idx] = float(self.months_since_action)

        return row

    def attach_feature_index(self, feature_col_index: Dict[str, int]) -> None:
        self.feature_col_index = feature_col_index

    def step(self, action: int) -> Tuple[np.ndarray, State, float, bool]:
        done = False

        if self.current_step >= self.total_steps - 1:
            return self.path_x[-1].copy(), self.tq_state, 0.0, True

        next_step = self.current_step + 1
        base_next_row = self.patient.rnn_inputs[next_step].copy().astype(np.float32)

        if action == 1:
            self.previous_extra_action = True

        self.action_history = np.append(self.action_history, int(action))
        next_row = self._update_next_row_features(base_next_row, int(action), next_step)
        self.path_x = np.vstack([self.path_x, next_row.reshape(1, -1)])
        self.current_step = next_step
        self._refresh_state_from_last_row()

        with torch.no_grad():
            x = torch.tensor(self.path_x, dtype=torch.float32, device=self.device).unsqueeze(0)
            risk = self.rnn.predict_proba(x)[0, -1, :].detach().cpu().numpy()

            risk_severe = float(risk[self.reward_outcome_idx])

            inf_idx = self.outcome_indices.get("inf_next", None)
            risk_inf = float(risk[inf_idx]) if inf_idx is not None else risk_severe

            self.next_month_inf = bool(np.random.choice([False, True], p=[1 - risk_inf, risk_inf], size=1).item())
            self.next_month_severe_inf = bool(
                np.random.choice([False, True], p=[1 - risk_severe, risk_severe], size=1).item()
            )

            if self.reward_type == "linear":
                reward = -(risk_severe + action * self.vax_cost) * 10000
            elif self.reward_type == "log":
                reward = -np.log(risk_severe + action * self.vax_cost)
            elif self.reward_type == "logprop":
                reward = -np.log(risk_severe * (1 + action * self.vax_cost))
            elif self.reward_type == "prop":
                reward = -risk_severe * (1 + action * self.vax_cost) * 10000
            else:
                raise ValueError("unsupported reward_type")

        if self.next_month_severe_inf:
            done = True

        return self.path_x[-1].copy(), self.tq_state, float(reward), done


class BoosterLearning:
    def __init__(self, device: Optional[str] = None, seed: int = 2024) -> None:
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(self.seed)
        self.covariates_rnn: Optional[np.ndarray] = None
        self.outcomes_rnn: Optional[np.ndarray] = None
        self.seq_length: Optional[np.ndarray] = None
        self.rnn_model: Optional[RNNModel] = None
        self.loss_history: List[float] = []
        self.patient_artifacts: List[CovidPatientArtifacts] = []
        self.q_learner: Optional[TabularQLearner] = None
        self.long_format_df: Optional[pd.DataFrame] = None
        self.long_format_patients: List[LongFormatPatientArtifacts] = []
        self.rnn_covariate_cols: List[str] = []
        self.rnn_outcome_cols: List[str] = []
        self.rl_state_cols: List[str] = []
        self.feature_col_index: Dict[str, int] = {}
        self.outcome_col_index: Dict[str, int] = {}
        self.rl_state_maps: Dict[str, Dict[Any, int]] = {}
        self.patient_id_col: Optional[str] = None
        self.time_col: Optional[str] = None
        self.action_col: Optional[str] = None
        self.cumulative_action_col: Optional[str] = None
        self.months_since_action_col: Optional[str] = None
        self.months_since_action_state_col: Optional[str] = None
        self.months_since_action_state_bins: Optional[List[int]] = None
        self.reward_outcome_col: Optional[str] = None
        self.min_action_gap: int = 4
        self.static_covariate_cols: List[str] = []
        self.dynamic_covariate_cols: List[str] = []
        self.time_varying_covariate_cols: List[str] = []
        self.variant_col: Optional[str] = None
        self.variant_update_mode: Optional[str] = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _validate_preprocess_inputs(
        self,
        df: pd.DataFrame,
        patient_id_col: str,
        time_col: str,
        action_col: str,
        rnn_covariate_cols: Sequence[str],
        rnn_outcome_cols: Sequence[str],
        rl_state_cols: Sequence[str],
    ) -> None:
        required = {patient_id_col, time_col, action_col, *rnn_covariate_cols, *rnn_outcome_cols, *rl_state_cols}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if action_col not in rnn_covariate_cols:
            raise ValueError("action_col must be included in rnn_covariate_cols so simulated actions can be injected.")

    def _build_state_maps(self, df: pd.DataFrame, rl_state_cols: Sequence[str]) -> Dict[str, Dict[Any, int]]:
        state_maps: Dict[str, Dict[Any, int]] = {}
        for col in rl_state_cols:
            vals = pd.Series(df[col]).dropna().unique().tolist()
            try:
                vals = sorted(vals)
            except Exception:
                vals = list(vals)
            state_maps[col] = {v: i for i, v in enumerate(vals)}
        return state_maps

    def _default_history_start_idx(self, actions: np.ndarray) -> int:
        pos = np.where(actions == 1)[0]
        if len(pos) >= 2:
            return int(pos[1])
        if len(pos) == 1:
            return int(pos[0])
        return 0

    def load_rnn_arrays(self, covariates_rnn: np.ndarray, outcomes_rnn: np.ndarray, seq_length: np.ndarray) -> None:
        self.covariates_rnn = np.asarray(covariates_rnn, dtype=np.float32)
        self.outcomes_rnn = np.asarray(outcomes_rnn, dtype=np.float32)
        self.seq_length = np.asarray(seq_length, dtype=np.int64)

    def preprocess_long_format(
            self,
            df: pd.DataFrame,
            patient_id_col: str,
            time_col: str,
            action_col: str,
            rnn_covariate_cols: Sequence[str],
            rnn_outcome_cols: Sequence[str],
            rl_state_cols: Sequence[str],
            cumulative_action_col: Optional[str] = None,
            months_since_action_col: Optional[str] = None,
            months_since_action_state_col: Optional[str] = None,
            months_since_action_state_bins: Optional[Sequence[int]] = None,
            reward_outcome_col: Optional[str] = None,
            min_action_gap: int = 4,
    ) -> Dict[str, Any]:
        self._validate_preprocess_inputs(
            df=df,
            patient_id_col=patient_id_col,
            time_col=time_col,
            action_col=action_col,
            rnn_covariate_cols=rnn_covariate_cols,
            rnn_outcome_cols=rnn_outcome_cols,
            rl_state_cols=rl_state_cols,
        )

        work = df.copy().sort_values([patient_id_col, time_col]).reset_index(drop=True)

        self.long_format_df = work.copy()
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.action_col = action_col
        self.rnn_covariate_cols = list(rnn_covariate_cols)
        self.rnn_outcome_cols = list(rnn_outcome_cols)
        self.rl_state_cols = list(rl_state_cols)
        self.cumulative_action_col = cumulative_action_col
        self.months_since_action_col = months_since_action_col
        self.months_since_action_state_col = months_since_action_state_col
        self.months_since_action_state_bins = (
            list(months_since_action_state_bins)
            if months_since_action_state_bins is not None
            else None
        )
        self.reward_outcome_col = reward_outcome_col or rnn_outcome_cols[0]
        self.min_action_gap = int(min_action_gap)

        self.feature_col_index = {c: i for i, c in enumerate(self.rnn_covariate_cols)}
        self.outcome_col_index = {c: i for i, c in enumerate(self.rnn_outcome_cols)}
        self.rl_state_maps = self._build_state_maps(work, self.rl_state_cols)

        patients = []
        seq_x_list = []
        seq_y_list = []
        seq_len = []

        for patient_id, pat_df in work.groupby(patient_id_col, sort=False):
            pat_df = pat_df.sort_values(time_col).reset_index(drop=True)

            x = pat_df[self.rnn_covariate_cols].to_numpy(dtype=np.float32)
            y = pat_df[self.rnn_outcome_cols].to_numpy(dtype=np.float32)
            a = pat_df[action_col].to_numpy(dtype=int)

            rl_state_idx = np.column_stack(
                [
                    pat_df[col].map(self.rl_state_maps[col]).to_numpy(dtype=int)
                    for col in self.rl_state_cols
                ]
            )

            history_start_idx = self._default_history_start_idx(a)

            patients.append(
                LongFormatPatientArtifacts(
                    patient_id=patient_id,
                    times=pat_df[time_col].to_numpy(),
                    rnn_inputs=x,
                    rnn_outcomes=y,
                    observed_actions=a,
                    rl_state_raw=pat_df[self.rl_state_cols].copy(),
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
        p = len(self.rnn_covariate_cols)
        q = len(self.rnn_outcome_cols)

        covariates_rnn = np.zeros((n, max_len, p), dtype=np.float32)
        outcomes_rnn = np.zeros((n, max_len, q), dtype=np.float32)
        seq_length = np.asarray(seq_len, dtype=np.int64)

        for i, (x, y) in enumerate(zip(seq_x_list, seq_y_list)):
            covariates_rnn[i, : x.shape[0], :] = x
            outcomes_rnn[i, : y.shape[0], :] = y

        self.long_format_patients = patients
        self.load_rnn_arrays(covariates_rnn, outcomes_rnn, seq_length)

        return {
            "n_patients": len(self.long_format_patients),
            "max_seq_len": int(max_len),
            "input_size": int(p),
            "output_size": int(q),
            "rl_state_levels": tuple(len(self.rl_state_maps[c]) for c in self.rl_state_cols),
        }

    def load_covid_data(self, real_df: pd.DataFrame, demographics_df: pd.DataFrame) -> None:
        df = real_df.copy()
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df["Age.FirstDose"] = (
            pd.to_numeric(df["Age.FirstDose"].replace({">89": "90"}), errors="raise")
            .astype(int)
        )
        severe_idx = np.where(df["severe_infection_next"].to_numpy() == 1)[0]
        df.loc[severe_idx, "inf_next"] = 1

        demo = demographics_df.copy()
        if "Unnamed: 0" in demo.columns:
            demo = demo.drop(columns=["Unnamed: 0"])

        real_demo = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        demo = demo.reset_index(drop=True)
        age_cat = pd.cut(real_demo["Age.FirstDose"].astype(int), bins=[0, 18, 30, 50, 65, 100], include_lowest=True, right=False)
        age_dummies = pd.get_dummies(age_cat).iloc[:, 1:].reset_index(drop=True)
        action_list = (
            df.groupby("id")["action"]
            .apply(lambda s: np.where(s.to_numpy() == 1)[0])
            .to_dict()
        )
        variant_map = {"none": 0, "delta": 1, "omicron": 2}

        self.patient_artifacts = []
        for i in range(len(real_demo)):
            patid = real_demo.iloc[i]["id"]
            pat_df = df[df["id"] == patid].reset_index(drop=True)
            age_dummy_row = age_dummies.iloc[i].to_numpy(dtype=np.float32)
            age_cat_idx = 0 if np.where(age_dummy_row == 1)[0].size == 0 else int(np.where(age_dummy_row == 1)[0][0] + 1)
            self.patient_artifacts.append(
                CovidPatientArtifacts(
                    patid=patid,
                    age_scaled=float(demo.iloc[i]["Age.FirstDose"]),
                    age_cat=age_cat_idx,
                    imm_baseline=int(demo.iloc[i]["imm_baseline"]),
                    gender=int(demo.iloc[i]["Gender"]),
                    race=np.array([demo.iloc[i]["African American"], demo.iloc[i]["Other"]], dtype=np.float32),
                    visits_cat=np.array(
                        [demo.iloc[i]["[5, 10)"], demo.iloc[i]["[10, 20)"], demo.iloc[i]["[20, 50)"], demo.iloc[i]["[50, 1000)"]],
                        dtype=np.float32,
                    ),
                    com_cat=np.array([demo.iloc[i]["[1, 3)"], demo.iloc[i]["[3, 5)"], demo.iloc[i]["[5, 100)"]], dtype=np.float32),
                    variant_history=pat_df["variant"].replace(variant_map).to_numpy(dtype=int),
                    action_positions=np.asarray(action_list[patid], dtype=int),
                )
            )

    def _make_covid_generic_dataframe(self, real_df: pd.DataFrame, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a long-format dataframe suitable for preprocess_long_format(...)
        from the original COVID project files.

        The returned dataframe keeps:
        - action as an RNN covariate
        - severe_infection_next / inf_next as RNN outcomes
        - generic RL states:
            age_cat, imm_baseline, months_since_vax_cat
        - helper columns for simulation:
            numVax, months_since_vax, month_index
        """
        df = real_df.copy()
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        demo = demographics_df.copy()
        if "Unnamed: 0" in demo.columns:
            demo = demo.drop(columns=["Unnamed: 0"])

        # align a few quirks with your old covid loader
        df["Age.FirstDose"] = (
            pd.to_numeric(df["Age.FirstDose"].replace({">89": "90"}), errors="raise")
            .astype(int)
        )

        severe_idx = np.where(df["severe_infection_next"].to_numpy() == 1)[0]
        df.loc[severe_idx, "inf_next"] = 1

        # merge demographics if columns are missing from real_df
        merge_cols = ["id"]
        add_cols = [c for c in demo.columns if c != "id" and c not in df.columns]
        if add_cols:
            df = df.merge(demo[merge_cols + add_cols], on="id", how="left")

        df = df.sort_values(["id"]).reset_index(drop=True)
        df["month_index"] = df.groupby("id").cumcount()

        # ---------- Gender numeric encoding ----------
        if "Gender" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Gender"]):
                raw_gender = df["Gender"].copy()
                gender_map = {
                    "M": 1, "Male": 1, "male": 1,
                    "F": 0, "Female": 0, "female": 0,
                }
                df["Gender"] = raw_gender.map(gender_map)

                if df["Gender"].isna().any():
                    bad_vals = sorted(raw_gender[df["Gender"].isna()].astype(str).unique().tolist())
                    raise ValueError(f"Unrecognized Gender values: {bad_vals}")

                df["Gender"] = df["Gender"].astype(int)

        # ---------- age_cat ----------
        age_cat = pd.cut(
            df["Age.FirstDose"].astype(int),
            bins=[0, 18, 30, 50, 65, 100],
            include_lowest=True,
            right=False,
        )
        age_dummies = pd.get_dummies(age_cat)
        # same coding logic as your old loader: 0,1,2,3,4
        # first bin -> 0, then later bins -> 1..4
        age_cat_idx = np.zeros(len(df), dtype=int)
        if age_dummies.shape[1] > 1:
            for j in range(1, age_dummies.shape[1]):
                age_cat_idx[age_dummies.iloc[:, j].to_numpy(dtype=int) == 1] = j
        df["age_cat"] = age_cat_idx.astype(int)

        # ---------- variant dummies ----------
        # original file has "variant" in {"none","delta","omicron"}
        # generic RNN should use dummy columns, not raw string
        if "delta" not in df.columns:
            df["delta"] = (df["variant"] == "delta").astype(int)
        if "omicron" not in df.columns:
            df["omicron"] = (df["variant"] == "omicron").astype(int)

        # ---------- numVax ----------
        if "numVax" not in df.columns:
            df["numVax"] = df.groupby("id")["action"].cumsum()

        # ---------- months_since_vax ----------
        def _months_since_action(action_series: pd.Series) -> np.ndarray:
            arr = action_series.to_numpy(dtype=int)
            out = np.zeros(len(arr), dtype=int)
            last = None
            for t in range(len(arr)):
                if arr[t] == 1:
                    last = t
                    out[t] = 1
                else:
                    if last is None:
                        out[t] = 100
                    else:
                        out[t] = t - last + 1
            return out

        df["months_since_vax"] = (
            df.groupby("id")["action"]
            .transform(lambda s: pd.Series(_months_since_action(s), index=s.index))
            .astype(int)
        )

        # ---------- months_since_vax_cat ----------
        ms_cat = pd.cut(
            df["months_since_vax"],
            bins=[0, 5, 7, 1000],
            include_lowest=True,
            right=False,
        )
        ms_dummies = pd.get_dummies(ms_cat)
        ms_cat_idx = np.zeros(len(df), dtype=int)
        for j in range(ms_dummies.shape[1]):
            ms_cat_idx[ms_dummies.iloc[:, j].to_numpy(dtype=int) == 1] = j
        df["months_since_vax_cat"] = ms_cat_idx.astype(int)

        # ensure required columns exist
        required_cols = [
            "id",
            "month_index",
            "action",
            "Age.FirstDose",
            "imm_baseline",
            "Gender",
            "African American",
            "Other",
            "[5, 10)",
            "[10, 20)",
            "[20, 50)",
            "[50, 1000)",
            "[1, 3)",
            "[3, 5)",
            "[5, 100)",
            "numVax",
            "months_since_vax",
            "age_cat",
            "months_since_vax_cat",
            "severe_infection_next",
            "inf_next",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"COVID generic preprocessing missing required columns: {missing}")

        # keep only what we need plus variant string if present
        keep_cols = required_cols + ["delta", "omicron"]
        if "variant" in df.columns:
            keep_cols.append("variant")
        keep_cols = [c for c in keep_cols if c in df.columns]

        return df[keep_cols].copy()

    # NOTE:
    # This generic preprocessing is for interface demonstration only.
    # It is not the faithful path for reproducing the original COVID tabular Q-learning results.
    def preprocess_covid_for_generic(
        self,
        real_df: pd.DataFrame,
        demographics_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Convert the original COVID data into the generic long-format interface
        and call preprocess_long_format(...).
        """
        df_generic = self._make_covid_generic_dataframe(real_df, demographics_df)

        return self.preprocess_long_format(
            df=df_generic,
            patient_id_col="id",
            time_col="month_index",
            action_col="action",
            rnn_covariate_cols=[
                "action",
                "Age.FirstDose",
                "imm_baseline",
                "numVax",
                "Gender",
                "African American",
                "Other",
                "[5, 10)",
                "[10, 20)",
                "[20, 50)",
                "[50, 1000)",
                "[1, 3)",
                "[3, 5)",
                "[5, 100)",
                "delta",
                "omicron",
            ],
            rnn_outcome_cols=[
                "severe_infection_next",
                "inf_next",
            ],
            rl_state_cols=[
                "age_cat",
                "imm_baseline",
                "months_since_vax_cat",
            ],
            cumulative_action_col="numVax",
            months_since_action_col=None,
            months_since_action_state_col="months_since_vax_cat",
            months_since_action_state_bins=[0, 5, 7, 1000],
            reward_outcome_col="severe_infection_next",
            min_action_gap=4,
        )

    def summarize_covid_generic_setup(self) -> Dict[str, Any]:
        if not self.long_format_patients:
            raise ValueError("Run preprocess_covid_for_generic(...) first.")

        return {
            "n_patients": len(self.long_format_patients),
            "input_size": len(self.rnn_covariate_cols),
            "output_size": len(self.rnn_outcome_cols),
            "rl_state_cols": list(self.rl_state_cols),
            "rl_state_levels": {c: len(self.rl_state_maps[c]) for c in self.rl_state_cols},
            "reward_outcome_col": self.reward_outcome_col,
            "min_action_gap": self.min_action_gap,
        }

    def train_rnn(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 2000,
        lr: float = 1e-4,
        batch_size: int = 32,
        verbose_every: int = 100,
    ) -> Dict[str, Any]:
        dataset = SequenceDataset(self.covariates_rnn, self.outcomes_rnn, self.seq_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = RNNModel(
            input_size=self.covariates_rnn.shape[2],
            output_size=self.outcomes_rnn.shape[2],
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
                loss = nn.functional.binary_cross_entropy(torch.sigmoid(logits_masked), targets_masked, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)
            if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0 or epoch + 1 == epochs):
                print(f"[train_rnn] epoch {epoch + 1:4d}/{epochs} | loss={avg_loss:.6f}")

        self.rnn_model = model
        return {
            "final_loss": self.loss_history[-1],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "lr": lr,
        }

    def load_rnn(self, model_path: str, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2) -> None:
        model = RNNModel(
            input_size=self.covariates_rnn.shape[2],
            output_size=self.outcomes_rnn.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.rnn_model = model

    def build_generic_env(
        self,
        patient_index: int,
        action_history: Optional[np.ndarray] = None,
        vax_cost: float = 0.04,
        reward_type: str = "log",
    ) -> GenericBoosterEnv:
        if self.rnn_model is None:
            raise ValueError("Load or train the RNN first.")
        if not self.long_format_patients:
            raise ValueError("Run preprocess_long_format(...) first.")

        env = GenericBoosterEnv(
            rnn_model=self.rnn_model,
            patient=self.long_format_patients[patient_index],
            action_col_idx=self.feature_col_index[self.action_col],
            reward_outcome_idx=self.outcome_col_index[self.reward_outcome_col],
            outcome_indices=self.outcome_col_index,
            rl_state_cols=self.rl_state_cols,
            rl_state_maps=self.rl_state_maps,
            vax_cost=vax_cost,
            reward_type=reward_type,
            device=self.device,
            action_history=action_history,
            cumulative_action_col=self.cumulative_action_col,
            months_since_action_col=self.months_since_action_col,
            months_since_action_state_col=self.months_since_action_state_col,
            months_since_action_state_bins=self.months_since_action_state_bins,
            min_action_gap=self.min_action_gap,
        )
        env.attach_feature_index(self.feature_col_index)
        return env

    def simulate_env(
        self,
        n: Optional[int] = None,
        vax_cost: float = 0.04,
        reward_type: str = "log",
        policy_mode: str = "observed",
        q_table: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if not self.long_format_patients:
            raise ValueError("Run preprocess_long_format(...) first.")

        n_patients = len(self.long_format_patients) if n is None else min(int(n), len(self.long_format_patients))
        out_rows = []

        for patient_index in range(n_patients):
            pat = self.long_format_patients[patient_index]
            env = self.build_generic_env(patient_index, vax_cost=vax_cost, reward_type=reward_type)

            start_idx = pat.history_start_idx
            previous_extra_action = False
            tq_state = tuple(env.tq_state)

            for t in range(start_idx, pat.rnn_inputs.shape[0]):
                if t == start_idx:
                    action = int(pat.observed_actions[t])
                else:
                    if previous_extra_action:
                        action = 0
                    elif env.months_since_action <= self.min_action_gap:
                        action = 0
                    else:
                        if policy_mode == "observed":
                            action = int(pat.observed_actions[t])
                        elif policy_mode == "none":
                            action = 0
                        elif policy_mode == "all":
                            action = 1
                        elif policy_mode == "q_table":
                            if q_table is None:
                                raise ValueError("q_table must be provided when policy_mode='q_table'")
                            action = int(np.argmax(q_table[tq_state]))
                        else:
                            raise ValueError("policy_mode must be one of {'observed', 'none', 'all', 'q_table'}")

                    if action == 1:
                        previous_extra_action = True

                _, next_tq_state, reward, done = env.step(action)

                row = {
                    "patient_id": pat.patient_id,
                    "t": t,
                    "action": int(action),
                    "reward": float(reward),
                    "next_month_inf": int(env.next_month_inf),
                    "next_month_severe_inf": int(env.next_month_severe_inf),
                }
                for j, col in enumerate(self.rl_state_cols):
                    row[f"state_{col}"] = int(next_tq_state[j])

                out_rows.append(row)
                tq_state = tuple(next_tq_state)

                if done:
                    break

        return pd.DataFrame(out_rows)

    # action restriction is enforced inside agent.select_action(...)
    def tabular_q_learning(
        self,
        vax_cost: float = 0.04,
        reward_type: str = "log",
        repeats_train_eval: int = 30,
    ) -> Dict[str, Any]:
        if not self.long_format_patients:
            raise ValueError("Run preprocess_long_format(...) first.")

        state_levels = tuple(len(self.rl_state_maps[c]) for c in self.rl_state_cols)

        if vax_cost >= 0.05:
            lr = 0.1
        elif vax_cost >= 0.005:
            lr = 0.01
        elif vax_cost >= 0.0005:
            lr = 0.001
        else:
            lr = 0.0005

        agent = TabularQLearner(
            state_levels=state_levels,
            learning_rate=lr,
            min_learning_rate=lr / 1000.0,
            seed=self.seed,
        )

        epochs_train_eval = np.tile(np.repeat([False, True], [1, 1]), repeats_train_eval)
        epoch_reward_list = np.full((len(epochs_train_eval), len(self.long_format_patients)), np.nan, dtype=np.float32)

        for epoch, is_train in enumerate(epochs_train_eval):
            sample_idx_array = np.random.choice(len(self.long_format_patients), size=len(self.long_format_patients))

            for i, sample_idx in enumerate(sample_idx_array):
                pat = self.long_format_patients[int(sample_idx)]

                if len(pat.action_positions) < 2:
                    continue

                env = self.build_generic_env(int(sample_idx), vax_cost=vax_cost, reward_type=reward_type)
                tq_state = tuple(env.tq_state)
                previous_extra_action = False
                episodic_reward = 0.0
                horizon = max(1, pat.rnn_inputs.shape[0] - pat.history_start_idx)

                for t in range(pat.history_start_idx, pat.rnn_inputs.shape[0]):
                    if t == pat.history_start_idx:
                        action = int(pat.observed_actions[t])
                    else:
                        action = agent.select_action(
                            tq_state,
                            env.months_since_action,
                            previous_extra_action,
                            greedy_only=(not is_train),
                        )
                        if action == 1:
                            previous_extra_action = True

                    _, next_tq_state, reward, done = env.step(action)
                    episodic_reward += reward / horizon

                    if is_train and t != pat.history_start_idx:
                        agent.update(tq_state, action, reward, next_tq_state)

                    tq_state = tuple(next_tq_state)

                    if done:
                        break

                if not is_train:
                    epoch_reward_list[epoch, i] = episodic_reward

        self.q_learner = agent
        return {"q_table": agent.q_table.copy(), "epoch_reward_list": epoch_reward_list}

    def eval_q_learning(
        self,
        policy_mode: str,
        vax_cost: float = 0.04,
        reward_type: str = "log",
        epochs: int = 5,
    ) -> np.ndarray:
        if not self.long_format_patients:
            raise ValueError("Run preprocess_long_format(...) first.")

        out = np.full((epochs, len(self.long_format_patients)), np.nan, dtype=np.float32)

        for epoch in range(epochs):
            sample_idx_array = np.random.choice(len(self.long_format_patients), size=len(self.long_format_patients))

            for i, sample_idx in enumerate(sample_idx_array):
                pat = self.long_format_patients[int(sample_idx)]

                if len(pat.action_positions) < 2:
                    continue

                env = self.build_generic_env(int(sample_idx), vax_cost=vax_cost, reward_type=reward_type)
                previous_extra_action = False
                episodic_reward = 0.0
                tq_state = tuple(env.tq_state)
                horizon = max(1, pat.rnn_inputs.shape[0] - pat.history_start_idx)

                for t in range(pat.history_start_idx, pat.rnn_inputs.shape[0]):
                    if previous_extra_action:
                        action = 0
                    else:
                        if t == pat.history_start_idx:
                            action = int(pat.observed_actions[t])
                        elif env.months_since_action <= self.min_action_gap:
                            action = 0
                        else:
                            if policy_mode == "evaltable":
                                action = int(np.argmax(self.q_learner.q_table[tq_state]))
                            elif policy_mode == "none":
                                action = 0
                            elif policy_mode == "all":
                                action = 1
                            elif policy_mode == "data":
                                action = int(pat.observed_actions[t])
                            else:
                                raise ValueError("policy_mode must be one of {'evaltable', 'all', 'none', 'data'}")

                            if action == 1:
                                previous_extra_action = True

                    _, next_tq_state, reward, done = env.step(action)
                    episodic_reward += reward / horizon
                    tq_state = tuple(next_tq_state)

                    if done:
                        break

                out[epoch, i] = episodic_reward

        return out

    @staticmethod
    def generate_vaccine_pattern(num_vax: int, interval_length: int, from_real_data: Optional[np.ndarray] = None) -> np.ndarray:
        vaccine_pattern = np.zeros(interval_length, dtype=int)
        if from_real_data is None:
            if num_vax > 0:
                if num_vax <= 2:
                    when_to_receive = np.random.choice(np.arange(interval_length), size=num_vax, replace=False)
                elif num_vax == 3:
                    when_to_receive_vax3 = np.random.choice(np.arange(7, interval_length), size=1)
                    when_to_receive_vax12 = np.random.choice(np.arange(when_to_receive_vax3.item() - 5), size=2, replace=False)
                    when_to_receive = np.concatenate((when_to_receive_vax12, when_to_receive_vax3))
                elif num_vax == 4:
                    when_to_receive_vax4 = np.random.choice(np.arange(13, interval_length), size=1)
                    when_to_receive_vax3 = np.random.choice(np.arange(7, when_to_receive_vax4.item() - 5), size=1)
                    when_to_receive_vax12 = np.random.choice(np.arange(when_to_receive_vax3.item() - 5), size=2, replace=False)
                    when_to_receive = np.concatenate((when_to_receive_vax12, when_to_receive_vax3, when_to_receive_vax4))
                else:
                    raise ValueError("num_vax must be <= 4")
                vaccine_pattern[np.asarray(when_to_receive, dtype=int)] = 1
        else:
            vaccine_pattern[np.asarray(from_real_data, dtype=int)] = 1
        return vaccine_pattern

    def build_env(self, patient_index: int, vaccine_hist: np.ndarray, vax_cost: float = 0.04, reward_type: str = "log") -> CovidBoosterEnv:
        return CovidBoosterEnv(
            self.rnn_model,
            self.patient_artifacts[patient_index],
            vaccine_hist.astype(np.float32),
            vax_cost=vax_cost,
            reward_type=reward_type,
            device=self.device,
        )

    def tabular_q_learning_covid(self, vax_cost: float = 0.04, reward_type: str = "log", repeats_train_eval: int = 30) -> Dict[str, Any]:
        if vax_cost >= 0.05:
            lr = 0.1
        elif vax_cost >= 0.005:
            lr = 0.01
        elif vax_cost >= 0.0005:
            lr = 0.001
        else:
            lr = 0.0005

        agent = TabularQLearner(learning_rate=lr, min_learning_rate=lr / 1000.0, seed=self.seed)
        epochs_train_eval = np.tile(np.repeat([False, True], [1, 1]), repeats_train_eval)
        epoch_reward_list = np.full((len(epochs_train_eval), len(self.patient_artifacts)), np.nan, dtype=np.float32)

        for epoch, is_train in enumerate(epochs_train_eval):
            sample_idx_array = np.random.choice(len(self.patient_artifacts), size=len(self.patient_artifacts))
            for i, sample_idx in enumerate(sample_idx_array):
                art = self.patient_artifacts[int(sample_idx)]
                vaccine_pattern = self.generate_vaccine_pattern(0, 27, art.action_positions)
                if vaccine_pattern.sum() < 2:
                    continue

                row_idx = int(np.where(vaccine_pattern)[0][1])
                env = self.build_env(int(sample_idx), vaccine_pattern[: row_idx + 1], vax_cost=vax_cost, reward_type=reward_type)
                tq_state = tuple(env.tq_state)
                previous_booster = False
                episodic_reward = 0.0

                for t in range(row_idx, 27):
                    if t == row_idx:
                        action = int(vaccine_pattern[row_idx])
                    else:
                        action = agent.select_action(tq_state, env.months_last_vax, previous_booster, greedy_only=(not is_train))
                        if action == 1:
                            previous_booster = True

                    _, next_tq_state, reward, done = env.step(action)
                    episodic_reward += reward / (27 - row_idx)

                    if is_train and t != row_idx:
                        agent.update(tq_state, action, reward, next_tq_state)

                    tq_state = tuple(next_tq_state)
                    if done:
                        break

                if not is_train:
                    epoch_reward_list[epoch, i] = episodic_reward

        self.q_learner = agent
        return {"q_table": agent.q_table.copy(), "epoch_reward_list": epoch_reward_list}

    def eval_covid_policy(self, policy_mode: str, vax_cost: float = 0.04, reward_type: str = "log", epochs: int = 5) -> np.ndarray:
        out = np.full((epochs, len(self.patient_artifacts)), np.nan, dtype=np.float32)

        for epoch in range(epochs):
            sample_idx_array = np.random.choice(len(self.patient_artifacts), size=len(self.patient_artifacts))
            for i, sample_idx in enumerate(sample_idx_array):
                art = self.patient_artifacts[int(sample_idx)]
                vaccine_pattern = self.generate_vaccine_pattern(0, 27, art.action_positions)
                if vaccine_pattern.sum() < 2:
                    continue

                row_idx = int(np.where(vaccine_pattern)[0][1])
                env = self.build_env(int(sample_idx), vaccine_pattern[: row_idx + 1], vax_cost=vax_cost, reward_type=reward_type)
                previous_booster = False
                episodic_reward = 0.0
                tq_state = tuple(env.tq_state)

                if policy_mode == "all" and (row_idx + 1) < 27:
                    vaccine_pattern[(row_idx + 1) : 27] = 0
                    if (row_idx + 4) < 27:
                        vaccine_pattern[np.random.choice(np.arange(row_idx + 4, 27), size=1)] = 1

                for t in range(row_idx, 27):
                    if previous_booster:
                        action = 0
                    else:
                        if t == row_idx:
                            action = int(vaccine_pattern[row_idx])
                        elif env.months_last_vax <= 4:
                            action = 0
                        else:
                            if policy_mode == "evaltable":
                                action = int(np.argmax(self.q_learner.q_table[tq_state]))
                            elif policy_mode == "none":
                                action = 0
                            elif policy_mode in {"all", "data"}:
                                action = int(vaccine_pattern[t])
                            else:
                                raise ValueError("policy_mode must be one of {'evaltable', 'all', 'none', 'data'}")
                            if action == 1:
                                previous_booster = True

                    _, next_tq_state, reward, done = env.step(action)
                    episodic_reward += reward / (27 - row_idx)
                    tq_state = tuple(next_tq_state)
                    if done:
                        break

                out[epoch, i] = episodic_reward

        return out
