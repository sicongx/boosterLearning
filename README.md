# boosterLearning

A user-friendly reorganization of the COVID-19 booster learning code for recurrent neural network (RNN) microsimulation and tabular Q-learning.

This repository provides a generic trajectory-based interface for sequential decision learning, together with a COVID-19 booster example that reproduces the workflow in the paper. For public release, the intended example dataset is the synthesized `facsimile_data.csv`, which has the same structure as the original long-format EHR file but contains only synthetic entries.

## What this repository is for

The original project combined several separate scripts for:

- transforming long-format EHR data into padded patient trajectories,
- training an RNN environment simulator,
- simulating patient trajectories under different vaccination choices,
- training a tabular Q-learning policy, and
- evaluating learned and comparison policies.

This repository reorganizes that workflow around a cleaner interface so that users can:

1. start from long-format trajectory data,
2. specify which columns are RNN covariates, RNN outcomes, and RL states,
3. train or load an RNN sequence model,
4. train tabular Q-learning through a generic microsimulation environment, and
5. evaluate learned or user-defined policies.

The main user-facing code is:

- `booster_learning_aligned.py`
- `covid_generic_sanity_check.py`

## Current recommended entry points

### `booster_learning_aligned.py`

This is the main module. It defines the generic classes used throughout the reorganized workflow.

#### `TrajectoryDataset`

This class converts long-format data into the padded sequence representation used by the RNN and RL code. It:

- sorts trajectories by patient and time,
- builds padded arrays for RNN covariates and outcomes,
- stores sequence lengths,
- records patient-level trajectory artifacts, and
- builds mappings for discrete RL state variables.

Use this class when you already have long-format data and want to tell the framework which columns should be treated as:

- RNN inputs,
- RNN outputs, and
- RL state variables.

#### `MicrosimQLearner`

This is the main learner class. It wraps the full workflow needed for this project:

- `fit_sequence_model(...)` trains the RNN environment simulator.
- `load_sequence_model(...)` loads a pretrained RNN.
- `build_env(...)` builds a generic trajectory environment for one patient.
- `simulate(...)` rolls out a chosen policy inside the learned environment.
- `fit_tabular_q_learning(...)` trains the tabular Q-learning policy.
- `evaluate_policy(...)` evaluates learned, observed, or user-defined policies.

The class is generic by design. Domain-specific behavior is supplied through optional hooks such as:

- reward functions,
- action constraints,
- episode start rules,
- transition updates, and
- terminal conditions.

That separation is what allows the repository to keep the COVID example while also exposing a reusable interface.

### `covid_generic_sanity_check.py`

This is the main worked example and validation script.

It shows how to use the generic classes in `booster_learning_aligned.py` with the COVID-19 booster setting from the paper. In particular, it:

- loads the long-format COVID data,
- reconstructs the original 16-dimensional RNN covariates,
- builds COVID-specific hooks on top of the generic interface,
- either loads pretrained RNN weights or optionally refits the RNN,
- trains tabular Q-learning through the generic learner, and
- evaluates four policies: learned, observed data, always-booster, and never-booster.

For publication and code release, this script should be run with the synthesized facsimile data file rather than the original private EHR data.

## Public example data

The only example dataset intended for public use in this repository is:

- `facsimile_data.csv`

This file is the public code-illustration dataset. It is synthesized and is meant to replace the original private `RLdata_for_RNN_01242024.csv` in examples, documentation, and released code.

In other words:

- do **not** treat the private original data file as part of the public workflow,
- do use `facsimile_data.csv` as the example long-format input when demonstrating the code.

## Typical workflow

### 1. Prepare long-format data

Your input data should be in long format, with one row per patient-time point.

Users are expected to provide:

- a patient ID column,
- a time-order column,
- an action column,
- columns used as RNN covariates,
- columns used as RNN outcomes, and
- columns used as discrete RL states.

Continuous variables should already be scaled if needed, and categorical variables should already be encoded in the way the user wants to use them.

### 2. Build a dataset

```python
from booster_learning_aligned import TrajectoryDataset

dataset = TrajectoryDataset.from_long_format(
    df=long_df,
    patient_id_col="id",
    time_col="month_index",
    action_col="action",
    rnn_covariate_cols=[...],
    rnn_outcome_cols=[...],
    rl_state_cols=[...],
    reward_outcome_col="sev_inf_next",
    seed=2024,
)
```

### 3. Create a learner

```python
from booster_learning_aligned import MicrosimQLearner

learner = MicrosimQLearner(
    dataset=dataset,
    seed=2024,
    reward_fn=my_reward_fn,
    action_constraint_fn=my_action_constraint_fn,
    transition_fn=my_transition_fn,
    terminal_fn=my_terminal_fn,
)
```

All hook functions are optional. If they are not provided, the module falls back to generic default behavior.

### 4. Train or load the RNN

Train a new sequence model:

```python
learner.fit_sequence_model(
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    epochs=2000,
    lr=1e-4,
    batch_size=32,
)
```

Or load a pretrained model:

```python
learner.load_sequence_model(
    model_path="data/rnn_weights_2_128_2000_1e-04.pth",
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
)
```

### 5. Train tabular Q-learning

```python
fit_out = learner.fit_tabular_q_learning(
    repeats_train_eval=30,
    gamma=0.99,
    learning_rate=0.01,
    learning_rate_decay=0.998,
    min_learning_rate=1e-5,
    epsilon=0.5,
    epsilon_decay=0.99,
    decay_every=5000,
)
```

The returned dictionary contains the learned Q-table and the train/eval reward array.

### 6. Evaluate policies

```python
rewards = learner.evaluate_policy(policy="learned", epochs=5)
```

You may also pass a callable policy if you want custom logic.

## Quick start with the COVID example

Assuming you want to run the public example with the synthesized facsimile data and pretrained RNN weights:

```bash
python covid_generic_sanity_check.py \
  --rnn-data facsimile_data.csv \
  --weights data/rnn_weights_2_128_2000_1e-04.pth \
  --reward-type prop \
  --vax-cost 0.04 \
  --train-repeats 1 \
  --eval-epochs 1 \
  --seed 2024
```

A heavier setting, closer to the paper workflow, is:

```bash
python covid_generic_sanity_check.py \
  --rnn-data facsimile_data.csv \
  --weights data/rnn_weights_2_128_2000_1e-04.pth \
  --reward-type prop \
  --vax-cost 0.04 \
  --train-repeats 30 \
  --eval-epochs 5 \
  --seed 2024
```

Optional outputs:

```bash
python covid_generic_sanity_check.py \
  --rnn-data facsimile_data.csv \
  --weights data/rnn_weights_2_128_2000_1e-04.pth \
  --save-q-table results/q_table.npy \
  --save-results results/seed_2024.npz
```

## File-by-file guide

### Main reorganized files

#### `booster_learning_aligned.py`
The main generic implementation. This is the module users should read first if they want to understand the new class-based interface.

#### `covid_generic_sanity_check.py`
The main example runner showing how the generic interface is instantiated for the COVID booster application.

#### `aggregate_summary_rewards.py`
A utility script for combining per-seed saved results into summary tables. It reads saved `.npz` result files, extracts metrics such as the rewards for the learned, observed, all, and none policies, and writes aggregated CSV summaries. This is useful after running multiple seeds and wanting one final reporting table.

#### `facsimile_data.csv`
The synthesized long-format example dataset intended for the public repository and documentation.

### Legacy reference files

These files reflect the older script-by-script workflow. They are still useful as references for how the original implementation was organized, but they are no longer the primary interface.

#### `create_rnn_data.ipynb`
This notebook is the original preprocessing pipeline. It created the RNN-ready arrays from the raw long-format EHR data. In the reorganized code, its role is largely replaced by `TrajectoryDataset.from_long_format(...)` plus the COVID-specific transformation logic in `covid_generic_sanity_check.py`.

#### `train_rnn.py`
This is the original standalone RNN training script. In the reorganized version, its job is absorbed by `MicrosimQLearner.fit_sequence_model(...)`.

#### `simulate_env.py`
This script simulates trajectories using the trained RNN environment. Conceptually, this behavior now lives inside `GenericTrajectoryEnv` and `MicrosimQLearner.simulate(...)`.

#### `q_learning_table (1).py`
This is the original standalone tabular Q-learning training script. Its functionality is now covered by `MicrosimQLearner.fit_tabular_q_learning(...)`.

#### `q_learning_eval (2).py`
This is the original standalone evaluation script for learned and comparison policies. Its role is now covered by `MicrosimQLearner.evaluate_policy(...)` and the evaluation logic in `covid_generic_sanity_check.py`.

#### `q_learning.py`
This is the older deep Q-learning experiment script. It is included mainly for comparison with the tabular Q-learning approach discussed in the paper, not as the recommended workflow for this repository.

#### `helpers.py`
This file contains older helper classes and utility functions used by the original scripts, including the earlier RNN definition, the old booster environment, deep Q-learning components, vaccine pattern generation, and rate summaries. Some of its utility functions are still useful for compatibility and reporting, but the main new interface lives in `booster_learning_aligned.py`.

#### `testmdp.py`
This is an auxiliary script related to transition testing / MDP-style checks on simulated datasets. It is not required for the standard public example workflow.

## How the new files relate to the old files

The repository can be understood as a reorganization rather than a completely separate implementation.

- `create_rnn_data.ipynb` -> preprocessing logic is now expressed through `TrajectoryDataset` and the COVID example loader.
- `train_rnn.py` -> replaced by `MicrosimQLearner.fit_sequence_model(...)`.
- `simulate_env.py` -> replaced by the generic environment and simulation methods.
- `q_learning_table (1).py` -> replaced by `MicrosimQLearner.fit_tabular_q_learning(...)`.
- `q_learning_eval (2).py` -> replaced by `MicrosimQLearner.evaluate_policy(...)` and the example runner.
- `helpers.py` -> legacy support code and utility functions retained for continuity.
- `aggregate_summary_rewards.py` -> post-processing helper for multi-seed summaries.

So the main design shift is:

- **old structure:** many separate scripts, each handling one stage,
- **new structure:** one generic dataset class, one generic learner class, and one example script that wires them together for COVID.

## Notes for users

- The COVID example is meant to demonstrate how to specialize the generic interface through hook functions.
- The public example should use `facsimile_data.csv` as the input data illustration.
- The repository is organized so that COVID is the worked example, not the only possible use case.
- Users with other long-format sequential decision data can adapt the same interface by changing the selected columns and supplying custom hooks.

## Minimal import example

```python
from booster_learning_aligned import TrajectoryDataset, MicrosimQLearner
```

These are the two main objects most users need.

## Citation and data-use note

If you use this repository in academic work, please cite the corresponding paper.

The public repository example is based on synthesized facsimile data for code illustration. The original private EHR data used in the research workflow is not part of the public release.
