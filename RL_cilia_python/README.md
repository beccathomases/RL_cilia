# RL cilia Python benchmark

Current working directory for the discrete 2-ball cilia reinforcement-learning benchmark:

`/Users/bthomases/Documents/Student_Projects/RL_cilia/RL_cilia_python`

## Main purpose

This folder contains the current Python implementation of the small discrete 2-ball cilia control problem, used as a benchmark for comparing:

- value iteration
- tabular Q-learning
- PPO
- DQN

This is the canonical working location for the Python transition of the RL cilia project.

## Canonical current files

### Environment
- `cilia_2_ball_env.py`  
  Current canonical Python environment, using the cumulative-angle (relative-angle) representation.

- `cilia_2_ball_env_notcummang.py`  
  Older/non-cumulative-angle environment kept for comparison only.

### Main solvers
- `value_iteration_cilia_2_ball.py`  
  Value-iteration solver for the fixed discrete benchmark.

- `tabular_q_cilia_2_ball.py`  
  Tabular Q-learning code for the same benchmark.

- `cilia_2_ball_ppo.py`  
  PPO training script.

- `cilia_2_ball_deepQ.py`  
  DQN / DeepQ training script.

### Diagnostics / comparisons
- `plot_phase_plane.py`
- `plot_tabular_q_seed.py`
- `movie_best_VI_cycle.py`
- `evaluate_ppo_cilia_2_ball.py`
- `evaluate_dqn_cilia_2_ball.py`
- `compare_all_methods_cilia_2_ball.py`
- `compare_matlab_python.py`
- `compare_discounted_cycle_values.py`
- `compare_dqn_policy_to_VI.py`

## Canonical saved outputs

### Value iteration
- `value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy`  
  Current canonical VI result for the clip-penalty benchmark.

- `value_iteration_cilia_2_ball_stay_penalty_bins11x21_g0.990.npy`  
  VI result for the stay-penalty variant.

### Tabular Q
- `tabular_q_cilia_2_ball_clip_penalty_bins11x21_ep1000_steps500_g0.990_eps0.75_a0.99.npy`  
  Canonical tabular-Q result for the clip-penalty benchmark.

- `tabular_q_cilia_2_ball_stay_penalty_bins11x21_ep1000_steps500_g0.990_eps0.75_a0.99.npy`  
  Tabular-Q result for the stay-penalty variant.

### PPO
- `ppo_model.zip`  
  Saved PPO model.

### DQN
- `dqn_cilia_2_ball.pt`  
  Saved DQN network weights.

- `dqn_cilia_2_ball_results.npy`  
  Saved DQN diagnostics and cycle information.

## Current benchmark status

For the current cumulative-angle Python environment with clip-penalty boundary handling:

- value iteration converges to a unique length-24 discounted-optimal cycle
- tabular Q-learning recovers the same cycle (up to cyclic phase shift)
- PPO also recovers the same cycle
- DQN finds a nearby but suboptimal longer cycle
- the Python cycle agrees with the validated MATLAB cycle up to cyclic phase shift

## Quick rerun guide

### Recompute value iteration
```bash
python value_iteration_cilia_2_ball.py
````

### Recompute tabular Q-learning

```bash
python tabular_q_cilia_2_ball.py
```

### Train PPO

```bash
python cilia_2_ball_ppo.py
```

### Train DQN

```bash
python cilia_2_ball_deepQ.py
```

### Compare all methods

```bash
python compare_all_methods_cilia_2_ball.py
```

### Compare DQN policy to VI value function

```bash
python compare_dqn_policy_to_VI.py
```

## Notes

* The current canonical environment is `cilia_2_ball_env.py`.
* Older iteration counts or older saved files may correspond to earlier versions of the environment or reward scaling.
* If results look inconsistent, rerun value iteration first and treat that output as the canonical current benchmark.

