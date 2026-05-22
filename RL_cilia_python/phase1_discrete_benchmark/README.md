# Phase I discrete benchmark: RL cilia 2-ball model

This folder contains the locked Phase I benchmark for the small discrete 2-ball cilia-control problem.

## Purpose

The goal of this benchmark is to provide a clean, reproducible Python implementation of the canonical discrete 2-ball environment and the main comparison methods:

- value iteration
- tabular Q-learning
- PPO
- DQN

together with the scripts needed to regenerate the main benchmark figures and notes.

## Canonical benchmark setup

- Environment: `cilia_2_ball_env.py`
- State space: discrete 2-angle grid with `n_bins = [11, 21]`
- Boundary handling: `clip_penalty`
- Invalid-action penalty: `-0.1`
- Reward rescaling: `100.0`
- Discount factor for main benchmark: `gamma = 0.99`

## Main benchmark conclusion

For the current canonical benchmark:

- value iteration finds a unique length-24 recurrent cycle with average reward approximately `0.376394`
- tabular Q-learning recovers the same cycle up to cyclic shift
- PPO recovers the same cycle up to cyclic shift
- DQN is reproducible under corrected seeding and converges to a nearby but distinct length-25 cycle with average reward approximately `0.355267`

The Python VI cycle also agrees with the previously validated MATLAB cycle up to cyclic phase shift.

## Suggested run order

1. `value_iteration_cilia_2_ball.py`
2. `tabular_q_cilia_2_ball.py`
3. `cilia_2_ball_ppo.py` or load the saved PPO model
4. `cilia_2_ball_deepQ.py`
5. `compare_all_methods_cilia_2_ball.py`
6. `gamma_sweep_value_iteration.py`
7. `plot_gamma_sweep_vi.py`

## Main outputs

### Results
- value iteration result file
- tabular Q-learning result file
- DQN result file
- gamma sweep summary/results

### Figures
- VI cycle and reward trace
- all-method phase-plane comparison
- all-method reward traces
- all-method stroke overlays
- V* heatmap
- gamma sweep figures

## Notes

This folder is intended to remain fixed as the Phase I benchmark snapshot.
New exploratory work should be done outside this folder and copied in only if it becomes part of the canonical benchmark.