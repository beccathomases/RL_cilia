# RL_cilia_python

Top-level working directory for the Python side of the RL cilia project.

## Directory role

This directory is now organized as a project hub rather than a single benchmark folder. In particular, it contains:

- a **frozen Phase I discrete benchmark snapshot**
- older exploratory and historical material
- a place for future Phase II development

The main canonical benchmark for the small discrete 2-ball problem is no longer the top level of this directory. It now lives in:

- `phase1_discrete_benchmark/`

## Current directory structure

### `phase1_discrete_benchmark/`
Frozen Phase I benchmark snapshot for the small discrete 2-ball cilia-control problem.

This folder contains:
- the canonical cumulative-angle Python environment
- benchmark scripts for value iteration, tabular Q-learning, PPO, and DQN
- analysis scripts for method comparison and the gamma sweep
- saved benchmark results and models
- benchmark figures
- the current Phase I notes/writeup

This is the main place to look for the locked benchmark state.

### `phase2_exploration/`
Working area for future development beyond the frozen Phase I benchmark.

Intended uses include:
- new environments
- reward-design experiments
- larger cilia models
- additional RL methods
- refactoring and cleaner package structure

### `exploratory_old/`
Older exploratory, noncanonical, or superseded material retained for reference.

This includes:
- historical archives
- reproducibility checks
- older sweeps
- scratch scripts
- alternative environment variants

These files are not part of the frozen benchmark and should not be treated as canonical results.

### `old_top_level_copies/`
Copies of scripts and files that used to live at the top level before the Phase I benchmark was frozen into its own folder.

This folder is mainly for transition/backup purposes and is not intended as the main working location.

### `figures/`
Top-level figure staging area, if used. The canonical benchmark figures are stored inside `phase1_discrete_benchmark/figures/`.

## Recommended workflow

### For the frozen benchmark
Use:
- `phase1_discrete_benchmark/`

This is the correct location for:
- rerunning the benchmark
- reproducing the Phase I figures
- checking the benchmark notes/writeup

### For new experiments
Use:
- `phase2_exploration/`

This is the preferred place for new work so that the frozen benchmark remains unchanged.

### For historical reference
Look in:
- `exploratory_old/`
- `old_top_level_copies/`

## Benchmark summary

For the current frozen Phase I benchmark in `phase1_discrete_benchmark/`:

- value iteration finds a unique length-24 discounted-optimal cycle
- tabular Q-learning recovers the same cycle up to cyclic phase shift
- PPO recovers the same cycle up to cyclic phase shift
- DQN is reproducible under corrected seeding and converges to a nearby length-25 suboptimal cycle
- the Python VI cycle agrees with the previously validated MATLAB cycle up to cyclic phase shift

## Notes

- Treat `phase1_discrete_benchmark/` as read-only except for carefully documented fixes.
- Do new development outside the frozen benchmark.
- The top-level directory is now mainly organizational.