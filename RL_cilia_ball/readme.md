
# Cilia-Ball Tabular RL (Q-learning) — README

This folder turns the **2-ball cilia MATLAB model** into a **tabular reinforcement learning (RL)** environment so students can implement **Q-learning** on a finite state/action space.

The physics (flux reward) comes from the original cilia-ball code. The RL wrapper discretizes joint angles into a grid, enumerates discrete actions, and provides a standard `reset/step` interface.

---

## What is the RL problem?

### State
A state is the pair of **discrete joint angles**
\[
s \equiv (\phi_1,\phi_2)
\]
where each angle is snapped to a grid on `[-phimax(k), phimax(k)]` using `P.Nstates(k)` points.

- Example defaults: `P.Nstates = [11, 21]`
- Total number of states: `prod(P.Nstates)`

### Action
An action is a pair of increments
\[
a \equiv (a_1,a_2),\quad a_k \in \{-1,0,1\}
\]
meaning: “move one grid step left / stay / move one grid step right” in each angle.

- Total number of actions: `3^Nh` where `Nh = number of hinges` (here `Nh=2`)

### Reward
The reward is the **flux in the x-direction** returned by the physics code:

```matlab
[r, ~] = cilia_ball_reward(phi, action, P);
````

### Transition / boundaries

Angles are updated by one grid step (or stay), then **clipped to stay within bounds**.
If an action tries to step out of bounds, it is automatically reduced so the system stays on the boundary.

---

## Files

### RL Wrapper

* `ciliaBallTabularEnv.m`
  Creates the environment struct with:

  * `env.reset()` → initial state index
  * `env.step(s, aIdx)` → next state + reward (+ optional info)
  * `env.render(s)` → plot the current configuration (if plotting helpers exist)
  * optional `precompute` mode to build reward/transition lookup tables

### Q-learning (minimal)

* `qlearn_tabular.m`
  A simple tabular Q-learning loop (epsilon-greedy).

### Physics / model code (required)

These must be on the MATLAB path:

* `setdefaultparams_ciliaball.m`
* `cilia_ball_reward.m`
* `position_from_angle.m`
* `velocity_from_angvel.m`
* `angvel_from_action.m`
* `form_stokes_image_system_3D_cm.m`
  (and any dependencies those call)

---

## Quick start

```matlab
P = setdefaultparams_ciliaball;

% Create environment
env = ciliaBallTabularEnv(P, struct( ...
  'reset_mode','fixed', ...
  'phi0',[0, -P.phimax(2)], ...
  'precompute', true));     % set false if you want direct calls each step

% Train Q-learning
[Q, epReturn] = qlearn_tabular(env, 200, 50, 0.2, 0.95, 0.2);

% Greedy rollout + visualization
s = env.reset();
for t = 1:60
  env.render(s);
  [~, a] = max(Q(s,:));
  [s, r] = env.step(s, a);
  fprintf('t=%d, r=%g\n', t, r);
end
```


## Episodes vs steps

* **Step:** one call to `env.step(s,a)` (one discrete angle move).
* **Episode:** many steps until a stopping rule is met.

There is no “goal state” here, so common episode endings are:

1. **Fixed horizon** (e.g., 50 steps), and/or
2. **Limit-cycle detection**: stop when the rollout repeats a previously visited state.

### Limit-cycle termination (recommended for greedy evaluation)

During evaluation with `epsilon = 0`, terminate the episode when a state repeats.
That indicates the policy has entered a cycle (periodic motion).

(If using this during training with `epsilon>0`, use a “confirmed cycle” rule or prefer a fixed horizon.)

---

## Notes / suggestions for students

* Start by confirming you can:

  1. construct `env`, 2) take a few random steps, 3) plot with `env.render`.
* Then implement epsilon-greedy Q-learning.
* After training, evaluate with `epsilon = 0` and look for a stable limit cycle.
* Compare policies by:

  * mean reward per step,
  * mean reward over one full cycle,
  * cycle period.

---

## Troubleshooting

* If you see “function not found”: add the physics code folder to your path:

  ```matlab
  addpath(genpath(pwd))
  ```
* If precompute is slow: set `'precompute', false` while debugging, then enable once stable.
* If plots are missing: make sure `position_from_angle.m` is on the path.

---

```
```
