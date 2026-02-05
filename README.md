# RL_cilia

A small, self-contained MATLAB project for exploring **tabular Q-learning** on a “shape space” model inspired by cilia/filament centerlines.

We represent a 2D inextensible centerline by a **low-dimensional curvature model** (Chebyshev basis), then train a Q-learning agent to adjust a small set of shape coefficients to improve a simple objective.

---

## What’s in this repo

### Core idea

* Parameterize curvature $\kappa(s)$ with **3 coefficients** (c = [c0,c1,c2]) (integers in ([-4,4])).
* Reconstruct the curve by integrating:

  * $\theta'(s) = \kappa(s)$
  * $x'(s) = \cos\theta(s),; y'(s) = \sin\theta(s)$
* Enforce a **free-tip** moment condition by construction: $\kappa(L)=0$.

### Task A (current)

**Maximize side-to-side sweep** of the filament while discouraging:

- wall crossings `y(s) < 0` (soft penalty + optional “crash” termination)
- excessive bending (bending-energy penalty)

**RL interpretation (state / action / episode).** We treat “designing a good cilium shape” as a small reinforcement-learning environment. A **state** is a discrete coefficient vector `c = [c0,c1,c2]` (each integer in `[-4,4]`) that parameterizes the curvature of a centerline via a Chebyshev basis. An **action** is a local edit: increase or decrease exactly one coefficient by 1 (6 actions total). An **episode** starts from the straight shape `c = (0,0,0)` and applies a fixed number of edits `H`; at the end of the episode we evaluate the **final** curve and reward large side-to-side sweep (`span = max(x) - min(x)`) while penalizing bending and any wall violation (`y < 0`). After training, we run a **greedy rollout** (no exploration) by repeatedly taking the action with the largest learned Q-value to produce the final designed shape.


**Objective:**
- `span = max_s x(s) - min_s x(s)` (lateral sweep of the centerline)

Notes:
- The base is clamped (fixed position and orientation). In the demos we typically set `theta0 = pi/2` so the filament starts pointing upward.


---

## Folder layout 

```
RL_cilia/
  README.md
  src/        % functions (curve model, reward, stepping, plotting helpers)
  demos/      % scripts you run (training, gallery, etc.)
```

---

## Requirements

* MATLAB R2020b+ (should work on newer versions)
* No toolboxes required

---

## Quick start

1. Clone the repo and open MATLAB in the repo folder.
2. Run the main demo:

```matlab
% From MATLAB, in repo root:
run("demos/main_qlearning_cilia_taskA.m")
```

If MATLAB can’t find functions in `src/`, add the folder to your path:

```matlab
addpath("src")
```

---

## Key scripts

### `demos/main_qlearning_cilia_taskA.m`

Trains a tabular Q-learning agent over the 3D coefficient grid and produces:

* training return plot
* final greedy shape plot (base/tip + wall y=0)
* curvature plot
* (optional) value slice / policy arrows
* (optional) best-episode shape

### Gallery 

A quick visualization to build intuition for how coefficients affect shape:

```matlab
valsToShow = [-4 -2 0 2 4];
cilia_gallery_shapes(-4, 4, 0, 1.0, 400, params, valsToShow);
```

---

## Model details 

* Arclength parameter: $s \in [0,L]$
* Map to Chebyshev coordinate: $\xi = 2s/L - 1$
* Curvature basis:
  
  $$\kappa(s) = (1-\xi)\sum_{n=0}^{N-1} c_n T_n(\xi)$$
  so that $\kappa(L)=0$ automatically $(\xi(L)=1)$.

---

## Parameters you’ll likely tweak

In `main_qlearning_cilia_taskA.m`:

* Coefficient bounds: `cmin=-4`, `cmax=4`
* Episode horizon: `H`
* Q-learning: `alpha0`, `alphaMin`, `gamma`, `eps0`, `epsMin`, `N_EPISODES`
* Reward weights:

  * `params.wx` (forward reach)
  * `params.wb` (bending penalty)
  * `params.mu_wall`, `params.ww` (wall penalty strength)
  * `params.terminate_minY` (hard termination threshold)

---

## Suggested student extensions

* Increase modes from N=3 to N=5 (Q-table becomes large → motivate function approximation)
* Replace Q-table with linear features or a small neural network approximation of (Q(s,a))
* Change the task:

  * target-reaching
  * maximize swept area $\int y(s) ds$
* Add a smoother free-tip condition by using a $(1-\xi)^2$ prefactor (enforces $\kappa(L)=0$ and $\kappa'(L)=0)$

