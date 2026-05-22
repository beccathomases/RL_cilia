# plot_gamma_sweep_vi.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# User settings
# ============================================================

SUMMARY_CSV = "gamma_sweep_value_iteration_summary.csv"
RESULTS_NPY = "gamma_sweep_value_iteration_results.npy"
FIGDIR = "figures"

# 4 gammas to show in tiled stroke plots
STROKE_GAMMAS = [0.990, 0.995, 0.997, 0.999]

SHOW_PLOTS = True


# ============================================================
# Helpers
# ============================================================

def ensure_figdir(figdir):
    os.makedirs(figdir, exist_ok=True)


def load_results_dict(fname):
    data = np.load(fname, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("Expected results npy to contain a dict keyed by gamma.")
    return data


def get_best_cycle(entry):
    if "cycles" not in entry or len(entry["cycles"]) == 0:
        raise ValueError(f"No cycle information found in entry. Keys: {list(entry.keys())}")
    return entry["cycles"][0]


def state_to_angles(env, state):
    s = np.array(state, dtype=int)
    return env.state_to_angles(s)


def state_to_shape_xz(env, state):
    """
    Physical 2-ball cilium shape in x-z plane using cumulative-angle convention.
    """
    phi1, phi2 = state_to_angles(env, state)

    psi1 = phi1
    psi2 = phi1 + phi2

    x0 = env.X0[0]
    z0 = env.X0[2]

    x1 = x0 + env.len * np.sin(psi1)
    z1 = z0 + env.len * np.cos(psi1)

    x2 = x1 + env.len * np.sin(psi2)
    z2 = z1 + env.len * np.cos(psi2)

    return np.array([x0, x1, x2]), np.array([z0, z1, z2])


def close_curve(arr):
    arr = np.asarray(arr)
    return np.vstack([arr, arr[0]])


def match_entries_by_gamma(results_list, requested_gammas, tol=1e-12):
    matched = []
    for g in requested_gammas:
        candidates = [entry for entry in results_list if abs(entry["_gamma"] - g) < tol]
        if len(candidates) == 0:
            print(f"Warning: gamma={g:.3f} not found.")
        else:
            matched.append(candidates[0])
    return matched


# ============================================================
# Load data
# ============================================================

ensure_figdir(FIGDIR)

summary_df = pd.read_csv(SUMMARY_CSV).sort_values("gamma").reset_index(drop=True)
results_dict = load_results_dict(RESULTS_NPY)

results_list = []
for g in sorted(float(k) for k in results_dict.keys()):
    entry = dict(results_dict[g])
    entry["_gamma"] = float(g)
    results_list.append(entry)

env = Cilia2BallEnv(precompute=False)


# ============================================================
# Collect cycle angle data for shared axis limits
# ============================================================

all_cycle_angles = []
for entry in results_list:
    cyc = get_best_cycle(entry)
    states = cyc["cycle"]
    ang = np.array([state_to_angles(env, s) for s in states], dtype=float)
    all_cycle_angles.append(ang)

all_phi1 = np.concatenate([a[:, 0] for a in all_cycle_angles])
all_phi2 = np.concatenate([a[:, 1] for a in all_cycle_angles])

xpad = 0.05 * max(1e-12, all_phi1.max() - all_phi1.min())
ypad = 0.05 * max(1e-12, all_phi2.max() - all_phi2.min())

phi1_lim = (all_phi1.min() - xpad, all_phi1.max() + xpad)
phi2_lim = (all_phi2.min() - ypad, all_phi2.max() + ypad)


# ============================================================
# Figure 1: average reward vs gamma
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(summary_df["gamma"], summary_df["avg_reward"], "-o", linewidth=2, markersize=8)
plt.xlabel(r"$\gamma$")
plt.ylabel("Average reward per cycle")
plt.title("Average reward of optimal cycle vs discount factor")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "gamma_sweep_avg_reward_vs_gamma.png"), dpi=200)


# ============================================================
# Figure 2: cycle length vs gamma
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(summary_df["gamma"], summary_df["cycle_length"], "-o", linewidth=2, markersize=8)
plt.xlabel(r"$\gamma$")
plt.ylabel("Cycle length")
plt.title("Optimal cycle length vs discount factor")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "gamma_sweep_cycle_length_vs_gamma.png"), dpi=200)


# ============================================================
# Figure 3: tiled phase-plane plots for ALL gammas
# ============================================================

n = len(results_list)
ncols = min(4, n)
nrows = int(math.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
axes = np.array(axes).reshape(-1)

for ax, entry in zip(axes, results_list):
    gamma = entry["_gamma"]
    cyc = get_best_cycle(entry)
    states = cyc["cycle"]

    ang = np.array([state_to_angles(env, s) for s in states], dtype=float)
    ang_closed = close_curve(ang)

    nseg = len(ang_closed) - 1
    for j in range(nseg):
        color = plt.cm.viridis(j / max(nseg - 1, 1))
        ax.plot(
            ang_closed[j:j+2, 0],
            ang_closed[j:j+2, 1],
            "-o",
            color=color,
            linewidth=2.0,
            markersize=4,
            alpha=0.9,
        )

    ax.plot(ang[0, 0], ang[0, 1], "o", color="black", markersize=7, zorder=5)
    if len(ang) > 1:
        ax.plot(ang[1, 0], ang[1, 1], "o", color="red", markersize=6, zorder=5)

    ax.set_title(rf"$\gamma={gamma:.3f}$" + "\n" +
                 f"len={cyc['length']}, avg={cyc['avg_reward']:.4f}")
    ax.set_xlabel(r"$\phi_1$ (rad)")
    ax.set_ylabel(r"$\phi_2$ (rad)")
    ax.set_xlim(phi1_lim)
    ax.set_ylim(phi2_lim)
    ax.grid(True)

for k in range(n, len(axes)):
    axes[k].axis("off")

fig.suptitle("Optimal recurrent cycles in phase plane", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIGDIR, "gamma_sweep_phase_plane_tiled.png"), dpi=200)


# ============================================================
# Figure 4: reward traces overlay
# ============================================================

plt.figure(figsize=(10, 7))
for entry in results_list:
    gamma = entry["_gamma"]
    cyc = get_best_cycle(entry)
    rewards = np.array(cyc["rewards"], dtype=float)

    plt.plot(
        np.arange(1, len(rewards) + 1),
        rewards,
        "-o",
        linewidth=2,
        markersize=5,
        label=rf"$\gamma={gamma:.3f}$"
    )

plt.xlabel("Step in cycle")
plt.ylabel("Reward")
plt.title("Reward traces along optimal cycles")
plt.grid(True)
plt.legend(loc="best", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "gamma_sweep_reward_traces.png"), dpi=200)


# ============================================================
# Figure 5: tiled stroke plots for selected gammas
# ============================================================

selected_entries = match_entries_by_gamma(results_list, STROKE_GAMMAS)
m = len(selected_entries)
if m == 0:
    raise ValueError("None of the requested STROKE_GAMMAS were found in results.")

stroke_ncols = 2 if m > 1 else 1
stroke_nrows = int(math.ceil(m / stroke_ncols))

fig, axes = plt.subplots(
    stroke_nrows,
    stroke_ncols,
    figsize=(6.0 * stroke_ncols, 5.0 * stroke_nrows)
)
axes = np.array(axes).reshape(-1)

for ax, entry in zip(axes, selected_entries):
    gamma = entry["_gamma"]
    cyc = get_best_cycle(entry)
    states = cyc["cycle"]

    nstates = len(states)
    for j, s in enumerate(states):
        x, z = state_to_shape_xz(env, s)
        color = plt.cm.viridis(j / max(nstates - 1, 1))
        lw = 2.0

        if j == 0:
            color = "black"
            lw = 3.0
        elif j == 1:
            color = "red"
            lw = 2.5

        ax.plot(x, z, "-o", color=color, linewidth=lw, markersize=4, alpha=0.9)

    ax.set_title(rf"$\gamma={gamma:.3f}$" + "\n" +
                 f"len={cyc['length']}, avg={cyc['avg_reward']:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.grid(True)

for k in range(m, len(axes)):
    axes[k].axis("off")

fig.suptitle("Stroke plots for selected discount factors", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIGDIR, "gamma_sweep_selected_strokes_tiled.png"), dpi=200)


# ============================================================
# Done
# ============================================================

print("\nSaved figures:")
print(f"  {os.path.join(FIGDIR, 'gamma_sweep_avg_reward_vs_gamma.png')}")
print(f"  {os.path.join(FIGDIR, 'gamma_sweep_cycle_length_vs_gamma.png')}")
print(f"  {os.path.join(FIGDIR, 'gamma_sweep_phase_plane_tiled.png')}")
print(f"  {os.path.join(FIGDIR, 'gamma_sweep_reward_traces.png')}")
print(f"  {os.path.join(FIGDIR, 'gamma_sweep_selected_strokes_tiled.png')}")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")