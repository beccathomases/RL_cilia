import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cilia_3_ball_env import Cilia3BallEnv


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
FILE_TEMPLATE = (
    "tabular_q_cilia_3_ball_clip_penalty_bins11x21x21_"
    "ep10000_steps1500_g0.990_eps0.75_a0.99_seed{seed}.npy"
)

OUTDIR = "figures_tabq_3ball_seed_sweep"
TMAX = 60

os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_npy_dict(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def repeat_to_length(arr, length_needed):
    arr = np.asarray(arr)
    reps = int(np.ceil(length_needed / len(arr)))
    out = np.tile(arr, reps)
    return out[:length_needed]


def state_to_shape_xy(env, state):
    psi = env.state_to_segment_angles(np.array(state, dtype=float))

    p0 = np.array(env.X0, dtype=float)
    pts = [p0.copy()]
    curr = p0.copy()

    for k in range(env.Nballs):
        curr = curr + env.len * np.array(
            [np.sin(psi[k]), 0.0, np.cos(psi[k])], dtype=float
        )
        pts.append(curr.copy())

    pts = np.array(pts)
    x = pts[:, 0]
    z = pts[:, 2]
    return x, z


# ------------------------------------------------------------
# Load runs
# ------------------------------------------------------------
runs = []
summary_rows = []

for seed in SEEDS:
    path = FILE_TEMPLATE.format(seed=seed)
    data = load_npy_dict(path)
    env_settings = data["env_settings"]

    env = Cilia3BallEnv(
        max_steps=env_settings["max_steps"],
        precompute=True,
        boundary_mode=env_settings["boundary_mode"],
        invalid_penalty=env_settings["invalid_penalty"],
        reward_rescale=env_settings["reward_rescale"],
        n_bins=env_settings["n_bins"],
        angle_mins=env_settings["angle_mins"],
        angle_maxs=env_settings["angle_maxs"],
    )

    cycles = list(data["cycles"])
    if len(cycles) == 0:
        raise ValueError(f"No cycles found in {path}")

    best = cycles[0]
    cycle_states = list(best["cycle"])
    cycle_actions = list(best["actions"])
    cycle_rewards = list(best["rewards"])
    cycle_len = int(best["length"])
    avg_reward = float(best["avg_reward"])

    phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])

    runs.append({
        "seed": seed,
        "file": path,
        "env": env,
        "states": cycle_states,
        "actions": cycle_actions,
        "rewards": cycle_rewards,
        "length": cycle_len,
        "avg_reward": avg_reward,
        "phi1": phis[:, 0],
        "phi2": phis[:, 1],
        "phi3": phis[:, 2],
    })

    summary_rows.append({
        "seed": seed,
        "cycle_length": cycle_len,
        "avg_reward": avg_reward,
        "file": path,
    })

df = pd.DataFrame(summary_rows)
df = df.sort_values(["avg_reward", "cycle_length"], ascending=[False, False]).reset_index(drop=True)
summary_csv = os.path.join(OUTDIR, "tabq_3ball_seed_summary.csv")
df.to_csv(summary_csv, index=False)

print("\nSummary:")
print(df.to_string(index=False))
print(f"\nSaved summary CSV: {summary_csv}")


# ------------------------------------------------------------
# Figure 1: tiled repeated angle traces
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.ravel()
t = np.arange(TMAX + 1)

for ax, run in zip(axes, runs):
    phi1_rep = repeat_to_length(run["phi1"], TMAX + 1)
    phi2_rep = repeat_to_length(run["phi2"], TMAX + 1)
    phi3_rep = repeat_to_length(run["phi3"], TMAX + 1)

    ax.plot(t, phi1_rep, "-", linewidth=2, label=r"$\phi_1$")
    ax.plot(t, phi2_rep, "--", linewidth=2, label=r"$\phi_2$")
    ax.plot(t, phi3_rep, "-.", linewidth=2, label=r"$\phi_3$")

    ax.set_title(
        f"seed={run['seed']}\nlen={run['length']}, avg={run['avg_reward']:.6f}",
        fontsize=10
    )
    ax.set_xlabel("t")
    ax.set_ylabel("angle")
    ax.grid(True)

for ax in axes[len(runs):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle(r"Tabular Q, 3-ball: repeated angle traces to $t=60$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

angles_file = os.path.join(OUTDIR, "tabq_3ball_seed_angle_traces_tiled.png")
fig.savefig(angles_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 2: tiled physical stroke overlays
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.ravel()
cmap = plt.get_cmap("viridis")

for ax, run in zip(axes, runs):
    n = len(run["states"])
    for j, state in enumerate(run["states"]):
        x, z = state_to_shape_xy(run["env"], state)
        color = cmap(j / max(1, n - 1))
        ax.plot(x, z, "o-", linewidth=2, markersize=4, color=color, alpha=0.9)

    x0, z0 = state_to_shape_xy(run["env"], run["states"][0])
    ax.plot(x0, z0, "o-", linewidth=2.8, markersize=5, color="red")

    xL, zL = state_to_shape_xy(run["env"], run["states"][-1])
    ax.plot(xL, zL, "o-", linewidth=2.8, markersize=5, color="black")

    ax.set_title(
        f"seed={run['seed']}\nlen={run['length']}, avg={run['avg_reward']:.6f}",
        fontsize=10
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.grid(True)

for ax in axes[len(runs):]:
    ax.axis("off")

fig.suptitle("Tabular Q, 3-ball: physical stroke overlays by seed", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

stroke_file = os.path.join(OUTDIR, "tabq_3ball_seed_strokes_tiled.png")
fig.savefig(stroke_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 3: tiled pairwise phase projections
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.ravel()

for ax, run in zip(axes, runs):
    ax.plot(run["phi1"], run["phi2"], "o-", linewidth=2, markersize=4, label=r"$(\phi_1,\phi_2)$")
    ax.plot(run["phi2"], run["phi3"], "s--", linewidth=1.8, markersize=3.5, label=r"$(\phi_2,\phi_3)$")
    ax.plot(run["phi1"], run["phi3"], "d-.", linewidth=1.8, markersize=3.5, label=r"$(\phi_1,\phi_3)$")

    ax.set_title(
        f"seed={run['seed']}\nlen={run['length']}, avg={run['avg_reward']:.6f}",
        fontsize=10
    )
    ax.set_xlabel("first angle")
    ax.set_ylabel("second angle")
    ax.grid(True)

for ax in axes[len(runs):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle("Tabular Q, 3-ball: pairwise phase projections by seed", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

phase_file = os.path.join(OUTDIR, "tabq_3ball_seed_phase_tiled.png")
fig.savefig(phase_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 4: tiled flux reward traces along cycle
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False, sharey=True)
axes = axes.ravel()

for ax, run in zip(axes, runs):
    rewards = np.array(run["rewards"], dtype=float)
    t_reward = np.arange(1, len(rewards) + 1)

    ax.plot(t_reward, rewards, "o-", linewidth=2, markersize=4)
    ax.axhline(0.0, linewidth=1)

    ax.set_title(
        f"seed={run['seed']}\nlen={run['length']}, avg={run['avg_reward']:.6f}",
        fontsize=10
    )
    ax.set_xlabel("step in cycle")
    ax.set_ylabel("flux reward")
    ax.grid(True)

for ax in axes[len(runs):]:
    ax.axis("off")

fig.suptitle("Tabular Q, 3-ball: flux reward traces along best cycle by seed", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

reward_trace_file = os.path.join(OUTDIR, "tabq_3ball_seed_reward_traces_tiled.png")
fig.savefig(reward_trace_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 5: tiled repeated flux reward traces
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.ravel()
t = np.arange(TMAX + 1)

for ax, run in zip(axes, runs):
    rewards = np.array(run["rewards"], dtype=float)
    rewards_rep = repeat_to_length(rewards, TMAX + 1)

    ax.plot(t, rewards_rep, "-", linewidth=2)
    ax.axhline(0.0, linewidth=1)

    ax.set_title(
        f"seed={run['seed']}\nlen={run['length']}, avg={run['avg_reward']:.6f}",
        fontsize=10
    )
    ax.set_xlabel("t")
    ax.set_ylabel("flux reward")
    ax.grid(True)

for ax in axes[len(runs):]:
    ax.axis("off")

fig.suptitle(r"Tabular Q, 3-ball: repeated flux reward traces to $t=60$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

reward_repeat_file = os.path.join(OUTDIR, "tabq_3ball_seed_reward_traces_repeated_tiled.png")
fig.savefig(reward_repeat_file, dpi=200, bbox_inches="tight")


print("\nSaved figures:")
print(" ", angles_file)
print(" ", stroke_file)
print(" ", phase_file)
print(" ", reward_trace_file)
print(" ", reward_repeat_file)

plt.show()