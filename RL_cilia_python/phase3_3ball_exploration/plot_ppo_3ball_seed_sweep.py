import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cilia_3_ball_env import Cilia3BallEnv


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
TIMESTEPS_LIST = [300000, 1000000]
SEEDS = [0, 1, 2]

FILE_TEMPLATE = "ppo_3ball_runs/results/ppo_3ball_steps{steps}_seed{seed}.npy"

OUTDIR = "figures_ppo_3ball_seed_sweep"
TMAX = 100

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

for steps in TIMESTEPS_LIST:
    for seed in SEEDS:
        path = FILE_TEMPLATE.format(steps=steps, seed=seed)
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
            "steps": steps,
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
            "steps": steps,
            "seed": seed,
            "cycle_length": cycle_len,
            "avg_reward": avg_reward,
            "file": path,
        })

df = pd.DataFrame(summary_rows)
df = df.sort_values(["steps", "seed"]).reset_index(drop=True)
summary_csv = os.path.join(OUTDIR, "ppo_3ball_summary.csv")
df.to_csv(summary_csv, index=False)

print("\nSummary:")
print(df.to_string(index=False))
print(f"\nSaved summary CSV: {summary_csv}")


# ------------------------------------------------------------
# Common title helper
# ------------------------------------------------------------
def panel_title(run):
    return (
        f"steps={run['steps']}, seed={run['seed']}\n"
        f"len={run['length']}, avg={run['avg_reward']:.6f}"
    )


# ------------------------------------------------------------
# Figure 1: tiled repeated angle traces
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True, sharey=True)
axes = axes.ravel()
t = np.arange(TMAX + 1)

for ax, run in zip(axes, runs):
    phi1_rep = repeat_to_length(run["phi1"], TMAX + 1)
    phi2_rep = repeat_to_length(run["phi2"], TMAX + 1)
    phi3_rep = repeat_to_length(run["phi3"], TMAX + 1)

    ax.plot(t, phi1_rep, "-", linewidth=2, label=r"$\phi_1$")
    ax.plot(t, phi2_rep, "--", linewidth=2, label=r"$\phi_2$")
    ax.plot(t, phi3_rep, "-.", linewidth=2, label=r"$\phi_3$")

    ax.set_title(panel_title(run), fontsize=10)
    ax.set_xlabel("t")
    ax.set_ylabel("angle")
    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle(r"PPO 3-ball: repeated angle traces to $t=100$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

angles_file = os.path.join(OUTDIR, "ppo_3ball_angle_traces_tiled.png")
fig.savefig(angles_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 2: tiled physical stroke overlays
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
axes = axes.ravel()
cmap = plt.get_cmap("viridis")

for ax, run in zip(axes, runs):
    n = len(run["states"])
    for j, state in enumerate(run["states"]):
        x, z = state_to_shape_xy(run["env"], state)
        color = cmap(j / max(1, n - 1))
        ax.plot(x, z, "o-", linewidth=1.6, markersize=3.5, color=color, alpha=0.9)

    x0, z0 = state_to_shape_xy(run["env"], run["states"][0])
    ax.plot(x0, z0, "o-", linewidth=2.5, markersize=5, color="red")

    xL, zL = state_to_shape_xy(run["env"], run["states"][-1])
    ax.plot(xL, zL, "o-", linewidth=2.5, markersize=5, color="black")

    ax.set_title(panel_title(run), fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.grid(True)

fig.suptitle("PPO 3-ball: physical stroke overlays", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

stroke_file = os.path.join(OUTDIR, "ppo_3ball_stroke_overlays_tiled.png")
fig.savefig(stroke_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 3: tiled pairwise phase projections
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
axes = axes.ravel()

for ax, run in zip(axes, runs):
    ax.plot(run["phi1"], run["phi2"], "o-", linewidth=2, markersize=3.5, label=r"$(\phi_1,\phi_2)$")
    ax.plot(run["phi2"], run["phi3"], "s--", linewidth=1.8, markersize=3.0, label=r"$(\phi_2,\phi_3)$")
    ax.plot(run["phi1"], run["phi3"], "d-.", linewidth=1.8, markersize=3.0, label=r"$(\phi_1,\phi_3)$")

    ax.set_title(panel_title(run), fontsize=10)
    ax.set_xlabel("first angle")
    ax.set_ylabel("second angle")
    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.suptitle("PPO 3-ball: pairwise phase projections", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

phase_file = os.path.join(OUTDIR, "ppo_3ball_phase_projections_tiled.png")
fig.savefig(phase_file, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------
# Figure 4: tiled repeated flux reward traces
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True, sharey=True)
axes = axes.ravel()
t = np.arange(TMAX + 1)

for ax, run in zip(axes, runs):
    rewards = np.array(run["rewards"], dtype=float)
    rewards_rep = repeat_to_length(rewards, TMAX + 1)

    ax.plot(t, rewards_rep, linewidth=2)
    ax.axhline(0.0, linewidth=1)

    ax.set_title(panel_title(run), fontsize=10)
    ax.set_xlabel("t")
    ax.set_ylabel("flux reward")
    ax.grid(True)

fig.suptitle(r"PPO 3-ball: repeated flux reward traces to $t=100$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

reward_file = os.path.join(OUTDIR, "ppo_3ball_reward_traces_repeated_tiled.png")
fig.savefig(reward_file, dpi=200, bbox_inches="tight")


print("\nSaved figures:")
print(" ", angles_file)
print(" ", stroke_file)
print(" ", phase_file)
print(" ", reward_file)

plt.show()