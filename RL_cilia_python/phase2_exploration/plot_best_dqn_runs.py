import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cilia_2_ball_env import Cilia2BallEnv


SUMMARY_CSV = "dqn_tiny_grid_runs/summary.csv"
OUTDIR = "dqn_tiny_grid_runs/figures"
os.makedirs(OUTDIR, exist_ok=True)

TOP_K = 3


def load_result(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def state_to_shape_xy(env, state):
    psi = env.state_to_segment_angles(np.array(state, dtype=float))

    p0 = np.array(env.X0, dtype=float)
    p1 = p0 + env.len * np.array([np.sin(psi[0]), 0.0, np.cos(psi[0])], dtype=float)
    p2 = p1 + env.len * np.array([np.sin(psi[1]), 0.0, np.cos(psi[1])], dtype=float)

    x = np.array([p0[0], p1[0], p2[0]])
    z = np.array([p0[2], p1[2], p2[2]])
    return x, z


def cycle_to_phase(env, cycle_states):
    phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])
    return phis[:, 0], phis[:, 1]


df = pd.read_csv(SUMMARY_CSV)
df = df.sort_values(["best_avg_reward", "best_cycle_length"], ascending=[False, False]).reset_index(drop=True)
top = df.iloc[:TOP_K].copy()

runs = []
for _, row in top.iterrows():
    result = load_result(row["result_path"])
    env = Cilia2BallEnv(**result["env_settings"])
    cycles = list(result["cycles"])
    best = cycles[0]

    runs.append({
        "run_name": row["run_name"],
        "lr": row["lr"],
        "episodes": int(row["episodes"]),
        "seed": int(row["seed"]),
        "avg_reward": float(best["avg_reward"]),
        "length": int(best["length"]),
        "cycle": list(best["cycle"]),
        "rewards": list(best["rewards"]),
        "env": env,
    })

# ------------------------------------------------------------
# phase-plane tiled
# ------------------------------------------------------------
fig, axes = plt.subplots(1, TOP_K, figsize=(5 * TOP_K, 4.5), sharex=True, sharey=True)
if TOP_K == 1:
    axes = [axes]

for ax, run in zip(axes, runs):
    phi1, phi2 = cycle_to_phase(run["env"], run["cycle"])
    ax.plot(phi1, phi2, "o-", linewidth=2, markersize=4)
    ax.plot(phi1[0], phi2[0], "o", markersize=8)
    ax.set_title(
        f"{run['run_name']}\nlen={run['length']}, avg={run['avg_reward']:.6f}"
    )
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.grid(True)

fig.suptitle("Top 3 DQN runs: phase-plane cycles", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUTDIR, "dqn_tiny_grid_top3_phase_plane.png"), dpi=200, bbox_inches="tight")

# ------------------------------------------------------------
# stroke overlays tiled
# ------------------------------------------------------------
fig, axes = plt.subplots(1, TOP_K, figsize=(5 * TOP_K, 4.5))
if TOP_K == 1:
    axes = [axes]

cmap = plt.get_cmap("viridis")

for ax, run in zip(axes, runs):
    shape_list = [state_to_shape_xy(run["env"], s) for s in run["cycle"]]
    n = len(shape_list)

    for j, (x, z) in enumerate(shape_list):
        color = cmap(j / max(1, n - 1))
        ax.plot(x, z, "o-", color=color, linewidth=2, markersize=4, alpha=0.9)

    x0, z0 = shape_list[0]
    ax.plot(x0, z0, "o-", color="red", linewidth=2.5, markersize=5)

    xL, zL = shape_list[-1]
    ax.plot(xL, zL, "o-", color="black", linewidth=2.5, markersize=5)

    ax.set_title(
        f"{run['run_name']}\nlen={run['length']}, avg={run['avg_reward']:.6f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.grid(True)

fig.suptitle("Top 3 DQN runs: stroke reconstructions", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUTDIR, "dqn_tiny_grid_top3_strokes.png"), dpi=200, bbox_inches="tight")

# ------------------------------------------------------------
# reward traces
# ------------------------------------------------------------
fig = plt.figure(figsize=(7, 4.5))
for run in runs:
    rewards = np.array(run["rewards"], dtype=float)
    plt.plot(
        np.arange(1, len(rewards) + 1),
        rewards,
        "o-",
        linewidth=2,
        markersize=4,
        label=f"{run['run_name']} (len={run['length']})"
    )

plt.xlabel("Step in cycle")
plt.ylabel("Reward")
plt.title("Top 3 DQN runs: reward traces")
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "dqn_tiny_grid_top3_reward_traces.png"), dpi=200, bbox_inches="tight")

print("Saved:")
print(" ", os.path.join(OUTDIR, "dqn_tiny_grid_top3_phase_plane.png"))
print(" ", os.path.join(OUTDIR, "dqn_tiny_grid_top3_strokes.png"))
print(" ", os.path.join(OUTDIR, "dqn_tiny_grid_top3_reward_traces.png"))

plt.show()