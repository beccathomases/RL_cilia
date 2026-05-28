import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cilia_2_ball_env import Cilia2BallEnv


SUMMARY_CSV = "dqn_tiny_grid_runs/summary.csv"
OUTDIR = "dqn_tiny_grid_runs/figures"
OUTFILE = os.path.join(OUTDIR, "dqn_all_runs_phase_tiled.png")

os.makedirs(OUTDIR, exist_ok=True)


def load_result(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def cycle_to_phase(env, cycle_states):
    phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])
    return phis[:, 0], phis[:, 1]


df = pd.read_csv(SUMMARY_CSV)
df = df.sort_values(["episodes", "lr", "seed"]).reset_index(drop=True)

n = len(df)
ncols = 5
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.2 * ncols, 3.8 * nrows),
    sharex=True, sharey=True
)

axes = np.atleast_1d(axes).ravel()

for ax, (_, row) in zip(axes, df.iterrows()):
    result = load_result(row["result_path"])
    env = Cilia2BallEnv(**result["env_settings"])

    cycles = list(result["cycles"])
    if len(cycles) == 0:
        ax.set_title(f"{row['run_name']}\nno cycle found")
        ax.grid(True)
        continue

    best = cycles[0]
    cycle_states = list(best["cycle"])
    phi1, phi2 = cycle_to_phase(env, cycle_states)

    ax.plot(phi1, phi2, "o-", linewidth=1.8, markersize=3.5)
    ax.plot(phi1[0], phi2[0], "o", markersize=7)  # start marker

    ax.set_title(
        f"lr={row['lr']:.0e}, ep={int(row['episodes'])}, seed={int(row['seed'])}\n"
        f"len={int(best['length'])}, avg={float(best['avg_reward']):.6f}",
        fontsize=9
    )
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.grid(True)

# hide unused axes
for ax in axes[len(df):]:
    ax.axis("off")

fig.suptitle("DQN tiny grid: best-cycle phase-plane plots for all runs", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTFILE, dpi=200, bbox_inches="tight")

print("Saved:", OUTFILE)
plt.show()