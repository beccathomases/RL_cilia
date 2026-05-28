import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cilia_2_ball_env import Cilia2BallEnv


SUMMARY_CSV = "dqn_tiny_grid_runs/summary.csv"
OUTDIR = "dqn_tiny_grid_runs/figures"
OUTFILE = os.path.join(OUTDIR, "dqn_all_runs_angles_tiled_t100.png")

TMAX = 100

os.makedirs(OUTDIR, exist_ok=True)


def load_result(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def cycle_to_angles(env, cycle_states):
    phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])
    return phis[:, 0], phis[:, 1]


def repeat_to_length(arr, length_needed):
    arr = np.asarray(arr)
    reps = int(np.ceil(length_needed / len(arr)))
    out = np.tile(arr, reps)
    return out[:length_needed]


df = pd.read_csv(SUMMARY_CSV)
df = df.sort_values(["episodes", "lr", "seed"]).reset_index(drop=True)

n = len(df)
ncols = 5
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.4 * ncols, 3.6 * nrows),
    sharex=True, sharey=True
)
axes = np.atleast_1d(axes).ravel()

t = np.arange(TMAX + 1)

for ax, (_, row) in zip(axes, df.iterrows()):
    result = load_result(row["result_path"])
    env = Cilia2BallEnv(**result["env_settings"])

    cycles = list(result["cycles"])
    if len(cycles) == 0:
        ax.set_title(f"{row['run_name']}\nno cycle found", fontsize=9)
        ax.grid(True)
        continue

    best = cycles[0]
    cycle_states = list(best["cycle"])
    phi1, phi2 = cycle_to_angles(env, cycle_states)

    # repeat periodically to common horizon
    phi1_rep = repeat_to_length(phi1, TMAX + 1)
    phi2_rep = repeat_to_length(phi2, TMAX + 1)

    ax.plot(t, phi1_rep, "-", linewidth=1.8, label=r"$\phi_1$")
    ax.plot(t, phi2_rep, "--", linewidth=1.6, label=r"$\phi_2$")

    ax.set_title(
        f"lr={row['lr']:.0e}, ep={int(row['episodes'])}, seed={int(row['seed'])}\n"
        f"len={int(best['length'])}, avg={float(best['avg_reward']):.6f}",
        fontsize=8.5
    )
    ax.set_xlabel("t")
    ax.set_ylabel("angle")
    ax.grid(True)

# hide unused axes
for ax in axes[len(df):]:
    ax.axis("off")

# single legend
handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

fig.suptitle(r"DQN tiny grid: periodic $\phi_1,\phi_2$ traces repeated to $t=100$", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTFILE, dpi=200, bbox_inches="tight")

print("Saved:", OUTFILE)
plt.show()