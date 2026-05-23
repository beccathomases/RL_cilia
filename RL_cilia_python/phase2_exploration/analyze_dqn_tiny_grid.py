import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================
summary_csv = "dqn_tiny_grid_runs/summary.csv"
figdir = "dqn_tiny_grid_runs/figures"
os.makedirs(figdir, exist_ok=True)


# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(summary_csv)

print("\nLoaded summary:")
print(df.head())

# helpful sorted views
print("\nSorted by best average reward (descending):")
print(
    df.sort_values(["best_avg_reward", "best_cycle_length"], ascending=[False, False])[
        ["run_name", "seed", "lr", "episodes", "best_cycle_length", "best_avg_reward"]
    ].to_string(index=False)
)

print("\nGrouped summary by (lr, episodes):")
grouped = (
    df.groupby(["lr", "episodes"])
      .agg(
          n_runs=("seed", "count"),
          mean_cycle_length=("best_cycle_length", "mean"),
          std_cycle_length=("best_cycle_length", "std"),
          mean_avg_reward=("best_avg_reward", "mean"),
          std_avg_reward=("best_avg_reward", "std"),
          min_avg_reward=("best_avg_reward", "min"),
          max_avg_reward=("best_avg_reward", "max"),
      )
      .reset_index()
)
print(grouped.to_string(index=False))

grouped.to_csv(os.path.join(figdir, "dqn_tiny_grid_grouped_summary.csv"), index=False)


# ============================================================
# PLOT 1: avg reward vs lr, separate panels by episodes
# ============================================================
episodes_list = sorted(df["episodes"].unique())
lrs = sorted(df["lr"].unique())

fig, axes = plt.subplots(1, len(episodes_list), figsize=(6 * len(episodes_list), 5), sharey=True)
if len(episodes_list) == 1:
    axes = [axes]

for ax, ep in zip(axes, episodes_list):
    sub = df[df["episodes"] == ep]

    # raw seed points
    for lr in lrs:
        s = sub[sub["lr"] == lr]
        ax.plot([lr] * len(s), s["best_avg_reward"], "o", alpha=0.7)

    # mean line
    means = [sub[sub["lr"] == lr]["best_avg_reward"].mean() for lr in lrs]
    ax.plot(lrs, means, "o-", linewidth=2)

    ax.set_title(f"Episodes = {ep}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Best cycle avg reward")
    ax.set_xscale("log")
    ax.grid(True)

plt.suptitle("DQN tiny grid: average reward by learning rate")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(figdir, "dqn_tiny_grid_avg_reward_by_lr.png"), dpi=200, bbox_inches="tight")


# ============================================================
# PLOT 2: cycle length vs lr, separate panels by episodes
# ============================================================
fig, axes = plt.subplots(1, len(episodes_list), figsize=(6 * len(episodes_list), 5), sharey=True)
if len(episodes_list) == 1:
    axes = [axes]

for ax, ep in zip(axes, episodes_list):
    sub = df[df["episodes"] == ep]

    for lr in lrs:
        s = sub[sub["lr"] == lr]
        ax.plot([lr] * len(s), s["best_cycle_length"], "o", alpha=0.7)

    means = [sub[sub["lr"] == lr]["best_cycle_length"].mean() for lr in lrs]
    ax.plot(lrs, means, "o-", linewidth=2)

    ax.set_title(f"Episodes = {ep}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Best cycle length")
    ax.set_xscale("log")
    ax.grid(True)

plt.suptitle("DQN tiny grid: cycle length by learning rate")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(figdir, "dqn_tiny_grid_cycle_length_by_lr.png"), dpi=200, bbox_inches="tight")


# ============================================================
# PLOT 3: avg reward vs episodes, separate panels by lr
# ============================================================
fig, axes = plt.subplots(1, len(lrs), figsize=(6 * len(lrs), 5), sharey=True)
if len(lrs) == 1:
    axes = [axes]

for ax, lr in zip(axes, lrs):
    sub = df[df["lr"] == lr]

    for ep in episodes_list:
        s = sub[sub["episodes"] == ep]
        ax.plot([ep] * len(s), s["best_avg_reward"], "o", alpha=0.7)

    means = [sub[sub["episodes"] == ep]["best_avg_reward"].mean() for ep in episodes_list]
    ax.plot(episodes_list, means, "o-", linewidth=2)

    ax.set_title(f"lr = {lr:.0e}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Best cycle avg reward")
    ax.grid(True)

plt.suptitle("DQN tiny grid: average reward by training length")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(figdir, "dqn_tiny_grid_avg_reward_by_episodes.png"), dpi=200, bbox_inches="tight")


# ============================================================
# PLOT 4: cycle length vs episodes, separate panels by lr
# ============================================================
fig, axes = plt.subplots(1, len(lrs), figsize=(6 * len(lrs), 5), sharey=True)
if len(lrs) == 1:
    axes = [axes]

for ax, lr in zip(axes, lrs):
    sub = df[df["lr"] == lr]

    for ep in episodes_list:
        s = sub[sub["episodes"] == ep]
        ax.plot([ep] * len(s), s["best_cycle_length"], "o", alpha=0.7)

    means = [sub[sub["episodes"] == ep]["best_cycle_length"].mean() for ep in episodes_list]
    ax.plot(episodes_list, means, "o-", linewidth=2)

    ax.set_title(f"lr = {lr:.0e}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Best cycle length")
    ax.grid(True)

plt.suptitle("DQN tiny grid: cycle length by training length")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(figdir, "dqn_tiny_grid_cycle_length_by_episodes.png"), dpi=200, bbox_inches="tight")


# ============================================================
# SAVE FULL SORTED TABLES
# ============================================================
df.sort_values(["best_avg_reward", "best_cycle_length"], ascending=[False, False]).to_csv(
    os.path.join(figdir, "dqn_tiny_grid_sorted_by_reward.csv"),
    index=False
)

df.sort_values(["best_cycle_length", "best_avg_reward"], ascending=[False, False]).to_csv(
    os.path.join(figdir, "dqn_tiny_grid_sorted_by_length.csv"),
    index=False
)

print("\nSaved:")
print(" ", os.path.join(figdir, "dqn_tiny_grid_grouped_summary.csv"))
print(" ", os.path.join(figdir, "dqn_tiny_grid_sorted_by_reward.csv"))
print(" ", os.path.join(figdir, "dqn_tiny_grid_sorted_by_length.csv"))
print(" ", os.path.join(figdir, "dqn_tiny_grid_avg_reward_by_lr.png"))
print(" ", os.path.join(figdir, "dqn_tiny_grid_cycle_length_by_lr.png"))
print(" ", os.path.join(figdir, "dqn_tiny_grid_avg_reward_by_episodes.png"))
print(" ", os.path.join(figdir, "dqn_tiny_grid_cycle_length_by_episodes.png"))

plt.show()