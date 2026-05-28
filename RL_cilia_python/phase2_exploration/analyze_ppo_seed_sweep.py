import os
import pandas as pd
import matplotlib.pyplot as plt
SUMMARY_CSV = "ppo_seed_sweep_runs/summary_reevaluated.csv"
#SUMMARY_CSV = "ppo_seed_sweep_runs/summary.csv"
OUTDIR = "ppo_seed_sweep_runs/figures"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(SUMMARY_CSV)

print("\nLoaded summary:")
print(df)

print("\nSorted by average reward:")
print(
    df.sort_values(["best_avg_reward", "best_cycle_length"], ascending=[False, False])[
        ["run_name", "seed", "total_timesteps", "best_cycle_length", "best_avg_reward"]
    ].to_string(index=False)
)

grouped = (
    df.groupby("total_timesteps")
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

print("\nGrouped by total_timesteps:")
print(grouped.to_string(index=False))

grouped.to_csv(os.path.join(OUTDIR, "ppo_grouped_summary.csv"), index=False)

# avg reward by timesteps
plt.figure(figsize=(6, 4.5))
for steps in sorted(df["total_timesteps"].unique()):
    sub = df[df["total_timesteps"] == steps]
    plt.plot([steps] * len(sub), sub["best_avg_reward"], "o")
means = grouped["mean_avg_reward"].values
steps = grouped["total_timesteps"].values
plt.plot(steps, means, "o-", linewidth=2)
plt.xlabel("Total timesteps")
plt.ylabel("Best cycle avg reward")
plt.title("PPO seed sweep: average reward")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ppo_avg_reward_by_timesteps.png"), dpi=200, bbox_inches="tight")

# cycle length by timesteps
plt.figure(figsize=(6, 4.5))
for steps in sorted(df["total_timesteps"].unique()):
    sub = df[df["total_timesteps"] == steps]
    plt.plot([steps] * len(sub), sub["best_cycle_length"], "o")
means = grouped["mean_cycle_length"].values
steps = grouped["total_timesteps"].values
plt.plot(steps, means, "o-", linewidth=2)
plt.xlabel("Total timesteps")
plt.ylabel("Best cycle length")
plt.title("PPO seed sweep: cycle length")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ppo_cycle_length_by_timesteps.png"), dpi=200, bbox_inches="tight")

df.sort_values(["best_avg_reward", "best_cycle_length"], ascending=[False, False]).to_csv(
    os.path.join(OUTDIR, "ppo_sorted_by_reward.csv"), index=False
)

print("\nSaved:")
print(" ", os.path.join(OUTDIR, "ppo_grouped_summary.csv"))
print(" ", os.path.join(OUTDIR, "ppo_sorted_by_reward.csv"))
print(" ", os.path.join(OUTDIR, "ppo_avg_reward_by_timesteps.png"))
print(" ", os.path.join(OUTDIR, "ppo_cycle_length_by_timesteps.png"))

plt.show()