#!/usr/bin/env python3
"""
plot_phase5_n5_summary.py
=========================

Make a compact Phase 5 summary figure for the N=5 PPO sweep.

Input:
    results/ppo_sweeps_n5/summary_all.csv

Output:
    figures/ppo_sweeps_n5/phase5_n5_summary/phase5_n5_summary_pi20.png
    figures/ppo_sweeps_n5/phase5_n5_summary/phase5_n5_summary_pi20.pdf

The figure shows, by seed:
    1) cycle mean reward
    2) cycle length
    3) tip absolute area
    4) reward per absolute tip area (rho)

Default behavior:
    - use dtheta_label = "pi20"
    - use only the largest timesteps present in the csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INFILE = Path("results/ppo_sweeps_n5/summary_all.csv")
OUTDIR = Path("figures/ppo_sweeps_n5/phase5_n5_summary")

CASE = "pi30"   # change to "pi30" later if needed


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of these columns: {candidates}")


def main():
    if not INFILE.exists():
        raise FileNotFoundError(f"Could not find {INFILE}")

    df = pd.read_csv(INFILE)

    # Flexible column names
    n_col = pick_col(df, ["nballs", "N"])
    seed_col = pick_col(df, ["seed"])
    dtheta_col = pick_col(df, ["dtheta_label", "dtheta"])
    timesteps_col = pick_col(df, ["timesteps"])
    L_col = pick_col(df, ["cycle_length", "PPO_L"])
    mean_col = pick_col(df, ["cycle_mean_reward", "PPO_mean_reward", "ppo_avg_reward"])
    area_col = pick_col(df, ["tip_abs_area", "tip_area"])
    rho_col = pick_col(df, ["reward_per_abs_tip_area", "cycle_reward_per_tip_area", "rho"])
    wrong_col = pick_col(df, ["wrong_orientation_flag", "wrong_orientation", "wrong_orient"])

    # Keep N=5, chosen case, and only the largest training horizon in the file
    df = df[df[n_col] == 5].copy()
    df = df[df[dtheta_col] == CASE].copy()

    if df.empty:
        raise ValueError(f"No rows found for N=5 and case={CASE}")

    max_timesteps = df[timesteps_col].max()
    df = df[df[timesteps_col] == max_timesteps].copy()

    df = df.sort_values(seed_col).reset_index(drop=True)

    # Basic summary stats
    n_runs = len(df)
    reward_med = df[mean_col].median()
    reward_min = df[mean_col].min()
    reward_max = df[mean_col].max()

    L_med = df[L_col].median()
    L_min = df[L_col].min()
    L_max = df[L_col].max()

    area_med = df[area_col].median()
    area_min = df[area_col].min()
    area_max = df[area_col].max()

    rho_med = df[rho_col].median()
    rho_min = df[rho_col].min()
    rho_max = df[rho_col].max()

    n_wrong = int(df[wrong_col].sum()) if df[wrong_col].dtype != object else int(df[wrong_col].astype(bool).sum())

    seeds = df[seed_col].to_numpy()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axs = axs.ravel()

    # Panel 1: mean reward
    axs[0].plot(seeds, df[mean_col], marker="o", linewidth=1.8)
    axs[0].axhline(reward_med, linestyle="--", linewidth=1)
    axs[0].set_title("cycle mean reward")
    axs[0].set_xlabel("seed")
    axs[0].set_ylabel("mean reward")
    axs[0].grid(True, alpha=0.3)

    # Panel 2: cycle length
    axs[1].plot(seeds, df[L_col], marker="o", linewidth=1.8)
    axs[1].axhline(L_med, linestyle="--", linewidth=1)
    axs[1].set_title("cycle length")
    axs[1].set_xlabel("seed")
    axs[1].set_ylabel("L")
    axs[1].grid(True, alpha=0.3)

    # Panel 3: tip area
    axs[2].plot(seeds, df[area_col], marker="o", linewidth=1.8)
    axs[2].axhline(area_med, linestyle="--", linewidth=1)
    axs[2].set_title("tip absolute area")
    axs[2].set_xlabel("seed")
    axs[2].set_ylabel("area")
    axs[2].grid(True, alpha=0.3)

    # Panel 4: rho
    axs[3].plot(seeds, df[rho_col], marker="o", linewidth=1.8)
    axs[3].axhline(rho_med, linestyle="--", linewidth=1)
    axs[3].set_title(r"reward per area, $\rho$")
    axs[3].set_xlabel("seed")
    axs[3].set_ylabel(r"$\rho$")
    axs[3].grid(True, alpha=0.3)

    # Shared title
    fig.suptitle(
        f"Phase 5 summary: N=5 PPO at Δθ={CASE.replace('pi', 'π/') if CASE.startswith('pi') else CASE}, "
        f"{int(max_timesteps):,} training steps",
        fontsize=16
    )

    # Summary text box
    summary_text = (
        f"runs = {n_runs}\n"
        f"wrong-orientation = {n_wrong}\n\n"
        f"mean reward: {reward_min:.3f} to {reward_max:.3f}\n"
        f"cycle length: {int(L_min)} to {int(L_max)}\n"
        f"tip area: {area_min:.3f} to {area_max:.3f}\n"
        f"rho: {rho_min:.2f} to {rho_max:.2f}"
    )

    fig.text(
        0.77, 0.28, summary_text,
        ha="left", va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = OUTDIR / f"phase5_n5_summary_{CASE}.png"
    pdf_path = OUTDIR / f"phase5_n5_summary_{CASE}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")

    print("\nQuick summary:")
    print(df[[seed_col, L_col, mean_col, area_col, rho_col, wrong_col]].to_string(index=False))


if __name__ == "__main__":
    main()