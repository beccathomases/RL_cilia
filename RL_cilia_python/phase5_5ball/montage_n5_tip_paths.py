#!/usr/bin/env python3
"""
montage_n5_tip_paths.py
=======================

Build a 2x5 montage of N=5 tip-path figures, one panel per seed.

Input:
    results/ppo_sweeps_n5/summary_all.csv
    figures/ppo_sweeps_n5/N5_dtheta_<case>/seed_XXX/<tip-path-image>

Output:
    figures/ppo_sweeps_n5/montages/n5_tip_path_montage_<case>.png
    figures/ppo_sweeps_n5/montages/n5_tip_path_montage_<case>.pdf

Default:
    - case = pi20
    - use only rows with the largest timesteps found for that case
"""

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ----------------------------
# settings
# ----------------------------
CASE = "pi20"   # change to "pi30" later if needed

SUMMARY_CSV = Path("results/ppo_sweeps_n5/summary_all.csv")
FIG_BASE = Path("figures/ppo_sweeps_n5")
OUTDIR = FIG_BASE / "montages"


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of these columns: {candidates}")


def find_tip_image(seed_dir: Path):
    """
    Look for a plausible tip-path image filename inside the seed directory.
    Adjust/add names here if your visualize script uses a different filename.
    """
    candidates = [
        "tip_path.png",
        "tip_path_cycle0.png",
        "tip_path_cycle_0.png",
        "tip_path_result.png",
        "tip_path_onecycle.png",
    ]
    for name in candidates:
        p = seed_dir / name
        if p.exists():
            return p

    # fallback: anything with 'tip' in the name
    matches = sorted(seed_dir.glob("*tip*.png"))
    if matches:
        return matches[0]

    return None


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Could not find {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    n_col = pick_col(df, ["nballs", "N"])
    seed_col = pick_col(df, ["seed"])
    dtheta_col = pick_col(df, ["dtheta_label", "dtheta"])
    timesteps_col = pick_col(df, ["timesteps"])
    L_col = pick_col(df, ["cycle_length", "PPO_L"])
    mean_col = pick_col(df, ["cycle_mean_reward", "PPO_mean_reward", "ppo_avg_reward"])
    area_col = pick_col(df, ["tip_abs_area", "tip_area"])
    rho_col = pick_col(df, ["reward_per_abs_tip_area", "cycle_reward_per_tip_area", "rho"])

    # Filter to N=5 and chosen case
    df = df[df[n_col] == 5].copy()
    df = df[df[dtheta_col] == CASE].copy()

    if df.empty:
        raise ValueError(f"No rows found for N=5 and dtheta_label={CASE}")

    # Use only largest timesteps present
    max_timesteps = df[timesteps_col].max()
    df = df[df[timesteps_col] == max_timesteps].copy()

    df = df.sort_values(seed_col).reset_index(drop=True)

    # where the seed figure folders live
    case_dir = FIG_BASE / f"N5_dtheta_{CASE}"

    OUTDIR.mkdir(parents=True, exist_ok=True)

    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 7.5))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.iterrows()):
        seed = int(row[seed_col])
        L = int(row[L_col])
        mean_reward = float(row[mean_col])
        area = float(row[area_col])
        rho = float(row[rho_col])

        seed_dir = case_dir / f"seed_{seed:03d}"
        img_path = find_tip_image(seed_dir)

        if img_path is None:
            ax.text(
                0.5, 0.5,
                f"missing image\nseed_{seed:03d}",
                ha="center", va="center", fontsize=11
            )
            ax.set_axis_off()
            continue

        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            f"seed {seed}\nL={L}, mean={mean_reward:.3f}, area={area:.3f}, rho={rho:.2f}",
            fontsize=10
        )

    # blank any extra panels if fewer than 10 rows
    for k in range(len(df), len(axes)):
        axes[k].set_axis_off()

    dtheta_title = "pi/20" if CASE == "pi20" else ("pi/30" if CASE == "pi30" else CASE)

    fig.suptitle(
        f"N=5 PPO tip-path montage, dtheta={dtheta_title}, "
        f"{int(max_timesteps):,} training steps",
        fontsize=18
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = OUTDIR / f"n5_tip_path_montage_{CASE}.png"
    pdf_path = OUTDIR / f"n5_tip_path_montage_{CASE}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()