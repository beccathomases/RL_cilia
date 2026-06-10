#!/usr/bin/env python3
"""
make_phase5_n5_tables.py
========================

Create compact Phase 5 N=5 summary tables from the PPO sweep.

Input:
    results/ppo_sweeps_n5/summary_all.csv

Outputs:
    results/ppo_sweeps_n5/phase5_n5_per_seed_table.csv
    results/ppo_sweeps_n5/phase5_n5_group_summary.csv
    results/ppo_sweeps_n5/phase5_n5_per_seed_table.tex
    results/ppo_sweeps_n5/phase5_n5_group_summary.tex
"""

from pathlib import Path
import pandas as pd


INFILE = Path("results/ppo_sweeps_n5/summary_all.csv")
OUTDIR = Path("results/ppo_sweeps_n5")
CASE = "pi20"


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of these columns: {candidates}")


def fmt_range(xmin, xmax, ndigits=3):
    return f"{xmin:.{ndigits}f}--{xmax:.{ndigits}f}"


def main():
    df = pd.read_csv(INFILE)

    n_col = pick_col(df, ["nballs", "N"])
    seed_col = pick_col(df, ["seed"])
    dtheta_col = pick_col(df, ["dtheta_label", "dtheta"])
    timesteps_col = pick_col(df, ["timesteps"])
    L_col = pick_col(df, ["cycle_length", "PPO_L"])
    mean_col = pick_col(df, ["cycle_mean_reward", "PPO_mean_reward", "ppo_avg_reward"])
    total_col = pick_col(df, ["cycle_total_reward", "PPO_cycle_reward"])
    area_col = pick_col(df, ["tip_abs_area", "tip_area"])
    path_col = pick_col(df, ["tip_path_length"])
    rho_col = pick_col(df, ["reward_per_abs_tip_area", "cycle_reward_per_tip_area", "rho"])
    wrong_col = pick_col(df, ["wrong_orientation_flag", "wrong_orientation", "wrong_orient"])

    # N=5, selected case, largest training horizon only
    df = df[(df[n_col] == 5) & (df[dtheta_col] == CASE)].copy()
    max_timesteps = df[timesteps_col].max()
    df = df[df[timesteps_col] == max_timesteps].copy()
    df = df.sort_values(seed_col).reset_index(drop=True)

    # Per-seed table
    tab = pd.DataFrame({
        "seed": df[seed_col].astype(int),
        "L": df[L_col].astype(int),
        "mean reward": df[mean_col],
        "cycle reward": df[total_col],
        "tip area": df[area_col],
        "tip path": df[path_col],
        "rho": df[rho_col],
        "wrong orient": df[wrong_col].astype(bool),
    })

    # Rounded display version
    tab_display = tab.copy()
    for c in ["mean reward", "cycle reward", "tip area", "tip path", "rho"]:
        tab_display[c] = tab_display[c].map(lambda x: f"{x:.3f}")
    tab_display["wrong orient"] = tab_display["wrong orient"].map(lambda x: "yes" if x else "no")

    per_seed_csv = OUTDIR / "phase5_n5_per_seed_table.csv"
    per_seed_tex = OUTDIR / "phase5_n5_per_seed_table.tex"

    tab.to_csv(per_seed_csv, index=False)
    tab_display.to_latex(
        per_seed_tex,
        index=False,
        escape=False,
        column_format="rrrrrrrl",
        caption=(
            r"Per-seed diagnostics for the $N=5$ PPO sweep at "
            r"$\Delta\theta=\pi/20$ using $2\times 10^6$ training steps."
        ),
        label="tab:phase5-n5-per-seed",
    )

    # Group summary
    group = pd.DataFrame([{
        "N": 5,
        "dtheta": CASE,
        "n seeds": len(df),
        "timesteps": int(max_timesteps),
        "mean reward median": df[mean_col].median(),
        "mean reward range": fmt_range(df[mean_col].min(), df[mean_col].max(), 3),
        "L range": f"{int(df[L_col].min())}--{int(df[L_col].max())}",
        "tip area median": df[area_col].median(),
        "tip area range": fmt_range(df[area_col].min(), df[area_col].max(), 3),
        "rho median": df[rho_col].median(),
        "rho range": fmt_range(df[rho_col].min(), df[rho_col].max(), 2),
        "wrong orientation": int(df[wrong_col].astype(bool).sum()),
    }])

    group_display = group.copy()
    for c in ["mean reward median", "tip area median", "rho median"]:
        group_display[c] = group_display[c].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    group_csv = OUTDIR / "phase5_n5_group_summary.csv"
    group_tex = OUTDIR / "phase5_n5_group_summary.tex"

    group.to_csv(group_csv, index=False)
    group_display.to_latex(
        group_tex,
        index=False,
        escape=False,
        caption=(
            r"Aggregate diagnostics for the first $N=5$ PPO sweep. "
            r"All 10 seeds produce positive, nondegenerate strokes."
        ),
        label="tab:phase5-n5-summary",
    )

    print(f"wrote {per_seed_csv}")
    print(f"wrote {per_seed_tex}")
    print(f"wrote {group_csv}")
    print(f"wrote {group_tex}")

    print("\nPer-seed table:")
    print(tab_display.to_string(index=False))

    print("\nGroup summary:")
    print(group_display.to_string(index=False))


if __name__ == "__main__":
    main()