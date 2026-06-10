from pathlib import Path
import pandas as pd
import numpy as np

root = Path("results/ppo_sweeps_phase6_n12_t6e6_radscale0p4")
outdir = root / "tables"
outdir.mkdir(exist_ok=True)

rename = {
    "cycle_mean_reward": "mean_reward",
    "cycle_total_reward": "cycle_reward",
    "tip_abs_area": "tip_area",
    "reward_per_abs_tip_area": "rho_reward_per_area",
    "wrong_orientation_flag": "wrong_orientation",
}

case_denoms = {"pi20": 20, "pi30": 30, "pi40": 40}

all_rows = []
for case, denom in case_denoms.items():
    path = root / f"N12_dtheta_{case}_factored" / "summary.csv"
    df = pd.read_csv(path).rename(columns=rename)
    df["case"] = case
    df["dtheta"] = np.pi / denom
    df["cycle_phase_length"] = df["cycle_length"] * df["dtheta"]
    df["cycle_phase_over_pi2"] = df["cycle_phase_length"] / (np.pi / 2)
    df["mean_reward_over_dtheta"] = df["mean_reward"] / df["dtheta"]
    all_rows.append(df)

all_df = pd.concat(all_rows, ignore_index=True)

compact_cols = [
    "case",
    "seed",
    "cycle_length",
    "cycle_phase_length",
    "cycle_phase_over_pi2",
    "mean_reward",
    "mean_reward_over_dtheta",
    "cycle_reward",
    "tip_area",
    "tip_signed_area",
    "tip_path_length",
    "rho_reward_per_area",
    "noop_fraction",
    "wrong_orientation",
    "result_path",
    "model_path",
]
compact_cols = [c for c in compact_cols if c in all_df.columns]
compact = all_df[compact_cols].copy()

compact.to_csv(outdir / "n12_phase6_all_seeds_compact.csv", index=False)

summary_rows = []
for case, df in all_df.groupby("case", sort=False):
    summary_rows.append({
        "case": case,
        "runs": len(df),
        "cycle_lengths": ", ".join(map(str, sorted(df["cycle_length"].astype(int).unique()))),
        "median_L": df["cycle_length"].median(),
        "median_phase": df["cycle_phase_length"].median(),
        "median_phase/(pi/2)": df["cycle_phase_over_pi2"].median(),
        "median_mean_reward": df["mean_reward"].median(),
        "min_mean_reward": df["mean_reward"].min(),
        "max_mean_reward": df["mean_reward"].max(),
        "median_mean_reward/dtheta": df["mean_reward_over_dtheta"].median(),
        "median_cycle_reward": df["cycle_reward"].median(),
        "median_tip_area": df["tip_area"].median(),
        "median_tip_path_length": df["tip_path_length"].median(),
        "median_rho": df["rho_reward_per_area"].median(),
        "wrong_orientation_seeds": ", ".join(map(str, df.loc[df["wrong_orientation"].astype(bool), "seed"].astype(int).tolist())),
        "noop_fraction_max": df["noop_fraction"].max(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(outdir / "n12_phase6_summary_compact.csv", index=False)

# Seed picks for plotting
pick_rows = []
for case, df in all_df.groupby("case", sort=False):
    dfm = df.sort_values("mean_reward").reset_index(drop=True)
    picks = [
        ("worst_mean", dfm.iloc[0]),
        ("median_mean", dfm.iloc[len(dfm)//2]),
        ("best_mean", dfm.iloc[-1]),
        ("longest_cycle", df.sort_values("cycle_length").iloc[-1]),
    ]

    # Also include the modal cycle-length median, useful for the clean branch
    mode_L = int(df["cycle_length"].mode().iloc[0])
    mode_df = df[df["cycle_length"] == mode_L].sort_values("mean_reward").reset_index(drop=True)
    picks.append(("modal_L_median", mode_df.iloc[len(mode_df)//2]))

    seen = set()
    for label, row in picks:
        key = (case, int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        r = row.copy()
        r["pick_label"] = label
        pick_rows.append(r)

picks = pd.DataFrame(pick_rows)
pick_cols = [
    "case",
    "pick_label",
    "seed",
    "cycle_length",
    "cycle_phase_length",
    "mean_reward",
    "cycle_reward",
    "tip_area",
    "tip_path_length",
    "rho_reward_per_area",
    "result_path",
]
picks[pick_cols].to_csv(outdir / "n12_phase6_seed_picks_for_vis.csv", index=False)

# Round for LaTeX
summary_tex = summary.copy()
for c in summary_tex.columns:
    if pd.api.types.is_numeric_dtype(summary_tex[c]):
        summary_tex[c] = summary_tex[c].map(lambda x: f"{x:.4g}")
summary_tex.to_latex(outdir / "n12_phase6_summary_compact.tex", index=False)

picks_tex = picks[pick_cols].copy()
for c in picks_tex.columns:
    if pd.api.types.is_numeric_dtype(picks_tex[c]):
        if c == "seed" or c == "cycle_length":
            picks_tex[c] = picks_tex[c].astype(int).astype(str)
        else:
            picks_tex[c] = picks_tex[c].map(lambda x: f"{x:.4g}")
picks_tex.to_latex(outdir / "n12_phase6_seed_picks_for_vis.tex", index=False)

print("\nWrote:")
for p in [
    outdir / "n12_phase6_all_seeds_compact.csv",
    outdir / "n12_phase6_summary_compact.csv",
    outdir / "n12_phase6_seed_picks_for_vis.csv",
    outdir / "n12_phase6_summary_compact.tex",
    outdir / "n12_phase6_seed_picks_for_vis.tex",
]:
    print(" ", p)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSeed picks for visualization:")
print(picks[pick_cols].to_string(index=False))
