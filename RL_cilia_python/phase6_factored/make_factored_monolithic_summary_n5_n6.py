#!/usr/bin/env python3
"""
make_factored_monolithic_summary_n5_n6.py
========================================

Compare monolithic vs factored PPO sweeps for N=5 and N=6.

No Howard benchmark is used here. The table summarizes each
(N, dtheta, action_mode) case by seed-level medians/min/maxes.

Run from either phase6_factored or the parent RL_cilia_python folder:

    python make_factored_monolithic_summary_n5_n6.py

Outputs:
    results/comparison_n5_n6_factored_vs_monolithic/by_seed.csv
    results/comparison_n5_n6_factored_vs_monolithic/by_case.csv
    results/comparison_n5_n6_factored_vs_monolithic/compact.csv

Edit PATHS below if your result folders differ.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------

HERE = Path.cwd()

# If running from phase6_factored, parent is RL_cilia_python.
# If running from RL_cilia_python, parent is itself.
if HERE.name == "phase6_factored":
    BASE = HERE.parent
else:
    BASE = HERE

PHASE5 = BASE / "phase5_5ball"
PHASE6 = BASE / "phase6_factored"

# Edit these if needed.
PATHS = [
    # N=5 monolithic, Phase 5
    {
        "N": 5,
        "action_mode": "monolithic",
        "budget_label": "2M",
        "root": PHASE5 / "results" / "ppo_sweeps_n5",
    },
    # N=5 factored, Phase 6
    {
        "N": 5,
        "action_mode": "factored",
        "budget_label": "2M",
        "root": PHASE6 / "results" / "ppo_sweeps_phase6_n5_t2e6",
    },
    # N=6 monolithic, Phase 5
    {
        "N": 6,
        "action_mode": "monolithic",
        "budget_label": "3M",
        "root": PHASE5 / "results" / "ppo_sweeps_n6_t3e6",
    },
    # N=6 factored, Phase 6
    {
        "N": 6,
        "action_mode": "factored",
        "budget_label": "3M",
        "root": PHASE6 / "results" / "ppo_sweeps_phase6_n6_t3e6",
    },
]

OUTROOT = PHASE6 / "results" / "comparison_n5_n6_factored_vs_monolithic"


# ---------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------

MEAN_CANDIDATES = ["cycle_mean_reward", "ppo_avg_reward", "cycle_mean_reward_check"]
TOTAL_CANDIDATES = ["cycle_total_reward"]
AREA_CANDIDATES = ["tip_abs_area"]
PATH_CANDIDATES = ["tip_path_length"]
RHO_CANDIDATES = ["reward_per_abs_tip_area", "rho_reward_per_abs_area"]
WRONG_CANDIDATES = ["wrong_orientation_flag", "wrong_orientation"]
NOOP_CANDIDATES = ["noop_fraction"]


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"missing required column; tried {candidates}")
    return None


def read_summary_files(root: Path) -> pd.DataFrame:
    """Read per-case summary.csv files under root.

    Prefer per-case summary.csv files over summary_all.csv to avoid duplicate
    rows, but fall back to summary_all.csv if no per-case summaries exist.
    """
    root = Path(root)

    if not root.exists():
        print(f"[missing] {root}")
        return pd.DataFrame()

    per_case = sorted(
        p for p in root.glob("N*_dtheta_*_*/summary.csv")
        if p.name == "summary.csv"
    )

    if per_case:
        frames = []
        for p in per_case:
            df = pd.read_csv(p)
            df["source_csv"] = str(p)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    summary_all = root / "summary_all.csv"
    if summary_all.exists():
        df = pd.read_csv(summary_all)
        df["source_csv"] = str(summary_all)
        return df

    print(f"[missing summaries] {root}")
    return pd.DataFrame()


def standardize(df: pd.DataFrame, *, N: int, action_mode: str, budget_label: str, root: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    mean_col = find_col(df, MEAN_CANDIDATES, required=True)
    total_col = find_col(df, TOTAL_CANDIDATES)
    area_col = find_col(df, AREA_CANDIDATES)
    path_col = find_col(df, PATH_CANDIDATES)
    rho_col = find_col(df, RHO_CANDIDATES)
    wrong_col = find_col(df, WRONG_CANDIDATES)
    noop_col = find_col(df, NOOP_CANDIDATES)

    out = pd.DataFrame()

    # Prefer columns from file if present; otherwise use config.
    out["N"] = pd.to_numeric(df["nballs"], errors="coerce") if "nballs" in df.columns else N
    out["N"] = out["N"].fillna(N).astype(int)

    if "dtheta_label" not in df.columns:
        raise KeyError(f"{root} summary has no dtheta_label column")
    out["dtheta"] = df["dtheta_label"].astype(str)

    out["seed"] = pd.to_numeric(df["seed"], errors="coerce") if "seed" in df.columns else np.arange(len(df))
    out["seed"] = out["seed"].astype(int)

    out["action_mode"] = action_mode
    out["budget_label"] = budget_label
    out["root"] = str(root)

    out["timesteps"] = pd.to_numeric(df["timesteps"], errors="coerce") if "timesteps" in df.columns else np.nan
    out["cycle_length"] = pd.to_numeric(df["cycle_length"], errors="coerce") if "cycle_length" in df.columns else np.nan

    out["cycle_mean_reward"] = pd.to_numeric(df[mean_col], errors="coerce")
    out["cycle_total_reward"] = pd.to_numeric(df[total_col], errors="coerce") if total_col else np.nan
    out["tip_abs_area"] = pd.to_numeric(df[area_col], errors="coerce") if area_col else np.nan
    out["tip_path_length"] = pd.to_numeric(df[path_col], errors="coerce") if path_col else np.nan

    if rho_col:
        out["rho_reward_per_abs_area"] = pd.to_numeric(df[rho_col], errors="coerce")
    else:
        out["rho_reward_per_abs_area"] = out["cycle_total_reward"] / out["tip_abs_area"]

    if wrong_col:
        out["wrong_orientation_flag"] = df[wrong_col].astype(bool)
    else:
        out["wrong_orientation_flag"] = out["rho_reward_per_abs_area"] < 0

    out["noop_fraction"] = pd.to_numeric(df[noop_col], errors="coerce") if noop_col else np.nan

    if "source_csv" in df.columns:
        out["source_csv"] = df["source_csv"].astype(str)
    else:
        out["source_csv"] = ""

    # Keep only requested N and real rows.
    out = out[
        (out["N"] == int(N))
        & out["seed"].notna()
        & out["cycle_mean_reward"].notna()
    ].copy()

    return out


def load_all() -> pd.DataFrame:
    pieces = []

    for spec in PATHS:
        raw = read_summary_files(spec["root"])
        if raw.empty:
            continue

        std = standardize(
            raw,
            N=spec["N"],
            action_mode=spec["action_mode"],
            budget_label=spec["budget_label"],
            root=spec["root"],
        )

        if len(std):
            pieces.append(std)
            print(
                f"[loaded] N={spec['N']} {spec['action_mode']} "
                f"{spec['budget_label']}: {len(std)} rows from {spec['root']}"
            )
        else:
            print(f"[no rows] {spec}")

    if not pieces:
        raise RuntimeError("No data loaded. Check PATHS at top of script.")

    df = pd.concat(pieces, ignore_index=True)

    # Remove duplicate rows if summary_all and per-case summaries both slipped in.
    df = df.sort_values("timesteps", na_position="first")
    df = df.drop_duplicates(
        subset=["N", "dtheta", "action_mode", "seed", "budget_label"],
        keep="last",
    ).reset_index(drop=True)

    return df


def summarize_case(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (N, dtheta, action_mode, budget), g in df.groupby(["N", "dtheta", "action_mode", "budget_label"]):
        lengths = sorted(set(int(x) for x in g["cycle_length"].dropna()))

        rows.append({
            "N": int(N),
            "dtheta": dtheta,
            "action_mode": action_mode,
            "budget": budget,
            "runs": int(len(g)),
            "timesteps_median": float(np.nanmedian(g["timesteps"])) if g["timesteps"].notna().any() else np.nan,

            "mean_reward_median": float(np.nanmedian(g["cycle_mean_reward"])),
            "mean_reward_min": float(np.nanmin(g["cycle_mean_reward"])),
            "mean_reward_max": float(np.nanmax(g["cycle_mean_reward"])),

            "rho_median": float(np.nanmedian(g["rho_reward_per_abs_area"])),
            "rho_min": float(np.nanmin(g["rho_reward_per_abs_area"])),
            "rho_max": float(np.nanmax(g["rho_reward_per_abs_area"])),

            "area_median": float(np.nanmedian(g["tip_abs_area"])) if g["tip_abs_area"].notna().any() else np.nan,
            "path_median": float(np.nanmedian(g["tip_path_length"])) if g["tip_path_length"].notna().any() else np.nan,

            "cycle_length_median": float(np.nanmedian(g["cycle_length"])) if g["cycle_length"].notna().any() else np.nan,
            "cycle_lengths": lengths,

            "wrong_orientation_count": int(g["wrong_orientation_flag"].sum()),
            "noop_fraction_median": float(np.nanmedian(g["noop_fraction"])) if g["noop_fraction"].notna().any() else np.nan,
            "noop_fraction_max": float(np.nanmax(g["noop_fraction"])) if g["noop_fraction"].notna().any() else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["N", "dtheta", "action_mode"]).reset_index(drop=True)


def add_pairwise_differences(case: pd.DataFrame) -> pd.DataFrame:
    """Add factored - monolithic differences when both arms are present."""
    rows = []

    for (N, dtheta), g in case.groupby(["N", "dtheta"]):
        arms = {r["action_mode"]: r for _, r in g.iterrows()}
        if "factored" not in arms or "monolithic" not in arms:
            continue

        f = arms["factored"]
        m = arms["monolithic"]

        rows.append({
            "N": int(N),
            "dtheta": dtheta,
            "factored_minus_monolithic_mean_reward_median": f["mean_reward_median"] - m["mean_reward_median"],
            "factored_over_monolithic_mean_reward_median": f["mean_reward_median"] / m["mean_reward_median"],
            "factored_minus_monolithic_rho_median": f["rho_median"] - m["rho_median"],
            "factored_minus_monolithic_area_median": f["area_median"] - m["area_median"],
            "factored_minus_monolithic_path_median": f["path_median"] - m["path_median"],
            "factored_wrong_count": int(f["wrong_orientation_count"]),
            "monolithic_wrong_count": int(m["wrong_orientation_count"]),
            "factored_noop_median": f["noop_fraction_median"],
            "factored_noop_max": f["noop_fraction_max"],
        })

    return pd.DataFrame(rows).sort_values(["N", "dtheta"]).reset_index(drop=True)


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)

    by_seed = load_all()
    by_case = summarize_case(by_seed)
    diff = add_pairwise_differences(by_case)

    by_seed_path = OUTROOT / "by_seed.csv"
    by_case_path = OUTROOT / "by_case.csv"
    diff_path = OUTROOT / "factored_minus_monolithic.csv"

    by_seed.to_csv(by_seed_path, index=False)
    by_case.to_csv(by_case_path, index=False)
    diff.to_csv(diff_path, index=False)

    compact_cols = [
        "N", "dtheta", "action_mode", "budget", "runs",
        "mean_reward_median", "mean_reward_min", "mean_reward_max",
        "rho_median", "area_median", "path_median",
        "cycle_length_median", "cycle_lengths",
        "wrong_orientation_count", "noop_fraction_median", "noop_fraction_max",
    ]

    print("\n" + "="*100)
    print("N=5/6 monolithic vs factored summary")
    print("="*100)
    print(by_case[compact_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if len(diff):
        print("\n" + "="*100)
        print("Factored - monolithic differences")
        print("="*100)
        print(diff.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n[wrote]")
    print(f"  {by_seed_path}")
    print(f"  {by_case_path}")
    print(f"  {diff_path}")


if __name__ == "__main__":
    main()
