#!/usr/bin/env python3
"""
Rebuild case-level PPO summary.csv files from all seed_*/summary.json files.

This fixes the problem where rerunning only selected seeds overwrote
N*_dtheta_*/summary.csv with only those seeds.

Run from phase4_Nball_exploration:

    python rebuild_summaries_from_seed_json.py
    python add_tip_area_to_all_summaries.py
    python make_phase4_csv_tables.py
"""

from pathlib import Path
import json
import pandas as pd


ROOT = Path("results/ppo_sweeps_general")


FIELD_ORDER = [
    "nballs",
    "dtheta_label",
    "dtheta",
    "seed",
    "timesteps",
    "elapsed_sec",
    "cycle_start",
    "cycle_length",
    "ppo_avg_reward",
    "howard_gain",
    "howard_cycle_length",
    "ppo_fraction_of_howard",
    "ppo_percent_of_howard",
    "matches_howard_length",
    "result_path",
    "model_path",
]


def rebuild_case(case_dir):
    rows = []

    for seed_dir in sorted(case_dir.glob("seed_*")):
        summary_json = seed_dir / "summary.json"
        if not summary_json.exists():
            print(f"  [skip] missing {summary_json}")
            continue

        with open(summary_json, "r") as f:
            row = json.load(f)

        rows.append(row)

    if not rows:
        print(f"  [warn] no seed summaries found in {case_dir}")
        return None

    df = pd.DataFrame(rows)

    # Keep known fields first, then any extras.
    ordered = [c for c in FIELD_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    # Sort by seed.
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df = df.sort_values("seed").reset_index(drop=True)

    out = case_dir / "summary.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} rows)")
    return out


def main():
    case_dirs = sorted(ROOT.glob("N*_dtheta_*"))

    if not case_dirs:
        print(f"No case directories found under {ROOT}")
        return

    print(f"Found {len(case_dirs)} case directories.")

    for case_dir in case_dirs:
        print(f"\nRebuilding {case_dir}")
        rebuild_case(case_dir)

    print("\nDone.")
    print("Next run:")
    print("  python add_tip_area_to_all_summaries.py")
    print("  python make_phase4_csv_tables.py")


if __name__ == "__main__":
    main()