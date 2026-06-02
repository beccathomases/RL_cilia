#!/usr/bin/env python3
"""
add_tip_area_to_all_summaries.py
================================

Simpler tip-area analyzer.

Finds every case-level summary.csv under

    results/ppo_sweeps_general/N*_dtheta_*/summary.csv

and writes, in the same folder,

    summary_with_tip_area.csv

This avoids relying on summary_all.csv, which may be partial while a sweep is
still running.

Run from phase4_Nball_exploration:

    python add_tip_area_to_all_summaries.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("results/ppo_sweeps_general")


def load_cycle(result_path):
    obj = np.load(result_path, allow_pickle=True).item()

    angles = np.asarray(obj["angles"], dtype=float)   # radians
    rewards = np.asarray(obj["rewards"], dtype=float)

    cycle_start = int(obj.get("cycle_start", -1))
    cycle_length = int(obj.get("cycle_length", -1))

    if cycle_start >= 0 and cycle_length > 0:
        angles = angles[cycle_start:cycle_start + cycle_length]
        rewards = rewards[cycle_start:cycle_start + cycle_length]

    return angles, rewards


def tip_positions(angles_rad):
    """
    Use total chain length = 1 so areas are comparable across N.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    T, N = angles_rad.shape
    seglen = 1.0 / N

    tips = np.zeros((T, 2))
    for i in range(T):
        psi = np.cumsum(angles_rad[i])
        x = 0.0
        z = 0.0
        for ang in psi:
            x += seglen * np.sin(ang)
            z += seglen * np.cos(ang)
        tips[i] = [x, z]

    return tips


def signed_area(points):
    """
    Shoelace area of the closed tip path.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def path_length(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def closure_error(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[-1] - pts[0]))


def diagnostics_for_result(result_path):
    angles, rewards = load_cycle(result_path)
    tips = tip_positions(angles)

    A = float(signed_area(tips))
    L = int(len(rewards))
    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards))

    return {
        "cycle_length_check": L,
        "cycle_total_reward": total_reward,
        "cycle_mean_reward_check": mean_reward,
        "tip_signed_area": A,
        "tip_abs_area": abs(A),
        "tip_path_length": path_length(tips),
        "tip_closure_error": closure_error(tips),
        "tip_abs_area_per_step": abs(A) / L if L > 0 else np.nan,
        "reward_per_abs_tip_area": total_reward / abs(A) if abs(A) > 1e-12 else np.nan,
    }


def analyze_one_summary(summary_path):
    df = pd.read_csv(summary_path)
    rows = []

    for _, row in df.iterrows():
        row = row.copy()
        result_path = Path(row["result_path"])

        if not result_path.exists():
            print(f"  [warn] missing {result_path}")
            diag = {
                "cycle_length_check": np.nan,
                "cycle_total_reward": np.nan,
                "cycle_mean_reward_check": np.nan,
                "tip_signed_area": np.nan,
                "tip_abs_area": np.nan,
                "tip_path_length": np.nan,
                "tip_closure_error": np.nan,
                "tip_abs_area_per_step": np.nan,
                "reward_per_abs_tip_area": np.nan,
            }
        else:
            diag = diagnostics_for_result(result_path)

        for k, v in diag.items():
            row[k] = v

        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = summary_path.parent / "summary_with_tip_area.csv"
    out.to_csv(out_path, index=False)

    print(f"  wrote {out_path}")

    return out_path


def main():
    summaries = sorted(ROOT.glob("N*_dtheta_*/summary.csv"))

    if not summaries:
        print(f"No case summary.csv files found under {ROOT}")
        return

    print(f"Found {len(summaries)} summaries.")
    written = []
    all_dfs = []

    for s in summaries:
        print(f"\nAnalyzing {s}")
        out_path = analyze_one_summary(s)
        written.append(out_path)

        # Read back the enhanced summary and keep it for the combined file.
        df = pd.read_csv(out_path)

        # Add a source folder column so we know where each row came from.
        df["source_summary_folder"] = str(s.parent)

        all_dfs.append(df)

    # Write combined summary across all N/dtheta cases.
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)

        # Optional: sort in a readable order if these columns exist.
        sort_cols = []
        for col in ["nballs", "dtheta_label", "seed"]:
            if col in all_df.columns:
                sort_cols.append(col)

        if sort_cols:
            all_df = all_df.sort_values(sort_cols).reset_index(drop=True)

        all_path = ROOT / "summary_all_with_tip_area.csv"
        all_df.to_csv(all_path, index=False)
        print(f"\nWrote combined summary:")
        print(f"  {all_path}")

    print("\nDone. Wrote case summaries:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()