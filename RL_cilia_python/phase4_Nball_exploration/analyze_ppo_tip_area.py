#!/usr/bin/env python3
"""
analyze_ppo_tip_area.py
=======================

Add tip-loop geometry diagnostics to PPO sweep summaries.

Reads:
    results/ppo_sweeps_general/summary_all.csv

For each result.npy, computes:
    - tip_signed_area
    - tip_abs_area
    - tip_path_length
    - tip_closure_error
    - cycle_total_reward
    - tip_abs_area_per_step
    - reward_per_abs_tip_area

Writes:
    results/ppo_sweeps_general/summary_all_with_tip_area.csv

Run:
    python analyze_ppo_tip_area.py
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_result(path):
    obj = np.load(path, allow_pickle=True).item()

    angles = np.asarray(obj["angles"], dtype=float)   # radians
    rewards = np.asarray(obj["rewards"], dtype=float)

    cycle_start = int(obj.get("cycle_start", -1))
    cycle_length = int(obj.get("cycle_length", -1))

    if cycle_start >= 0 and cycle_length > 0:
        angles = angles[cycle_start:cycle_start + cycle_length]
        rewards = rewards[cycle_start:cycle_start + cycle_length]

    return angles, rewards


def tip_positions_from_angles(angles_rad, total_length=1.0):
    """
    angles_rad has shape (T, N), relative joint angles in radians.

    Uses total chain length = 1 by default, so each segment has length 1/N.
    This makes tip area comparable across N.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    T, N = angles_rad.shape
    seglen = total_length / N

    tips = np.zeros((T, 2), dtype=float)

    for i in range(T):
        psi = np.cumsum(angles_rad[i])
        x = 0.0
        z = 0.0
        for ang in psi:
            x += seglen * np.sin(ang)
            z += seglen * np.cos(ang)
        tips[i, 0] = x
        tips[i, 1] = z

    return tips


def signed_polygon_area(points):
    """
    Signed area of the closed polygon through points.
    Positive/negative sign depends on orientation.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    # close path by rolling
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def path_length(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0

    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def closure_error(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[-1] - pts[0]))


def compute_cycle_diagnostics(result_path):
    angles, rewards = load_result(result_path)
    tips = tip_positions_from_angles(angles, total_length=1.0)

    signed_area = float(signed_polygon_area(tips))
    abs_area = abs(signed_area)
    plen = path_length(tips)
    cerr = closure_error(tips)

    mean_reward = float(np.mean(rewards)) if len(rewards) else np.nan
    total_reward = float(np.sum(rewards)) if len(rewards) else np.nan
    L = int(len(rewards))

    if L > 0:
        area_per_step = abs_area / L
    else:
        area_per_step = np.nan

    if abs_area > 1e-12:
        reward_per_abs_area = total_reward / abs_area
    else:
        reward_per_abs_area = np.nan

    return {
        "cycle_length_from_file": L,
        "cycle_total_reward": total_reward,
        "cycle_mean_reward_check": mean_reward,
        "tip_signed_area": signed_area,
        "tip_abs_area": abs_area,
        "tip_path_length": plen,
        "tip_closure_error": cerr,
        "tip_abs_area_per_step": area_per_step,
        "reward_per_abs_tip_area": reward_per_abs_area,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/ppo_sweeps_general/summary_all.csv"),
        help="input summary CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ppo_sweeps_general/summary_all_with_tip_area.csv"),
        help="output enhanced CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary)

    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        result_path = Path(row["result_path"])

        if not result_path.exists():
            print(f"[warn] missing result file: {result_path}")
            diag = {
                "cycle_length_from_file": np.nan,
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
            diag = compute_cycle_diagnostics(result_path)

        for k, v in diag.items():
            row[k] = v

        rows.append(row)

    outdf = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(args.out, index=False)

    print(f"[done] wrote {args.out}")

    # quick readable summary
    print()
    print("By N and dtheta:")
    group_cols = ["nballs", "dtheta_label"]
    if all(c in outdf.columns for c in group_cols):
        summary = (
            outdf.groupby(group_cols)
            .agg(
                n=("seed", "count"),
                mean_pct_howard=("ppo_percent_of_howard", "mean"),
                mean_tip_abs_area=("tip_abs_area", "mean"),
                median_tip_abs_area=("tip_abs_area", "median"),
                mean_path_length=("tip_path_length", "mean"),
                mean_cycle_total_reward=("cycle_total_reward", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()