#!/usr/bin/env python3
"""
make_ppo_npz_efficiency_tables.py
=================================

Compute scaled efficiency diagnostics from PPO result.npz files.

Designed for Phase 5 N=5/N=6 runs, where each seed directory contains:

    result.npz
    summary.json

The script does not retrain anything. It reads saved cycles and writes:

    efficiency_by_run.csv
    efficiency_by_case.csv
    compact_efficiency_all_runs.csv
    compact_efficiency_positive_only.csv

Example for N=5:

    python make_ppo_npz_efficiency_tables.py \
      --ppo-root results/ppo_sweeps_n5 \
      --outroot results/efficiency_n5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-14:
        return np.nan
    return float(a / b)


def closed_polygon_area(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    x2 = np.r_[x, x[0]]
    y2 = np.r_[y, y[0]]

    return float(0.5 * np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))


def path_length(points, close=True):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0

    if close:
        pts = np.vstack([pts, pts[0]])

    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def phi_to_tip_equal_total_length(phi):
    """
    Recompute tip path assuming total chain length is 1.

    This is useful as a consistency check across N. The saved tip path is also
    used by default for the main geometry diagnostics.
    """
    phi = np.asarray(phi, dtype=float)
    nballs = phi.shape[1]
    Lseg = np.ones(nballs, dtype=float) / nballs

    psi = np.cumsum(phi, axis=1)
    x = np.sum(Lseg[None, :] * np.sin(psi), axis=1)
    z = np.sum(Lseg[None, :] * np.cos(psi), axis=1)

    return np.column_stack([x, z])


def parse_metadata_from_path(path):
    s = str(path)

    m = re.search(r"N(\d+)_dtheta_(pi\d+)/seed_(\d+)", s)
    if m is None:
        raise ValueError(f"Could not parse metadata from path: {path}")

    return int(m.group(1)), m.group(2), int(m.group(3))


def load_summary(path):
    summary_path = path.with_name("summary.json")
    if not summary_path.exists():
        return {}

    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def compute_diagnostics(phi_all, tip_all, rewards_all, cycle_start, cycle_length):
    phi_all = np.asarray(phi_all, dtype=float)
    tip_all = np.asarray(tip_all, dtype=float)
    rewards_all = np.asarray(rewards_all, dtype=float).reshape(-1)

    cs = int(cycle_start)
    Lcyc = int(cycle_length)

    phi = phi_all[cs:cs + Lcyc]
    tip = tip_all[cs:cs + Lcyc]
    rewards = rewards_all[cs:cs + Lcyc]

    if len(phi) != Lcyc or len(tip) != Lcyc or len(rewards) != Lcyc:
        raise ValueError(
            f"Bad cycle slice: len(phi)={len(phi)}, len(tip)={len(tip)}, "
            f"len(rewards)={len(rewards)}, cycle_length={Lcyc}, cycle_start={cs}"
        )

    nballs = phi.shape[1]

    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards))

    signed_area = closed_polygon_area(tip)
    abs_area = abs(signed_area)
    tip_path = path_length(tip, close=True)
    rho = safe_ratio(total_reward, abs_area)

    # Optional recomputed geometry using equal total length = 1.
    # This should usually match the saved tip path if the run used the same
    # normalization convention.
    tip_equalL = phi_to_tip_equal_total_length(phi)
    signed_area_equalL = closed_polygon_area(tip_equalL)
    abs_area_equalL = abs(signed_area_equalL)
    tip_path_equalL = path_length(tip_equalL, close=True)
    rho_equalL = safe_ratio(total_reward, abs_area_equalL)

    # Closed-cycle joint-angle increments.
    dphi = np.roll(phi, shift=-1, axis=0) - phi

    # Phase-normalized angle path:
    # approximates integral sqrt(mean_j |phi_tau|^2) d tau.
    angle_path_rms = float(np.sum(np.sqrt(np.mean(dphi**2, axis=1))))

    # Phase-normalized squared effort:
    # approximates integral mean_j |phi_tau|^2 d tau.
    scaled_phase_effort = float(Lcyc * np.sum(np.mean(dphi**2, axis=1)))

    # Body roughness diagnostics.
    if nballs >= 2:
        psi = np.cumsum(phi, axis=1)

        d_body_phi = np.diff(phi, axis=1)
        d_body_psi = np.diff(psi, axis=1)

        phi_body_roughness = float(np.mean((nballs - 1) * np.sum(d_body_phi**2, axis=1)))
        psi_body_roughness = float(np.mean((nballs - 1) * np.sum(d_body_psi**2, axis=1)))
    else:
        phi_body_roughness = np.nan
        psi_body_roughness = np.nan

    return {
        "cycle_length": Lcyc,
        "cycle_total_reward": total_reward,
        "cycle_mean_reward": mean_reward,

        # Main geometry, using saved tip path.
        "tip_signed_area": signed_area,
        "tip_abs_area": abs_area,
        "tip_path_length": tip_path,
        "rho_reward_per_abs_area": rho,
        "wrong_orientation_flag": bool(np.isfinite(rho) and rho < 0),

        # Geometry recomputed from phi with total chain length normalized to 1.
        "tip_signed_area_equalL": signed_area_equalL,
        "tip_abs_area_equalL": abs_area_equalL,
        "tip_path_length_equalL": tip_path_equalL,
        "rho_reward_per_abs_area_equalL": rho_equalL,

        # Efficiency-style diagnostics.
        "angle_path_rms": angle_path_rms,
        "scaled_phase_effort": scaled_phase_effort,
        "reward_per_tip_path": safe_ratio(total_reward, tip_path),
        "reward_per_angle_path_rms": safe_ratio(total_reward, angle_path_rms),
        "reward_per_scaled_phase_effort": safe_ratio(total_reward, scaled_phase_effort),
        "area_per_scaled_phase_effort": safe_ratio(abs_area, scaled_phase_effort),
        "tip_path_per_scaled_phase_effort": safe_ratio(tip_path, scaled_phase_effort),

        # Same efficiency ratios using equal-length recomputed area/path.
        "area_equalL_per_scaled_phase_effort": safe_ratio(abs_area_equalL, scaled_phase_effort),
        "tip_path_equalL_per_scaled_phase_effort": safe_ratio(tip_path_equalL, scaled_phase_effort),

        # Chain-shape diagnostics.
        "phi_body_roughness": phi_body_roughness,
        "psi_body_roughness": psi_body_roughness,

        # Debugging / resolution checks.
        "raw_sum_dphi2": float(np.sum(dphi**2)),
        "raw_mean_dphi2": float(np.mean(dphi**2)),
        "max_abs_dphi": float(np.max(np.abs(dphi))) if dphi.size else np.nan,
    }


def process_result_npz(path):
    nballs, dtheta_label, seed = parse_metadata_from_path(path)
    summary = load_summary(path)

    data = np.load(path, allow_pickle=True)

    required = ["phi", "tip", "rewards", "cycle_start", "cycle_length"]
    for k in required:
        if k not in data:
            raise KeyError(f"{path} is missing required key {k!r}")

    row = {
        "method": "PPO",
        "nballs": int(summary.get("nballs", nballs)),
        "dtheta_label": summary.get("dtheta_label", dtheta_label),
        "dtheta": summary.get("dtheta", np.nan),
        "seed": int(summary.get("seed", seed)),
        "timesteps": summary.get("timesteps", np.nan),
        "elapsed_sec": summary.get("elapsed_sec", np.nan),
        "source_path": str(path),
        "model_path": summary.get("model_path", ""),
    }

    # Keep original summary fields too, with a prefix.
    for k, v in summary.items():
        row[f"summary_{k}"] = v

    row.update(
        compute_diagnostics(
            phi_all=data["phi"],
            tip_all=data["tip"],
            rewards_all=data["rewards"],
            cycle_start=int(np.asarray(data["cycle_start"]).reshape(-1)[0]),
            cycle_length=int(np.asarray(data["cycle_length"]).reshape(-1)[0]),
        )
    )

    return row


def make_case_table(df):
    group_cols = ["method", "nballs", "dtheta_label", "timesteps"]

    metrics = [
        "cycle_length",
        "cycle_mean_reward",
        "cycle_total_reward",
        "tip_abs_area",
        "tip_path_length",
        "rho_reward_per_abs_area",
        "angle_path_rms",
        "scaled_phase_effort",
        "reward_per_tip_path",
        "reward_per_angle_path_rms",
        "reward_per_scaled_phase_effort",
        "area_per_scaled_phase_effort",
        "phi_body_roughness",
        "psi_body_roughness",
    ]

    agg = {}
    for m in metrics:
        agg[m] = ["count", "median", "min", "max"]

    out = df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = ["_".join(c).strip("_") for c in out.columns]
    return out.reset_index()


def make_compact_table(df, outroot, label):
    group_cols = ["method", "nballs", "dtheta_label", "timesteps"]

    metrics = [
        "cycle_mean_reward",
        "cycle_total_reward",
        "tip_abs_area",
        "tip_path_length",
        "rho_reward_per_abs_area",
        "angle_path_rms",
        "scaled_phase_effort",
        "reward_per_tip_path",
        "reward_per_angle_path_rms",
        "reward_per_scaled_phase_effort",
        "area_per_scaled_phase_effort",
        "phi_body_roughness",
        "psi_body_roughness",
    ]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["runs"] = len(g)
        row["positive_runs"] = int(((g["cycle_mean_reward"] > 0) & (g["rho_reward_per_abs_area"] > 0)).sum())

        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_median"] = vals.median()
            row[f"{m}_min"] = vals.min()
            row[f"{m}_max"] = vals.max()

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["method", "nballs", "dtheta_label", "timesteps"])
    path = outroot / f"compact_efficiency_{label}.csv"
    out.to_csv(path, index=False)
    print(f"[wrote] {path}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-root", default="results/ppo_sweeps_n5")
    parser.add_argument("--outroot", default="results/efficiency_n5")
    args = parser.parse_args()

    ppo_root = Path(args.ppo_root)
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    result_files = sorted(ppo_root.rglob("result.npz"))
    print(f"Found {len(result_files)} result.npz files under {ppo_root}")

    if not result_files:
        raise RuntimeError("No result.npz files found. Check --ppo-root.")

    rows = []
    failures = []

    for path in result_files:
        try:
            rows.append(process_result_npz(path))
        except Exception as e:
            failures.append((str(path), repr(e)))

    df = pd.DataFrame(rows)

    by_run = outroot / "efficiency_by_run.csv"
    df.to_csv(by_run, index=False)
    print(f"[wrote] {by_run}")

    by_case = outroot / "efficiency_by_case.csv"
    case_table = make_case_table(df)
    case_table.to_csv(by_case, index=False)
    print(f"[wrote] {by_case}")

    if failures:
        fail_path = outroot / "efficiency_failures.csv"
        pd.DataFrame(failures, columns=["path", "error"]).to_csv(fail_path, index=False)
        print(f"[warn] {len(failures)} failures written to {fail_path}")

    df["in_family_basic"] = (df["cycle_mean_reward"] > 0) & (df["rho_reward_per_abs_area"] > 0)

    all_table = make_compact_table(df, outroot, "all_runs")
    positive_table = make_compact_table(df[df["in_family_basic"]].copy(), outroot, "positive_only")

    print("\nPositive-only compact preview:")
    cols = [
        "method",
        "nballs",
        "dtheta_label",
        "timesteps",
        "runs",
        "cycle_mean_reward_median",
        "rho_reward_per_abs_area_median",
        "scaled_phase_effort_median",
        "reward_per_scaled_phase_effort_median",
        "area_per_scaled_phase_effort_median",
    ]
    print(positive_table[cols].to_string(index=False))


if __name__ == "__main__":
    main()