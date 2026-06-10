#!/usr/bin/env python3
"""
detect_primitive_periods.py
===========================

Postprocess cilia PPO / Howard cycles to distinguish:

  L_full  = exact detected closure period
  L_prim  = estimated primitive stroke period
  k       = L_full / L_prim

The detector uses reward autocorrelation plus tip-path recurrence. It is meant
to catch benign k-fold commensurate orbits, e.g. a clean 5-pass orbit, and to
separate them from long path-heavy / modulated orbits.

Supported inputs:
  New PPO result.npz:
      phi, tip, rewards, cycle_start, cycle_length

  Old Phase 4 PPO result.npy:
      dict with angles, rewards, cycle_start, cycle_length

  Howard summary .npz:
      angles, rewards, cycle_start, cycle_length

Example:
  python detect_primitive_periods.py \
    --roots results/ppo_sweeps_n6_t3e6 \
    --outroot results/primitive_periods_n6_t3e6
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------

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
    phi = np.asarray(phi, dtype=float)
    nballs = phi.shape[1]
    Lseg = np.ones(nballs, dtype=float) / nballs

    psi = np.cumsum(phi, axis=1)
    x = np.sum(Lseg[None, :] * np.sin(psi), axis=1)
    z = np.sum(Lseg[None, :] * np.cos(psi), axis=1)

    return np.column_stack([x, z])


# ---------------------------------------------------------------------
# Loading saved cycles
# ---------------------------------------------------------------------

def load_summary_json_near(path):
    summary_path = path.with_name("summary.json")
    if not summary_path.exists():
        return {}

    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_metadata_from_path(path):
    s = str(path)
    out = {}

    m = re.search(r"N(\d+)_dtheta_(pi\d+)", s)
    if m:
        out["nballs"] = int(m.group(1))
        out["dtheta_label"] = m.group(2)

    m = re.search(r"seed_(\d+)", s)
    if m:
        out["seed"] = int(m.group(1))

    m = re.search(r"howard_summary_(\d+)ball", path.name)
    if m:
        out["nballs"] = int(m.group(1))
        out["method"] = "Howard"
        out["seed"] = -1

    if "howard" in s.lower():
        out.setdefault("method", "Howard")
    elif "ppo" in s.lower():
        out.setdefault("method", "PPO")
    else:
        out.setdefault("method", "unknown")

    return out


def align_phi_tip_rewards(phi_all, tip_all, rewards_all, cycle_start, cycle_length):
    """
    Align arrays to transition steps.

    Some old files store one extra terminal/repeated state/angle, so angles
    may have length len(rewards)+1. In that case, drop the final angle.
    """
    phi_all = np.asarray(phi_all, dtype=float)
    rewards_all = np.asarray(rewards_all, dtype=float).reshape(-1)

    if tip_all is None:
        tip_all = phi_to_tip_equal_total_length(phi_all)
    else:
        tip_all = np.asarray(tip_all, dtype=float)

    if len(phi_all) == len(rewards_all) + 1:
        phi_steps = phi_all[:-1]
        tip_steps = tip_all[:-1]
    else:
        phi_steps = phi_all[:len(rewards_all)]
        tip_steps = tip_all[:len(rewards_all)]

    cs = int(cycle_start)
    L = int(cycle_length)

    phi = phi_steps[cs:cs + L]
    tip = tip_steps[cs:cs + L]
    rewards = rewards_all[cs:cs + L]

    if len(phi) != L or len(tip) != L or len(rewards) != L:
        raise ValueError(
            f"Bad cycle slice: len(phi)={len(phi)}, len(tip)={len(tip)}, "
            f"len(rewards)={len(rewards)}, cycle_length={L}, cycle_start={cs}"
        )

    return phi, tip, rewards


def load_cycle(path):
    path = Path(path)
    meta = parse_metadata_from_path(path)
    summary = load_summary_json_near(path)

    for k, v in summary.items():
        if k in ["nballs", "dtheta_label", "seed", "timesteps", "elapsed_sec"]:
            meta[k] = v

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)

        if "phi" in data:
            phi_all = data["phi"]
            tip_all = data["tip"] if "tip" in data else None
            rewards_all = data["rewards"]
            cycle_start = int(np.asarray(data["cycle_start"]).reshape(-1)[0])
            cycle_length = int(np.asarray(data["cycle_length"]).reshape(-1)[0])

        elif "angles" in data:
            phi_all = data["angles"]
            tip_all = None
            rewards_all = data["rewards"]
            cycle_start = int(np.asarray(data["cycle_start"]).reshape(-1)[0])
            cycle_length = int(np.asarray(data["cycle_length"]).reshape(-1)[0])

        else:
            raise KeyError(f"{path} does not contain phi or angles.")

    elif path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True).item()

        phi_all = obj["angles"]
        tip_all = None
        rewards_all = obj["rewards"]
        cycle_start = int(obj["cycle_start"])
        cycle_length = int(obj["cycle_length"])

        train_settings = obj.get("train_settings", {})
        env_settings = obj.get("env_settings", {})

        meta.setdefault("nballs", env_settings.get("Nballs", train_settings.get("Nballs", np.nan)))
        meta.setdefault("seed", train_settings.get("seed", np.nan))
        meta.setdefault("timesteps", train_settings.get("timesteps", np.nan))
        meta.setdefault("dtheta_label", train_settings.get("dtheta_label", ""))

    else:
        raise ValueError(f"Unsupported file type: {path}")

    phi, tip, rewards = align_phi_tip_rewards(
        phi_all=phi_all,
        tip_all=tip_all,
        rewards_all=rewards_all,
        cycle_start=cycle_start,
        cycle_length=cycle_length,
    )

    meta["source_path"] = str(path)
    meta["cycle_start"] = cycle_start
    meta["cycle_length"] = cycle_length

    if "nballs" not in meta or pd.isna(meta["nballs"]):
        meta["nballs"] = phi.shape[1]

    return meta, phi, tip, rewards


# ---------------------------------------------------------------------
# Primitive period detection
# ---------------------------------------------------------------------

def circular_autocorr(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    L = len(x)

    y = x - np.mean(x)
    denom = float(np.dot(y, y))

    corr = np.zeros(L, dtype=float)
    if denom < 1e-14:
        return corr

    for lag in range(L):
        corr[lag] = float(np.dot(y, np.roll(y, -lag)) / denom)

    return corr


def rms_repeat_distance(arr, lag):
    arr = np.asarray(arr, dtype=float)
    diff = arr - np.roll(arr, -lag, axis=0)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def detect_primitive_period(
    phi,
    tip,
    rewards,
    min_period=8,
    min_corr=0.55,
    max_tip_rms_frac=0.30,
    min_k_for_primitive=1.35,
    tip_weight=0.75,
    phi_weight=0.10,
):
    """
    Estimate a primitive period from reward autocorrelation and tip recurrence.

    Returns a dict and a list of candidate dicts.

    If no shorter period passes the filters, L_prim = L_full.
    """
    phi = np.asarray(phi, dtype=float)
    tip = np.asarray(tip, dtype=float)
    rewards = np.asarray(rewards, dtype=float).reshape(-1)

    L = len(rewards)

    if L < 2 * min_period:
        return {
            "L_full": L,
            "L_prim": L,
            "k_est": 1.0,
            "primitive_method": "full_too_short",
            "primitive_reward_corr": np.nan,
            "primitive_tip_rms": 0.0,
            "primitive_tip_rms_frac": 0.0,
            "primitive_phi_rms": 0.0,
            "primitive_score": np.nan,
        }, []

    corr = circular_autocorr(rewards)

    tip_span = np.ptp(tip, axis=0)
    tip_scale = float(np.linalg.norm(tip_span))
    if tip_scale < 1e-12:
        tip_scale = max(path_length(tip, close=True), 1.0)

    phi_scale = math.pi

    max_lag = int(np.floor(L / min_k_for_primitive))
    max_lag = max(min(max_lag, L - min_period), min_period)

    candidates = []

    for lag in range(min_period, max_lag + 1):
        # Local maximum in reward autocorrelation, to avoid choosing a shoulder.
        if lag > 1 and lag < L - 1:
            if corr[lag] < corr[lag - 1] or corr[lag] < corr[lag + 1]:
                continue

        reward_corr = float(corr[lag])
        if reward_corr < min_corr:
            continue

        tip_rms = rms_repeat_distance(tip, lag)
        tip_frac = tip_rms / tip_scale

        if tip_frac > max_tip_rms_frac:
            continue

        phi_rms = rms_repeat_distance(phi, lag)

        score = (
            reward_corr
            - tip_weight * tip_frac
            - phi_weight * min(phi_rms / phi_scale, 2.0)
        )

        candidates.append(
            {
                "lag": int(lag),
                "k_est": float(L / lag),
                "reward_corr": reward_corr,
                "tip_rms": tip_rms,
                "tip_rms_frac": tip_frac,
                "phi_rms": phi_rms,
                "score": float(score),
            }
        )

    candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)

    if not candidates:
        return {
            "L_full": L,
            "L_prim": L,
            "k_est": 1.0,
            "primitive_method": "full_no_candidate",
            "primitive_reward_corr": np.nan,
            "primitive_tip_rms": 0.0,
            "primitive_tip_rms_frac": 0.0,
            "primitive_phi_rms": 0.0,
            "primitive_score": np.nan,
        }, []

    best = candidates[0]

    return {
        "L_full": L,
        "L_prim": int(best["lag"]),
        "k_est": float(best["k_est"]),
        "primitive_method": "reward_tip_autocorr",
        "primitive_reward_corr": float(best["reward_corr"]),
        "primitive_tip_rms": float(best["tip_rms"]),
        "primitive_tip_rms_frac": float(best["tip_rms_frac"]),
        "primitive_phi_rms": float(best["phi_rms"]),
        "primitive_score": float(best["score"]),
    }, candidates


# ---------------------------------------------------------------------
# Metrics and classification
# ---------------------------------------------------------------------

def full_cycle_metrics(phi, tip, rewards):
    A = closed_polygon_area(tip)
    abs_A = abs(A)
    path = path_length(tip, close=True)
    R = float(np.sum(rewards))
    mean_r = float(np.mean(rewards))

    return {
        "cycle_total_reward": R,
        "cycle_mean_reward": mean_r,
        "tip_signed_area": A,
        "tip_abs_area": abs_A,
        "tip_path_length": path,
        "rho_reward_per_abs_area": safe_ratio(R, abs_A),
        "path_per_area": safe_ratio(path, abs_A),
        "wrong_orientation_flag": bool(np.isfinite(safe_ratio(R, abs_A)) and safe_ratio(R, abs_A) < 0),
    }


def add_per_pass_metrics(row):
    k = row["k_est"]
    row["reward_per_pass"] = safe_ratio(row["cycle_total_reward"], k)
    row["area_per_pass"] = safe_ratio(row["tip_abs_area"], k)
    row["path_per_pass"] = safe_ratio(row["tip_path_length"], k)
    row["L_per_pass"] = safe_ratio(row["L_full"], k)
    row["mean_reward_per_step"] = row["cycle_mean_reward"]
    return row


def classify_within_case(
    df,
    benign_path_area_factor=1.35,
    degraded_path_area_factor=1.60,
    k_agree_factor=1.50,
):
    """
    Add case-relative references and classify each run.

    The reference primitive area/path is taken from apparent single-pass runs
    in the same (method, N, dtheta, timesteps) group when available.
    """
    if df.empty:
        return df

    group_cols = ["method", "nballs", "dtheta_label", "timesteps"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = np.nan

    out_pieces = []

    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.copy()

        single_ref = g[(g["k_est"] < 1.35) & (g["cycle_mean_reward"] > 0)]

        if len(single_ref) == 0:
            # Fall back to the lowest-k positive runs.
            positive = g[g["cycle_mean_reward"] > 0]
            if len(positive) > 0:
                single_ref = positive.sort_values("k_est").head(max(1, min(3, len(positive))))
            else:
                single_ref = g.sort_values("k_est").head(max(1, min(3, len(g))))

        ref_area = float(np.nanmedian(single_ref["area_per_pass"]))
        ref_path = float(np.nanmedian(single_ref["path_per_pass"]))
        ref_reward = float(np.nanmedian(single_ref["reward_per_pass"]))
        ref_path_area = float(np.nanmedian(single_ref["path_per_area"]))

        g["ref_area_per_pass"] = ref_area
        g["ref_path_per_pass"] = ref_path
        g["ref_reward_per_pass"] = ref_reward
        g["ref_path_per_area"] = ref_path_area

        g["k_area_ref"] = g["tip_abs_area"] / ref_area if abs(ref_area) > 1e-14 else np.nan
        g["k_path_ref"] = g["tip_path_length"] / ref_path if abs(ref_path) > 1e-14 else np.nan
        g["k_reward_ref"] = g["cycle_total_reward"] / ref_reward if abs(ref_reward) > 1e-14 else np.nan
        g["path_area_ratio_rel"] = g["path_per_area"] / ref_path_area if abs(ref_path_area) > 1e-14 else np.nan
        g["k_path_over_k_area"] = g["k_path_ref"] / g["k_area_ref"]

        labels = []
        for _, r in g.iterrows():
            if bool(r.get("wrong_orientation_flag", False)) or r["cycle_mean_reward"] <= 0:
                labels.append("failure_or_wrong_orientation")
                continue

            if r["k_est"] < 1.35:
                labels.append("single_pass")
                continue

            path_area_rel = r["path_area_ratio_rel"]
            k_ratio = r["k_path_over_k_area"]

            if (
                np.isfinite(path_area_rel)
                and np.isfinite(k_ratio)
                and path_area_rel <= benign_path_area_factor
                and (1 / k_agree_factor) <= k_ratio <= k_agree_factor
            ):
                labels.append("benign_kfold")
            elif (
                np.isfinite(path_area_rel)
                and (
                    path_area_rel >= degraded_path_area_factor
                    or (np.isfinite(k_ratio) and k_ratio > k_agree_factor)
                )
            ):
                labels.append("modulated_path_heavy")
            else:
                labels.append("long_unclear")

        g["primitive_class"] = labels
        out_pieces.append(g)

    return pd.concat(out_pieces, ignore_index=True)


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def find_result_files(roots):
    files = []

    for root in roots:
        root = Path(root)

        files.extend(sorted(root.rglob("result.npz")))
        files.extend(sorted(root.rglob("result.npy")))
        files.extend(sorted(root.rglob("howard_summary_*ball.npz")))

    # De-duplicate.
    seen = set()
    unique = []
    for p in files:
        s = str(p)
        if s not in seen:
            unique.append(p)
            seen.add(s)

    return unique


def summarize_by_case(df):
    if df.empty:
        return df

    group_cols = ["method", "nballs", "dtheta_label", "timesteps"]

    metrics = [
        "L_full",
        "L_prim",
        "k_est",
        "cycle_mean_reward",
        "rho_reward_per_abs_area",
        "tip_abs_area",
        "tip_path_length",
        "path_per_area",
        "area_per_pass",
        "path_per_pass",
        "reward_per_pass",
        "primitive_reward_corr",
        "primitive_tip_rms_frac",
        "path_area_ratio_rel",
    ]

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["runs"] = len(g)

        for cls, count in g["primitive_class"].value_counts().items():
            row[f"class_count_{cls}"] = int(count)

        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_median"] = vals.median()
            row[f"{m}_min"] = vals.min()
            row[f"{m}_max"] = vals.max()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--outroot", required=True)

    parser.add_argument("--min-period", type=int, default=8)
    parser.add_argument("--min-corr", type=float, default=0.55)
    parser.add_argument("--max-tip-rms-frac", type=float, default=0.30)
    parser.add_argument("--min-k-for-primitive", type=float, default=1.35)

    parser.add_argument("--benign-path-area-factor", type=float, default=1.35)
    parser.add_argument("--degraded-path-area-factor", type=float, default=1.60)
    parser.add_argument("--k-agree-factor", type=float, default=1.50)

    args = parser.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    result_files = find_result_files(args.roots)
    print(f"Found {len(result_files)} candidate result files")

    rows = []
    candidate_rows = []
    failures = []

    for path in result_files:
        try:
            meta, phi, tip, rewards = load_cycle(path)

            metrics = full_cycle_metrics(phi, tip, rewards)

            prim, candidates = detect_primitive_period(
                phi=phi,
                tip=tip,
                rewards=rewards,
                min_period=args.min_period,
                min_corr=args.min_corr,
                max_tip_rms_frac=args.max_tip_rms_frac,
                min_k_for_primitive=args.min_k_for_primitive,
            )

            row = {}
            row.update(meta)
            row.update(metrics)
            row.update(prim)
            row = add_per_pass_metrics(row)
            rows.append(row)

            for rank, c in enumerate(candidates[:10], start=1):
                cr = dict(meta)
                cr["candidate_rank"] = rank
                cr.update(c)
                candidate_rows.append(cr)

        except Exception as e:
            failures.append({"source_path": str(path), "error": repr(e)})

    df = pd.DataFrame(rows)

    if not df.empty:
        df = classify_within_case(
            df,
            benign_path_area_factor=args.benign_path_area_factor,
            degraded_path_area_factor=args.degraded_path_area_factor,
            k_agree_factor=args.k_agree_factor,
        )

    by_run = outroot / "primitive_by_run.csv"
    df.to_csv(by_run, index=False)
    print(f"[wrote] {by_run}")

    cand_df = pd.DataFrame(candidate_rows)
    candidates_path = outroot / "primitive_candidates.csv"
    cand_df.to_csv(candidates_path, index=False)
    print(f"[wrote] {candidates_path}")

    by_case = summarize_by_case(df)
    by_case_path = outroot / "primitive_by_case.csv"
    by_case.to_csv(by_case_path, index=False)
    print(f"[wrote] {by_case_path}")

    if failures:
        fail_path = outroot / "primitive_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"[warn] {len(failures)} failures written to {fail_path}")

    if not df.empty:
        print("\nPreview:")
        cols = [
            "method", "nballs", "dtheta_label", "seed", "timesteps",
            "L_full", "L_prim", "k_est",
            "cycle_mean_reward", "rho_reward_per_abs_area",
            "tip_abs_area", "tip_path_length", "path_per_area",
            "area_per_pass", "path_per_pass",
            "primitive_reward_corr", "primitive_tip_rms_frac",
            "path_area_ratio_rel", "primitive_class",
        ]
        existing_cols = [c for c in cols if c in df.columns]
        print(df[existing_cols].sort_values(
            [c for c in ["nballs", "dtheta_label", "timesteps", "seed"] if c in df.columns]
        ).to_string(index=False))


if __name__ == "__main__":
    main()