#!/usr/bin/env python3
"""
merge_comparison_phase6_v3.py

Build a Phase-6 comparison table using:
  - Howard benchmarks from Phase-4 Howard npz files and/or Phase-4 per-seed summaries
  - Phase-4 monolithic PPO per-seed summary.json files
  - Phase-6 factored PPO summary CSV files

Run from phase6_factored:
    python merge_comparison_phase6_v3.py

Outputs:
    factored_vs_monolithic_vs_howard.csv
    factored_vs_monolithic_vs_howard_by_seed.csv
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = "/Users/bthomases/Documents/Student_Projects/RL_cilia/RL_cilia_python"

PHASE4_MONO_ROOT = os.path.join(
    BASE, "phase4_Nball_exploration", "results", "ppo_sweeps_general"
)

PHASE4_HOWARD_ROOT = os.path.join(
    BASE, "phase4_Nball_exploration", "results", "howard"
)

PHASE4_EFFICIENCY_BY_RUN = os.path.join(
    BASE, "phase4_Nball_exploration", "results",
    "efficiency_phase4", "efficiency_by_run.csv"
)

FACTORED_SEARCH_ROOTS = [
    os.path.join(BASE, "phase6_factored", "results", "ppo_sweeps_phase6"),
]

N_VALUES = [3, 4]
OUTPUT_CSV = "factored_vs_monolithic_vs_howard.csv"

MEAN_CANDIDATES = ["cycle_mean_reward", "ppo_avg_reward", "cycle_mean_reward_check"]
RHO_CANDIDATES = ["reward_per_abs_tip_area", "rho_reward_per_abs_area"]
AREA_CANDIDATES = ["tip_abs_area"]
LEN_CANDIDATES = ["cycle_length", "cycle_length_check"]


def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_phase4_path(path):
    s = str(path)
    m = re.search(r"N(\d+)_dtheta_(pi\d+)/seed_(\d+)", s)
    if not m:
        return {}
    return {
        "nballs": int(m.group(1)),
        "dtheta_label": m.group(2),
        "seed": int(m.group(3)),
    }


def load_phase4_monolithic():
    rows = []
    paths = sorted(Path(PHASE4_MONO_ROOT).glob("N*_dtheta_*/seed_*/summary.json"))

    if not paths:
        raise FileNotFoundError(f"No Phase-4 seed summaries found under {PHASE4_MONO_ROOT}")

    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)

        meta = parse_phase4_path(p)

        nballs = int(d.get("nballs", meta.get("nballs", -1)))
        dtheta_label = str(d.get("dtheta_label", meta.get("dtheta_label", "")))
        seed = int(d.get("seed", meta.get("seed", -1)))

        mean_reward = d.get("cycle_mean_reward", d.get("ppo_avg_reward", np.nan))

        rows.append({
            "nballs": nballs,
            "dtheta_label": dtheta_label,
            "seed": seed,
            "timesteps": d.get("timesteps", np.nan),
            "mean_reward": mean_reward,
            "cycle_length": d.get("cycle_length", np.nan),
            "rho": d.get("reward_per_abs_tip_area", np.nan),
            "area": d.get("tip_abs_area", np.nan),
            "noop_fraction": np.nan,
            "arm": "monolithic",
            "source": str(p),
            "howard_gain_from_summary": d.get("howard_gain", np.nan),
            "howard_L_from_summary": d.get("howard_cycle_length", np.nan),
        })

    df = pd.DataFrame(rows)

    if os.path.isfile(PHASE4_EFFICIENCY_BY_RUN):
        try:
            eff = pd.read_csv(PHASE4_EFFICIENCY_BY_RUN)
            eff = eff[eff["method"].astype(str) == "PPO"].copy()
            eff = eff.rename(columns={
                "rho_reward_per_abs_area": "rho_eff",
                "tip_abs_area": "area_eff",
                "cycle_mean_reward": "mean_reward_eff",
                "cycle_length": "cycle_length_eff",
            })
            keep = [
                "nballs", "dtheta_label", "seed",
                "rho_eff", "area_eff", "mean_reward_eff", "cycle_length_eff",
            ]
            eff = eff[[c for c in keep if c in eff.columns]]
            df = df.merge(eff, on=["nballs", "dtheta_label", "seed"], how="left")

            for c, ce in [
                ("rho", "rho_eff"),
                ("area", "area_eff"),
                ("mean_reward", "mean_reward_eff"),
                ("cycle_length", "cycle_length_eff"),
            ]:
                if ce in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    df[ce] = pd.to_numeric(df[ce], errors="coerce")
                    df[c] = df[c].where(df[c].notna(), df[ce])
        except Exception as e:
            print(f"[warn] Could not merge efficiency table: {e}")

    return df


def load_howard_refs_from_npz():
    refs = {}

    for p in sorted(Path(PHASE4_HOWARD_ROOT).rglob("howard_summary_*ball.npz")):
        s = str(p)
        mN = re.search(r"howard_summary_(\d+)ball\.npz", p.name)
        md = re.search(r"dtheta_(pi\d+)", s)
        if not mN or not md:
            continue

        nballs = int(mN.group(1))
        dtheta_label = md.group(1)

        data = np.load(p, allow_pickle=True)
        rewards = np.asarray(data["rewards"], dtype=float).reshape(-1)
        cs = int(np.asarray(data["cycle_start"]).reshape(-1)[0])
        L = int(np.asarray(data["cycle_length"]).reshape(-1)[0])
        gain = float(np.mean(rewards[cs:cs + L]))

        refs[(nballs, dtheta_label)] = (gain, L)

    return refs


def load_howard_refs_from_mono(mono_df):
    refs = {}
    for (n, lab), g in mono_df.groupby(["nballs", "dtheta_label"]):
        gain = pd.to_numeric(g["howard_gain_from_summary"], errors="coerce").dropna()
        L = pd.to_numeric(g["howard_L_from_summary"], errors="coerce").dropna()
        if len(gain) and len(L):
            refs[(int(n), str(lab))] = (float(gain.iloc[0]), int(L.iloc[0]))
    return refs


def standardize_summary_df(df, arm, source=""):
    mean_col = first_existing_col(df, MEAN_CANDIDATES)
    if mean_col is None:
        raise KeyError(f"No mean-reward column found among {MEAN_CANDIDATES}")

    len_col = first_existing_col(df, LEN_CANDIDATES)
    rho_col = first_existing_col(df, RHO_CANDIDATES)
    area_col = first_existing_col(df, AREA_CANDIDATES)

    out = pd.DataFrame()
    out["nballs"] = pd.to_numeric(df["nballs"], errors="coerce")
    out["dtheta_label"] = df["dtheta_label"].astype(str)
    out["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    out["timesteps"] = pd.to_numeric(df["timesteps"], errors="coerce") if "timesteps" in df.columns else np.nan
    out["mean_reward"] = pd.to_numeric(df[mean_col], errors="coerce")
    out["cycle_length"] = pd.to_numeric(df[len_col], errors="coerce") if len_col else np.nan
    out["rho"] = pd.to_numeric(df[rho_col], errors="coerce") if rho_col else np.nan
    out["area"] = pd.to_numeric(df[area_col], errors="coerce") if area_col else np.nan
    out["noop_fraction"] = pd.to_numeric(df["noop_fraction"], errors="coerce") if "noop_fraction" in df.columns else np.nan
    out["arm"] = arm
    out["source"] = source

    out = out[
        out["nballs"].notna()
        & out["seed"].notna()
        & out["mean_reward"].notna()
    ].copy()

    out["nballs"] = out["nballs"].astype(int)
    out["seed"] = out["seed"].astype(int)

    return out


def load_factored():
    found = []
    seen = set()

    for root in FACTORED_SEARCH_ROOTS:
        for path in glob.glob(os.path.join(root, "**", "summary*.csv"), recursive=True):
            rp = os.path.realpath(path)
            if rp in seen:
                continue
            seen.add(rp)

            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            if "action_mode" not in df.columns:
                continue

            fac = df[df["action_mode"].astype(str) == "factored"].copy()
            if len(fac):
                found.append((path, fac))

    if not found:
        return None, []

    frames = []
    for path, f in found:
        frames.append(standardize_summary_df(f, "factored", source=path))

    combined = pd.concat(frames, ignore_index=True)

    combined = combined.sort_values("timesteps", na_position="first")
    combined = combined.drop_duplicates(
        subset=["nballs", "dtheta_label", "seed"],
        keep="last",
    ).reset_index(drop=True)

    return combined, [p for p, _ in found]


def add_recovery(df, refs):
    df = df.copy()
    gains = []
    Ls = []
    for _, r in df.iterrows():
        g, L = refs.get((int(r["nballs"]), str(r["dtheta_label"])), (np.nan, np.nan))
        gains.append(g)
        Ls.append(L)

    df["howard_gain"] = gains
    df["howard_L"] = Ls
    df["recovery_pct"] = 100.0 * df["mean_reward"] / df["howard_gain"]
    return df


def summarize(df):
    rows = []
    for (n, lab, arm), g in df.groupby(["nballs", "dtheta_label", "arm"]):
        rec = pd.to_numeric(g["recovery_pct"], errors="coerce").dropna()
        rho = pd.to_numeric(g["rho"], errors="coerce").dropna()
        noop = pd.to_numeric(g["noop_fraction"], errors="coerce").dropna()
        ts = pd.to_numeric(g["timesteps"], errors="coerce").dropna()
        lengths = sorted(set(int(x) for x in pd.to_numeric(g["cycle_length"], errors="coerce").dropna()))

        rows.append({
            "nballs": int(n),
            "dtheta": str(lab),
            "arm": str(arm),
            "howard_gain": float(pd.to_numeric(g["howard_gain"], errors="coerce").dropna().iloc[0])
                           if g["howard_gain"].notna().any() else np.nan,
            "howard_L": int(pd.to_numeric(g["howard_L"], errors="coerce").dropna().iloc[0])
                        if g["howard_L"].notna().any() else np.nan,
            "n_seeds": int(len(g)),
            "timesteps_median": float(np.median(ts)) if len(ts) else np.nan,
            "mean_reward_median": float(np.median(g["mean_reward"])),
            "mean_reward_min": float(np.min(g["mean_reward"])),
            "mean_reward_max": float(np.max(g["mean_reward"])),
            "recovery_median_pct": float(np.median(rec)) if len(rec) else np.nan,
            "recovery_min_pct": float(np.min(rec)) if len(rec) else np.nan,
            "recovery_max_pct": float(np.max(rec)) if len(rec) else np.nan,
            "rho_median": float(np.median(rho)) if len(rho) else np.nan,
            "noop_frac_median": float(np.median(noop)) if len(noop) else np.nan,
            "cycle_lengths": lengths,
        })

    return pd.DataFrame(rows).sort_values(["nballs", "dtheta", "arm"]).reset_index(drop=True)


def main():
    mono = load_phase4_monolithic()
    mono = mono[mono["nballs"].isin(N_VALUES)].copy()

    refs = load_howard_refs_from_npz()
    refs.update(load_howard_refs_from_mono(mono))

    fac, fac_paths = load_factored()

    if fac is not None:
        fac = fac[fac["nballs"].isin(N_VALUES)].copy()
        print(f"[factored] found {len(fac)} rows across {len(fac_paths)} file(s):")
        for p in fac_paths:
            print(f"    {p}")
        combined = pd.concat([mono, fac], ignore_index=True)
    else:
        print("[factored] none found yet")
        combined = mono

    combined = add_recovery(combined, refs)
    combined.to_csv("factored_vs_monolithic_vs_howard_by_seed.csv", index=False)

    table = summarize(combined)
    table.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 110)
    print("Recovery of Howard gain (%), by N x dtheta x arm")
    print("=" * 110)

    show_cols = [
        "nballs", "dtheta", "arm", "howard_gain", "howard_L", "n_seeds",
        "timesteps_median", "mean_reward_median",
        "recovery_median_pct", "recovery_min_pct", "recovery_max_pct",
        "rho_median", "noop_frac_median", "cycle_lengths",
    ]

    print(table[show_cols].to_string(
        index=False,
        float_format=lambda v: f"{v:.2f}" if abs(v) < 1e5 else f"{v:.1f}",
    ))

    print(f"\nwrote {OUTPUT_CSV}")
    print("wrote factored_vs_monolithic_vs_howard_by_seed.csv")


if __name__ == "__main__":
    main()
