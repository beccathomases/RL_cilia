#!/usr/bin/env python3
"""
merge_comparison.py
===================

Build the factored vs monolithic vs Howard recovery table at N=3,4.

It pulls three things together:
  * Howard benchmark (g, L)   -- from the phase-4 summary (howard_gain column)
  * monolithic PPO arm        -- from the phase-4 summary (ppo_avg_reward)
  * factored PPO arm          -- auto-discovered: any summary CSV whose
                                 action_mode column == "factored"

Recovery is recomputed identically for both PPO arms:
      recovery_% = 100 * mean_reward_per_step / howard_gain
where mean_reward_per_step is `ppo_avg_reward` (old format) or
`cycle_mean_reward` (new format) -- the script auto-detects which.

Run it from anywhere (paths below are absolute):
      python merge_comparison_phase6.py

Before interpreting the table, check that the factored paths printed by the
script point only to phase6_factored/results/ppo_sweeps_phase6, and that
n_seeds is the expected number for each completed case.

Outputs:
      ./factored_vs_monolithic_vs_howard.csv   (tidy long table)
      a printed summary
"""

import glob
import os
import numpy as np
import pandas as pd

# ============================== CONFIG =====================================
BASE = "/Users/bthomases/Documents/Student_Projects/RL_cilia/RL_cilia_python"

# Phase-4 master summary: has nballs, howard_gain, howard_cycle_length,
# ppo_avg_reward, tip_abs_area, reward_per_abs_tip_area for N2/3/4 x both grids.
MONO_OLD_CSV = os.path.join(
    BASE, "phase4_Nball_exploration",
    "results/ppo_sweeps_general/summary_all_with_tip_area.csv",
)

# Where to hunt for the factored sweep output (recursively). Add a root if you
# launched it somewhere else; auto-discovery keys off the action_mode column.
# Search only the Phase-6 sweep results by default. This avoids accidentally
# mixing smoke tests, old experiments, or duplicate summaries from other folders.
FACTORED_SEARCH_ROOTS = [
    os.path.join(BASE, "phase6_factored", "results", "ppo_sweeps_phase6"),
]

# Optional: if you ran a *matched* monolithic arm with the new sweep script
# (action_mode == "monolithic"), point here to use it instead of the phase-4
# numbers for stricter parity. Leave "" to use the phase-4 monolithic runs.
MONO_NEW_CSV = ""

N_VALUES = [3, 4]
OUTPUT_CSV = "factored_vs_monolithic_vs_howard.csv"
# ===========================================================================

MEAN_CANDIDATES = ["cycle_mean_reward", "ppo_avg_reward", "cycle_mean_reward_check"]
RHO_COL  = "reward_per_abs_tip_area"
AREA_COL = "tip_abs_area"
LEN_COL  = "cycle_length"


def _find_mean_col(df):
    for c in MEAN_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"no mean-reward column found among {MEAN_CANDIDATES}")


def standardize(df, arm):
    """Reduce any summary format to a common schema.

    Some hand-built / cached Phase-4 summary files can contain blank rows or
    aggregate rows. Coerce key columns to numeric and drop rows that do not
    correspond to actual seed runs before casting to int.
    """
    df = df.copy()

    if "nballs" not in df.columns:
        raise KeyError("expected 'nballs' column")
    if "dtheta_label" not in df.columns:
        raise KeyError("expected 'dtheta_label' column")

    mean_col = _find_mean_col(df)

    df["_nballs_num"] = pd.to_numeric(df["nballs"], errors="coerce")
    df["_mean_reward_num"] = pd.to_numeric(df[mean_col], errors="coerce")

    if "seed" in df.columns:
        df["_seed_num"] = pd.to_numeric(df["seed"], errors="coerce")
    else:
        df["_seed_num"] = np.arange(len(df), dtype=float)

    # Keep only real run rows.
    before = len(df)
    df = df[
        df["_nballs_num"].notna()
        & df["dtheta_label"].notna()
        & df["_seed_num"].notna()
        & df["_mean_reward_num"].notna()
    ].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[standardize:{arm}] dropped {dropped} non-run/blank row(s)")

    out = pd.DataFrame()
    out["nballs"] = df["_nballs_num"].astype(int)
    out["dtheta_label"] = df["dtheta_label"].astype(str)
    out["seed"] = df["_seed_num"].astype(int)

    if "timesteps" in df.columns:
        out["timesteps"] = pd.to_numeric(df["timesteps"], errors="coerce")
    else:
        out["timesteps"] = np.nan

    out["mean_reward"] = df["_mean_reward_num"].astype(float)
    out["cycle_length"] = (
        pd.to_numeric(df[LEN_COL], errors="coerce") if LEN_COL in df else np.nan
    )
    out["rho"] = (
        pd.to_numeric(df[RHO_COL], errors="coerce") if RHO_COL in df else np.nan
    )
    out["area"] = (
        pd.to_numeric(df[AREA_COL], errors="coerce") if AREA_COL in df else np.nan
    )
    out["noop_fraction"] = (
        pd.to_numeric(df["noop_fraction"], errors="coerce")
        if "noop_fraction" in df else np.nan
    )
    out["arm"] = arm
    return out


def load_monolithic_and_howard():
    if not os.path.isfile(MONO_OLD_CSV):
        raise FileNotFoundError(f"phase-4 summary not found:\n  {MONO_OLD_CSV}")
    df = pd.read_csv(MONO_OLD_CSV)
    if "nballs" not in df.columns:
        raise KeyError("expected 'nballs' column in the phase-4 master summary")

    # Howard reference (constant per (N, dtheta)). Coerce/drop blanks in case
    # the phase-4 cache contains aggregate or empty rows.
    df["_nballs_num"] = pd.to_numeric(df["nballs"], errors="coerce")
    df["_howard_gain_num"] = pd.to_numeric(df["howard_gain"], errors="coerce")
    df["_howard_L_num"] = pd.to_numeric(df["howard_cycle_length"], errors="coerce")
    hdf = df[
        df["_nballs_num"].notna()
        & df["dtheta_label"].notna()
        & df["_howard_gain_num"].notna()
        & df["_howard_L_num"].notna()
    ].copy()

    href = {}
    for (n, lab), g in hdf.groupby(["_nballs_num", "dtheta_label"]):
        href[(int(n), str(lab))] = (
            float(g["_howard_gain_num"].iloc[0]),
            int(g["_howard_L_num"].iloc[0]),
        )

    mono = standardize(df, "monolithic")
    return mono, href


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
        s = standardize(f, "factored")
        s["source_csv"] = path
        frames.append(s)

    combined = pd.concat(frames, ignore_index=True)

    # Per-case summary.csv and root summary_all.csv overlap. Also, smoke tests
    # and full runs may coexist. Sort by timesteps and keep the largest run per
    # (N, dtheta, seed). This makes 1M rows win over 300k smoke-test rows.
    combined = combined.sort_values("timesteps", na_position="first")
    combined = combined.drop_duplicates(
        subset=["nballs", "dtheta_label", "seed"],
        keep="last",
    ).reset_index(drop=True)

    return combined, [p for p, _ in found]

def add_recovery(df, href):
    g = df.apply(lambda r: href.get((int(r["nballs"]), r["dtheta_label"]),
                                    (np.nan, np.nan))[0], axis=1)
    df = df.copy()
    df["howard_gain"] = g.astype(float)
    df["recovery_pct"] = 100.0 * df["mean_reward"] / df["howard_gain"]
    return df


def summarize(df, href):
    rows = []
    for (n, lab), grp in df.groupby(["nballs", "dtheta_label"]):
        g, L = href.get((int(n), str(lab)), (np.nan, np.nan))
        for arm, a in grp.groupby("arm"):
            rec = a["recovery_pct"].to_numpy(float)
            lengths = sorted(set(int(x) for x in a["cycle_length"].dropna()))
            rows.append({
                "nballs": int(n), "dtheta": lab, "arm": arm,
                "howard_gain": g, "howard_L": L,
                "n_seeds": int(len(a)),
                "timesteps_median": (
                    float(np.nanmedian(a["timesteps"].to_numpy(float)))
                    if "timesteps" in a and a["timesteps"].notna().any()
                    else np.nan
                ),
                "recovery_median_pct": float(np.median(rec)),
                "recovery_min_pct": float(np.min(rec)),
                "recovery_max_pct": float(np.max(rec)),
                "rho_median": float(np.median(a["rho"].dropna())) if a["rho"].notna().any() else np.nan,
                "noop_frac_median": (float(np.median(a["noop_fraction"].dropna()))
                                     if a["noop_fraction"].notna().any() else np.nan),
                "cycle_lengths": lengths,
            })
    out = pd.DataFrame(rows).sort_values(["nballs", "dtheta", "arm"]).reset_index(drop=True)
    return out


def main():
    mono, href = load_monolithic_and_howard()
    mono = mono[mono["nballs"].isin(N_VALUES)]

    fac, fac_paths = load_factored()
    if fac is not None:
        fac = fac[fac["nballs"].isin(N_VALUES)]
        print(f"[factored] found {len(fac)} rows across {len(fac_paths)} file(s):")
        for p in fac_paths:
            print(f"    {p}")
        combined = pd.concat([mono, fac], ignore_index=True)
    else:
        print("[factored] none found yet (no summary CSV with action_mode='factored').")
        print("           Showing monolithic vs Howard only; re-run after the sweep.")
        combined = mono

    combined = add_recovery(combined, href)
    table = summarize(combined, href)
    table.to_csv(OUTPUT_CSV, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 90)
    print("Recovery of Howard gain (%), by N x dtheta x arm")
    print("=" * 90)
    show = table[["nballs", "dtheta", "arm", "howard_gain", "howard_L", "n_seeds",
                  "timesteps_median", "recovery_median_pct", "recovery_min_pct",
                  "recovery_max_pct", "rho_median", "noop_frac_median",
                  "cycle_lengths"]]
    print(show.to_string(index=False,
                         float_format=lambda v: f"{v:.2f}" if abs(v) < 1e4 else f"{v:.1f}"))
    print(f"\nwrote {OUTPUT_CSV}")
    print("\nNote: monolithic arm is the existing phase-4 run. If its PPO "
          "hyperparameters differ from the current script, re-run a matched "
          "monolithic sweep for strict parity (set MONO_NEW_CSV).")


if __name__ == "__main__":
    main()
