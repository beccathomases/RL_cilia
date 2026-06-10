#!/usr/bin/env python3
"""
primitive_period_detector.py

Post-hoc detection of the *primitive* stroke period and the pass multiplicity
for the discrete N-ball cilium PPO sweeps. Generalizes the Phase-4 "doubled
orbit" diagnostic (rho invariant, L ~ 2x) to k-fold commensurate closures, and
flags the case where exact-state closure no longer factors into k clean passes
("modulated/degraded").

Runs in two tiers:

  Tier 1 -- scalars only (needs just summary.csv). Builds a primitive reference
            from the single-pass seeds (smallest cycles) and classifies every
            seed by whether its length / area / path multiplicities agree.
            This reproduces the k_L vs k_area vs k_path agreement test.

  Tier 2 -- per-step arrays (needs result.npz with a reward trace and/or tip
            path). Estimates the primitive period from the autocorrelation of
            the per-step reward (primary) and the tip path (cross-check),
            refines the integer pass count, and reports an alignment score that
            distinguishes clean multi-pass from phase-drifting orbits.

Outputs (into OUTPUT_DIR):
    primitive_periods.csv        per-seed table with class + per-pass quantities
    primitive_periods_meta.json  config, primitive reference, provenance hashes
    multiplicity_scatter.png     k_area vs k_path classifier plot (always)
    autocorr_overlay.png         overlaid reward autocorrelations (Tier 2 only)

Dependencies: numpy, pandas, matplotlib. (No scipy required.)
"""

import os
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== CONFIG (edit me) =============================
# Path to the sweep summary written by your sweep driver.
SUMMARY_CSV = "results/ppo_sweeps_phase6_n12_t6e6_radscale0p4/N12_dtheta_pi20_factored/summary.csv"

# The summary's `result_path` entries are relative; BASE_DIR is prepended to
# them to locate each seed's result.npz. Set to the directory you launch from
# (usually the project root that contains `results/`). "." is the common case.
BASE_DIR = "."

# Where to write the analysis outputs.
OUTPUT_DIR = "results/ppo_sweeps_phase6_n12_t6e6_radscale0p4/N12_dtheta_pi20_factored/primitive_analysis"

#SUMMARY_CSV = "results/ppo_sweeps_n5/N5_dtheta_pi30/summary.csv"
#BASE_DIR = "."
#OUTPUT_DIR = "results/ppo_sweeps_n5/N5_dtheta_pi30/primitive_analysis"

# Column names in summary.csv (change only if your schema differs).
COL_SEED   = "seed"
COL_L      = "cycle_length"
COL_MEANR  = "cycle_mean_reward"
COL_TOTR   = "cycle_total_reward"
COL_AREA   = "tip_abs_area"
COL_PATH   = "tip_path_length"
COL_RHO    = "reward_per_abs_tip_area"
COL_RESULT = "result_path"

# result.npz array keys: the first candidate that exists in the file is used.
# If none are found the seed simply runs Tier 1 only. Adjust to match your
# visualize_cilia_run.py output if needed (the script prints the keys it sees).
REWARD_KEYS = ["cycle_rewards", "reward_trace", "rewards", "per_step_reward", "r"]
TIPXY_KEYS  = ["tip_xy", "tip_path", "tip", "tip_positions"]   # shape (L, 2)
TIPX_KEYS   = ["tip_x", "tipx"]
TIPZ_KEYS   = ["tip_z", "tipz"]

# Detection / classification knobs.
SINGLE_PASS_L_FACTOR = 1.5    # seeds with L <= factor * min(L) seed the primitive reference
MIN_PERIOD           = 5      # ignore autocorrelation peaks below this lag
AUTOCORR_PEAK_FRAC   = 0.80   # primitive period = first lag with autocorr >= frac * (in-window max)
K_AGREE_TOL          = 0.6    # |k_path - k_area| above this  => multiplicities disagree
PA_RATIO_OUTLIER     = 1.6    # path/area above this * family-median => geometric outlier
PRINT_NPZ_KEYS       = True   # print the keys of the first npz so you can fix REWARD_KEYS etc.
# ============================================================================


def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def resolve_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def load_reward_and_tip(npz_path):
    """Return (reward_trace, tip_xy) where either may be None if absent.

    If the file contains cycle_start and cycle_length, slice arrays to the
    detected cycle before running autocorrelation.
    """
    if not os.path.isfile(npz_path):
        return None, None, []

    with np.load(npz_path, allow_pickle=True) as npz:
        keys = list(npz.files)

        rk = resolve_key(npz, REWARD_KEYS)
        reward = np.asarray(npz[rk]).ravel().astype(float) if rk else None

        tip = None
        txy = resolve_key(npz, TIPXY_KEYS)
        if txy is not None:
            arr = np.asarray(npz[txy], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                tip = arr[:, :2]
            elif arr.ndim == 2 and arr.shape[0] == 2:
                tip = arr.T[:, :2]

        if tip is None:
            tx, tz = resolve_key(npz, TIPX_KEYS), resolve_key(npz, TIPZ_KEYS)
            if tx and tz:
                tip = np.column_stack([
                    np.asarray(npz[tx], float).ravel(),
                    np.asarray(npz[tz], float).ravel()
                ])

        # Slice to the detected cycle if cycle metadata is present.
        if "cycle_start" in npz.files and "cycle_length" in npz.files:
            cs = int(np.asarray(npz["cycle_start"]).reshape(-1)[0])
            L = int(np.asarray(npz["cycle_length"]).reshape(-1)[0])

            if reward is not None and cs + L <= len(reward):
                reward = reward[cs:cs + L]

            if tip is not None and cs + L <= len(tip):
                tip = tip[cs:cs + L]

    return reward, tip, keys

def normalized_autocorr(sig):
    sig = np.asarray(sig, float)
    sig = sig - sig.mean()
    n = len(sig)
    if n == 0 or not np.any(sig):
        return np.zeros(n)
    ac = np.correlate(sig, sig, mode="full")[n - 1:]
    if ac[0] == 0:
        return np.zeros(n)
    return ac / ac[0]


def detect_period(sig, min_period=MIN_PERIOD, peak_frac=AUTOCORR_PEAK_FRAC):
    """
    Smallest lag p >= min_period that is a local maximum of the autocorrelation
    and at least peak_frac of the in-window maximum. Returns (p, strength).
    If no internal period is found, returns (len(sig), 0.0) -> single pass.
    """
    n = len(sig)
    if n < 2 * min_period:
        return n, 0.0
    ac = normalized_autocorr(sig)
    hi = n // 2
    lags = np.arange(min_period, hi)
    seg = ac[min_period:hi]
    if seg.size == 0:
        return n, 0.0
    thr = peak_frac * seg.max()
    for i, lag in enumerate(lags):
        left  = seg[i] >= seg[i - 1] if i > 0 else True
        right = seg[i] >= seg[i + 1] if i < len(seg) - 1 else True
        if seg[i] >= thr and left and right:
            return int(lag), float(ac[lag])
    return n, float(seg.max())


def alignment_score(sig, p):
    """Mean correlation between consecutive period-length windows.
    High (->1) = clean repeats; lower = phase drift / modulation."""
    n = len(sig)
    k = n // p
    if k < 2 or p < 2:
        return np.nan
    wins = [sig[i * p:(i + 1) * p] for i in range(k)]
    cs = []
    for a, b in zip(wins[:-1], wins[1:]):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        if d > 0:
            cs.append(float((a @ b) / d))
    return float(np.mean(cs)) if cs else np.nan


def iround(x):
    """Round half away from zero (avoids numpy/python round-half-to-even)."""
    return int(np.floor(x + 0.5)) if np.isfinite(x) else 1


def classify(k_primary, k_area, k_path, pa_ratio, fam_pa):
    k_round = iround(k_primary) if np.isfinite(k_primary) else 1
    pa_out = pa_ratio > PA_RATIO_OUTLIER * fam_pa
    disagree = (np.isfinite(k_area) and np.isfinite(k_path)
                and abs(k_path - k_area) > K_AGREE_TOL)
    if k_round <= 1 and not pa_out:
        return "single-pass"
    if pa_out or disagree:
        return "modulated/degraded"
    return f"commensurate-{k_round}x"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV).sort_values(COL_SEED).reset_index(drop=True)

    L    = df[COL_L].to_numpy(float)
    area = df[COL_AREA].to_numpy(float)
    path = df[COL_PATH].to_numpy(float)

    # ----- Tier 1: primitive reference from the single-pass (shortest) seeds --
    lmin = L.min()
    sp_mask = L <= SINGLE_PASS_L_FACTOR * lmin
    L0    = float(np.median(L[sp_mask]))
    area0 = float(np.median(area[sp_mask]))
    path0 = float(np.median(path[sp_mask]))
    pa_ratio = path / area
    fam_pa = float(np.median(pa_ratio[sp_mask]))

    df["pa_ratio"] = pa_ratio
    df["k_L"]      = L / L0
    df["k_area"]   = area / area0
    df["k_path"]   = path / path0

    # ----- Tier 2: per-step autocorrelation period detection ------------------
    p_reward, p_tip, k_sig, align, tier2_seen, first_keys = [], [], [], [], [], None
    autocorr_curves = {}
    for _, row in df.iterrows():
        npz_path = os.path.join(BASE_DIR, str(row[COL_RESULT]))
        reward, tip, keys = load_reward_and_tip(npz_path)
        if first_keys is None and keys:
            first_keys = keys
            if PRINT_NPZ_KEYS:
                print(f"[npz keys in {npz_path}]: {keys}")

        pr = pt = np.nan
        a = np.nan
        if reward is not None and len(reward) >= 2 * MIN_PERIOD:
            pr, _ = detect_period(reward)
            a = alignment_score(reward, pr)
            autocorr_curves[int(row[COL_SEED])] = (len(reward), normalized_autocorr(reward))
            tier2_seen.append(True)
        else:
            tier2_seen.append(False)
        if tip is not None and len(tip) >= 2 * MIN_PERIOD:
            speed = np.r_[0.0, np.linalg.norm(np.diff(tip, axis=0), axis=1)]
            pt, _ = detect_period(speed)

        p_reward.append(pr)
        p_tip.append(pt)
        k_sig.append(row[COL_L] / pr if np.isfinite(pr) and pr > 0 else np.nan)
        align.append(a)

    df["p_reward"]   = p_reward
    df["p_tip"]      = p_tip
    df["k_signal"]   = k_sig
    df["alignment"]  = align
    have_tier2 = any(tier2_seen)

    # primary multiplicity:
    # Use scalar multiplicity as the default, because L/area/path scaling is the
    # actual k-fold traversal diagnostic. Reward autocorrelation can detect useful
    # internal structure, but it may also find within-stroke sub-bursts, especially
    # for single-pass strokes. Therefore only use k_signal when it agrees with the
    # scalar multiplicity.
    scalar_k = df[["k_L", "k_area", "k_path"]].median(axis=1)
    df["k_scalar"] = scalar_k

    def choose_k_primary(row):
        ks = row["k_signal"]
        kc = row["k_scalar"]

        if not np.isfinite(ks):
            return kc

        # Accept the signal-based multiplicity only if it agrees with the
        # length/area/path multiplicity.
        if abs(ks - kc) <= K_AGREE_TOL:
            return ks

        if iround(ks) == iround(kc):
            return ks

        # Otherwise treat the reward autocorrelation as an internal reward
        # subperiod, not as a pass count.
        return kc

    df["k_primary"] = df.apply(choose_k_primary, axis=1)

    # ----- classify + per-pass (intensive) quantities -------------------------
    df["class"] = [
        classify(r.k_primary, r.k_area, r.k_path, r.pa_ratio, fam_pa)
        for r in df.itertuples()
    ]
    kp = df["k_primary"].apply(iround).clip(lower=1)
    df["L_primitive"]    = (df[COL_L] / kp).round(1)
    df["area_per_pass"]  = df[COL_AREA] / kp
    df["path_per_pass"]  = df[COL_PATH] / kp
    df["reward_per_pass"] = df[COL_TOTR] / kp

    # ----- write table --------------------------------------------------------
    out_cols = [COL_SEED, COL_L, "L_primitive", "k_primary", "k_L", "k_area",
                "k_path", "p_reward", "p_tip", "alignment", COL_MEANR, COL_RHO,
                "pa_ratio", "area_per_pass", "path_per_pass", "reward_per_pass",
                "class"]
    table = df[out_cols].copy()
    csv_path = os.path.join(OUTPUT_DIR, "primitive_periods.csv")
    table.to_csv(csv_path, index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {csv_path}")

    # ----- metadata sidecar ---------------------------------------------------
    meta = {
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "summary_csv": os.path.abspath(SUMMARY_CSV),
        "summary_csv_sha256": sha256(SUMMARY_CSV),
        "tier2_active": bool(have_tier2),
        "npz_keys_first_seed": first_keys,
        "config": {
            "SINGLE_PASS_L_FACTOR": SINGLE_PASS_L_FACTOR,
            "MIN_PERIOD": MIN_PERIOD,
            "AUTOCORR_PEAK_FRAC": AUTOCORR_PEAK_FRAC,
            "K_AGREE_TOL": K_AGREE_TOL,
            "PA_RATIO_OUTLIER": PA_RATIO_OUTLIER,
        },
        "primitive_reference": {
            "single_pass_seeds": [int(s) for s in df[COL_SEED][sp_mask]],
            "L0": L0, "area0": area0, "path0": path0, "family_pa_ratio": fam_pa,
        },
        "class_counts": table["class"].value_counts().to_dict(),
    }
    meta_path = os.path.join(OUTPUT_DIR, "primitive_periods_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    # ----- classifier scatter (always) ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lim = float(np.nanmax([df["k_area"].max(), df["k_path"].max()])) + 0.6
    ax.plot([0, lim], [0, lim], "--", color="0.6", lw=1, zorder=0,
            label="clean multi-pass (k agree)")
    colors = {"single-pass": "tab:blue"}
    for _, r in df.iterrows():
        cls = r["class"]
        c = ("tab:blue" if cls == "single-pass"
             else "tab:red" if cls == "modulated/degraded" else "tab:green")
        ax.scatter(r["k_area"], r["k_path"], s=70, color=c, zorder=3,
                   edgecolor="k", linewidth=0.4)
        ax.annotate(f"s{int(r[COL_SEED])}", (r["k_area"], r["k_path"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="k",
                          label=lbl)
               for lbl, c in [("single-pass", "tab:blue"),
                              ("commensurate kx", "tab:green"),
                              ("modulated/degraded", "tab:red")]]
    handles.append(plt.Line2D([], [], ls="--", color="0.6", label="k agree (y=x)"))
    ax.set_xlabel(r"$k_{\rm area}$  (area / primitive area)")
    ax.set_ylabel(r"$k_{\rm path}$  (path / primitive path)")
    ax.set_title("Pass multiplicity: clean multi-pass lies on $y=x$;\n"
                 "off-diagonal = extra traversals not enclosing fresh area")
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    fig.tight_layout()
    sc_path = os.path.join(OUTPUT_DIR, "multiplicity_scatter.png")
    fig.savefig(sc_path, dpi=140); plt.close(fig)
    print(f"wrote {sc_path}")

    # ----- overlaid reward autocorrelations (Tier 2 only) ---------------------
    if autocorr_curves:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        cmap = plt.cm.viridis
        seeds = sorted(autocorr_curves)
        for i, s in enumerate(seeds):
            n, ac = autocorr_curves[s]
            hi = n // 2
            ax.plot(np.arange(hi), ac[:hi], lw=1.3,
                    color=cmap(i / max(1, len(seeds) - 1)), label=f"seed {s}")
            prow = df.loc[df[COL_SEED] == s].iloc[0]
            if np.isfinite(prow["p_reward"]) and prow["p_reward"] < hi:
                p = int(prow["p_reward"])
                ax.scatter([p], [ac[p]], color=cmap(i / max(1, len(seeds) - 1)),
                           s=40, zorder=5, edgecolor="k", linewidth=0.4)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("lag (steps)")
        ax.set_ylabel("normalized reward autocorrelation")
        ax.set_title("Reward autocorrelation per seed; dots mark the detected "
                     "primitive period")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        ac_path = os.path.join(OUTPUT_DIR, "autocorr_overlay.png")
        fig.savefig(ac_path, dpi=140); plt.close(fig)
        print(f"wrote {ac_path}")
    else:
        print("Tier 2 inactive (no reward trace found in result.npz) -- "
              "scalar classification only. Set REWARD_KEYS to your array name.")


if __name__ == "__main__":
    main()
