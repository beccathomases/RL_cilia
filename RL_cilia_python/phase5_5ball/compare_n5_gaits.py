#!/usr/bin/env python3
"""
compare_n5_gaits.py
===================

Compare learned N=5 PPO gaits across seeds.

This script loads result.npz files from the N=5 PPO sweep, extracts the detected
cycle from each seed, resamples each cycle to a common phase grid, phase-aligns
pairs by circular shift, and computes pairwise similarity metrics.

Inputs:
    results/ppo_sweeps_n5/N5_dtheta_pi20/seed_XXX/result.npz
    results/ppo_sweeps_n5/summary_all.csv

Outputs:
    figures/ppo_sweeps_n5/gait_comparison_pi20/
        pairwise_tip_rms.png
        pairwise_psi_rms.png
        pairwise_reward_corr.png
        aligned_tip_paths.png
        aligned_reward_traces.png

    results/ppo_sweeps_n5/gait_comparison_pi20/
        pairwise_metrics.csv
        pairwise_tip_rms.csv
        pairwise_phi_rms.csv
        pairwise_psi_rms.csv
        pairwise_reward_corr.csv

Example:
    python compare_n5_gaits.py
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_ROOT = Path("results/ppo_sweeps_n5")
DEFAULT_CASE = "pi30"
DEFAULT_NPHASE = 200


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_cycle_npz(path):
    data = np.load(path, allow_pickle=True)

    phi_all = np.asarray(data["phi"], dtype=float)
    tip_all = np.asarray(data["tip"], dtype=float)
    rewards_all = np.asarray(data["rewards"], dtype=float)

    cycle_start = int(np.asarray(data["cycle_start"]).item())
    cycle_length = int(np.asarray(data["cycle_length"]).item())

    i0 = cycle_start
    i1 = cycle_start + cycle_length

    phi = phi_all[i0:i1]
    tip = tip_all[i0:i1]
    rewards = rewards_all[i0:i1]

    psi = np.cumsum(phi, axis=1)

    return {
        "phi": phi,
        "psi": psi,
        "tip": tip,
        "rewards": rewards,
        "cycle_start": cycle_start,
        "cycle_length": cycle_length,
        "path": str(path),
    }


def load_summary(root, case):
    summary_path = root / "summary_all.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)
    df = df[(df["nballs"] == 5) & (df["dtheta_label"] == case)].copy()

    if df.empty:
        raise ValueError(f"No N=5 rows found for case={case} in {summary_path}")

    # Use largest training horizon present, in case old smoke tests are still around.
    max_timesteps = df["timesteps"].max()
    df = df[df["timesteps"] == max_timesteps].copy()

    df = df.sort_values("seed").reset_index(drop=True)
    return df, int(max_timesteps)


def load_all_cycles(root, case):
    df, max_timesteps = load_summary(root, case)

    cycles = {}
    case_dir = root / f"N5_dtheta_{case}"

    for _, row in df.iterrows():
        seed = int(row["seed"])
        result_path = case_dir / f"seed_{seed:03d}" / "result.npz"
        if not result_path.exists():
            print(f"[warn] missing {result_path}; skipping seed {seed}")
            continue

        cyc = load_cycle_npz(result_path)
        cyc["seed"] = seed
        cyc["summary"] = row.to_dict()
        cycles[seed] = cyc

    if not cycles:
        raise RuntimeError("No cycles loaded.")

    return cycles, df, max_timesteps


# ---------------------------------------------------------------------
# Resampling and alignment
# ---------------------------------------------------------------------

def periodic_resample_array(X, nphase):
    """
    Resample a periodic sequence X[t,...] onto nphase equally spaced phase points.
    Linear interpolation with periodic closure.
    """
    X = np.asarray(X, dtype=float)
    L = X.shape[0]

    if L == 1:
        return np.repeat(X, nphase, axis=0)

    # Original phase points 0, 1/L, ..., (L-1)/L plus closure at 1.
    old_phase = np.arange(L + 1) / L
    new_phase = np.arange(nphase) / nphase

    Xclosed = np.concatenate([X, X[:1]], axis=0)

    flat = Xclosed.reshape(L + 1, -1)
    out = np.zeros((nphase, flat.shape[1]))

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new_phase, old_phase, flat[:, j])

    return out.reshape((nphase,) + X.shape[1:])


def resample_cycle(cyc, nphase):
    return {
        **cyc,
        "phi_r": periodic_resample_array(cyc["phi"], nphase),
        "psi_r": periodic_resample_array(cyc["psi"], nphase),
        "tip_r": periodic_resample_array(cyc["tip"], nphase),
        "rewards_r": periodic_resample_array(cyc["rewards"][:, None], nphase)[:, 0],
    }


def circular_shift(X, shift):
    return np.roll(X, shift=shift, axis=0)


def rms(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return float(np.sqrt(np.mean((A - B) ** 2)))


def corr(A, B):
    a = np.asarray(A, dtype=float).reshape(-1)
    b = np.asarray(B, dtype=float).reshape(-1)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-14:
        return np.nan

    return float(np.dot(a, b) / denom)


def best_shift_by_tip(ref_tip, other_tip):
    """
    Choose circular phase shift minimizing tip-path RMS.
    """
    n = ref_tip.shape[0]
    best_s = 0
    best_d = np.inf

    for s in range(n):
        d = rms(ref_tip, circular_shift(other_tip, s))
        if d < best_d:
            best_d = d
            best_s = s

    return best_s, best_d


def align_to_reference(cycles_r, ref_seed):
    """
    Align all cycles to a reference seed using tip path RMS.
    """
    ref = cycles_r[ref_seed]
    aligned = {}

    for seed, cyc in cycles_r.items():
        if seed == ref_seed:
            shift = 0
        else:
            shift, _ = best_shift_by_tip(ref["tip_r"], cyc["tip_r"])

        out = dict(cyc)
        for key in ["phi_r", "psi_r", "tip_r", "rewards_r"]:
            out[key] = circular_shift(cyc[key], shift)

        out["align_shift_to_ref"] = int(shift)
        aligned[seed] = out

    return aligned


# ---------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------

def compute_pairwise_metrics(cycles_r):
    seeds = sorted(cycles_r)

    rows = []

    for i, si in enumerate(seeds):
        ci = cycles_r[si]

        for sj in seeds:
            cj = cycles_r[sj]

            shift, tip_rms_val = best_shift_by_tip(ci["tip_r"], cj["tip_r"])

            phi_j = circular_shift(cj["phi_r"], shift)
            psi_j = circular_shift(cj["psi_r"], shift)
            tip_j = circular_shift(cj["tip_r"], shift)
            rew_j = circular_shift(cj["rewards_r"], shift)

            row = {
                "seed_i": si,
                "seed_j": sj,
                "best_shift": int(shift),
                "tip_rms": rms(ci["tip_r"], tip_j),
                "phi_rms_rad": rms(ci["phi_r"], phi_j),
                "psi_rms_rad": rms(ci["psi_r"], psi_j),
                "phi_rms_deg": np.degrees(rms(ci["phi_r"], phi_j)),
                "psi_rms_deg": np.degrees(rms(ci["psi_r"], psi_j)),
                "reward_rms": rms(ci["rewards_r"], rew_j),
                "reward_corr": corr(ci["rewards_r"], rew_j),
                "tip_corr_x": corr(ci["tip_r"][:, 0], tip_j[:, 0]),
                "tip_corr_z": corr(ci["tip_r"][:, 1], tip_j[:, 1]),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def metric_matrix(pairwise, seeds, metric):
    M = np.zeros((len(seeds), len(seeds)), dtype=float)

    for a, si in enumerate(seeds):
        for b, sj in enumerate(seeds):
            row = pairwise[(pairwise["seed_i"] == si) & (pairwise["seed_j"] == sj)]
            M[a, b] = float(row.iloc[0][metric])

    return M


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_matrix(M, seeds, outpath, title, cbar_label, cmap=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.5))

    im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(seeds)))
    ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_yticklabels(seeds)
    ax.set_xlabel("seed j")
    ax.set_ylabel("seed i")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"[wrote] {outpath}")


def plot_aligned_tip_paths(aligned, outpath, title):
    fig, ax = plt.subplots(figsize=(6.3, 5.6))

    for seed, cyc in sorted(aligned.items()):
        tip = cyc["tip_r"]
        closed = np.vstack([tip, tip[0]])
        ax.plot(closed[:, 0], closed[:, 1], lw=1.5, label=f"seed {seed}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip z")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"[wrote] {outpath}")


def plot_aligned_rewards(aligned, outpath, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    phase = np.linspace(0, 1, next(iter(aligned.values()))["rewards_r"].shape[0], endpoint=False)

    for seed, cyc in sorted(aligned.items()):
        ax.plot(phase, cyc["rewards_r"], lw=1.3, label=f"seed {seed}")

    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("phase")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"[wrote] {outpath}")


def plot_aligned_psi_mean(aligned, outpath, title):
    """
    Plot cumulative angle waveforms for all seeds, one panel per segment angle.
    """
    seeds = sorted(aligned)
    N = aligned[seeds[0]]["psi_r"].shape[1]
    phase = np.linspace(0, 1, aligned[seeds[0]]["psi_r"].shape[0], endpoint=False)

    fig, axes = plt.subplots(N, 1, figsize=(8.2, 1.75 * N), sharex=True)

    if N == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        for seed in seeds:
            ax.plot(phase, np.degrees(aligned[seed]["psi_r"][:, k]), lw=1.1, alpha=0.85)
        ax.set_ylabel(rf"$\psi_{k+1}$ deg")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("phase")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"[wrote] {outpath}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--nphase", type=int, default=DEFAULT_NPHASE)
    parser.add_argument("--ref-seed", type=int, default=None)
    args = parser.parse_args()

    cycles, summary, max_timesteps = load_all_cycles(args.root, args.case)

    seeds = sorted(cycles)
    print(f"[load] seeds={seeds}")
    print(f"[load] max_timesteps={max_timesteps}")

    cycles_r = {s: resample_cycle(c, args.nphase) for s, c in cycles.items()}

    if args.ref_seed is None:
        # Choose median seed by cycle mean reward if available.
        med_seed = int(summary.sort_values("cycle_mean_reward").iloc[len(summary)//2]["seed"])
        ref_seed = med_seed if med_seed in cycles_r else seeds[0]
    else:
        ref_seed = args.ref_seed

    print(f"[align] reference seed={ref_seed}")

    aligned = align_to_reference(cycles_r, ref_seed=ref_seed)

    out_results = args.root / f"gait_comparison_{args.case}"
    out_figs = Path("figures") / "ppo_sweeps_n5" / f"gait_comparison_{args.case}"

    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # Pairwise comparison uses best pairwise shift, independent of reference.
    pairwise = compute_pairwise_metrics(cycles_r)
    pairwise_path = out_results / "pairwise_metrics.csv"
    pairwise.to_csv(pairwise_path, index=False)
    print(f"[wrote] {pairwise_path}")

    # Save individual matrices.
    for metric in [
        "tip_rms",
        "phi_rms_deg",
        "psi_rms_deg",
        "reward_rms",
        "reward_corr",
    ]:
        M = metric_matrix(pairwise, seeds, metric)
        pd.DataFrame(M, index=seeds, columns=seeds).to_csv(out_results / f"{metric}.csv")

    # Plot heatmaps.
    plot_matrix(
        metric_matrix(pairwise, seeds, "tip_rms"),
        seeds,
        out_figs / "pairwise_tip_rms.png",
        title="N=5 seed-to-seed tip-path RMS after phase alignment",
        cbar_label="tip RMS",
    )

    plot_matrix(
        metric_matrix(pairwise, seeds, "psi_rms_deg"),
        seeds,
        out_figs / "pairwise_psi_rms_deg.png",
        title="N=5 seed-to-seed cumulative-angle RMS after phase alignment",
        cbar_label="psi RMS (degrees)",
    )

    plot_matrix(
        metric_matrix(pairwise, seeds, "reward_corr"),
        seeds,
        out_figs / "pairwise_reward_corr.png",
        title="N=5 seed-to-seed reward correlation after phase alignment",
        cbar_label="reward correlation",
        cmap="viridis",
        vmin=-1,
        vmax=1,
    )

    # Overlay/reference-aligned plots.
    plot_aligned_tip_paths(
        aligned,
        out_figs / "aligned_tip_paths.png",
        title=f"N=5 aligned tip paths, {args.case}, ref seed {ref_seed}",
    )

    plot_aligned_rewards(
        aligned,
        out_figs / "aligned_reward_traces.png",
        title=f"N=5 aligned reward traces, {args.case}, ref seed {ref_seed}",
    )

    plot_aligned_psi_mean(
        aligned,
        out_figs / "aligned_cumulative_angles.png",
        title=f"N=5 aligned cumulative angle waveforms, {args.case}, ref seed {ref_seed}",
    )

    # Small JSON metadata.
    meta = {
        "root": str(args.root),
        "case": args.case,
        "nphase": args.nphase,
        "seeds": seeds,
        "max_timesteps": max_timesteps,
        "ref_seed": ref_seed,
    }

    with open(out_results / "gait_comparison_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[done]")
    print(f"Results: {out_results}")
    print(f"Figures: {out_figs}")


if __name__ == "__main__":
    main()