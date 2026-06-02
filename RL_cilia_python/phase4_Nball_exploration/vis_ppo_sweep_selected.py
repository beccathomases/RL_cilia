#!/usr/bin/env python3
"""
vis_ppo_sweep_selected.py
=========================

Visualize selected PPO sweep runs:
  - best percent-of-Howard run
  - median-ish percent-of-Howard run
  - longest-cycle outlier run

for each dtheta case in results/ppo_sweeps/summary_all.csv.

Run from phase4_Nball_exploration:

    python vis_ppo_sweep_selected.py

Outputs go to:

    figures/ppo_sweeps/n4_dtheta_pi20/seed_006_best/
    figures/ppo_sweeps/n4_dtheta_pi20/seed_003_median/
    ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------

def chain_positions_from_phi(phi):
    """
    Convert relative joint angles phi = [phi1, ..., phiN] to chain nodes.

    Physical segment angles:
        psi_k = sum_{j<=k} phi_j

    Segment length:
        L = 1/N
    """
    phi = np.asarray(phi, dtype=float)
    nballs = phi.size
    psi = np.cumsum(phi)
    L = 1.0 / nballs

    pts = [np.array([0.0, 0.0])]
    for k in range(nballs):
        prev = pts[-1]
        pts.append(prev + L * np.array([np.sin(psi[k]), np.cos(psi[k])]))

    return np.asarray(pts), psi


def all_chain_positions(angles):
    return np.asarray([chain_positions_from_phi(phi)[0] for phi in angles])


# ---------------------------------------------------------------------
# Loading PPO result.npy
# ---------------------------------------------------------------------

def load_ppo_result(path):
    path = Path(path)
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.shape == ():
        data = obj.item()
    elif isinstance(obj, np.ndarray) and obj.size == 1 and obj.dtype == object:
        data = obj.reshape(-1)[0]
    else:
        raise ValueError(f"Expected a saved dict in {path}, got array shape {obj.shape}")

    return data


def get_cycle_from_result(data):
    """
    Extract the detected cycle from a result dict produced by
    ppo_sweep_n4_compare_howard.py.
    """
    if "cycles" in data and data["cycles"] is not None and len(data["cycles"]) > 0:
        cyc = data["cycles"][0]
        if isinstance(cyc, np.ndarray) and cyc.shape == ():
            cyc = cyc.item()

        angles = np.asarray(cyc["angles"], dtype=float)
        rewards = np.asarray(cyc["rewards"], dtype=float)
        actions = np.asarray(cyc.get("actions", []), dtype=int)
        states = np.asarray(cyc.get("cycle", []), dtype=int)

        return angles, rewards, actions, states

    # Fallback using cycle_start / cycle_length in full arrays.
    angles_all = np.asarray(data["angles"], dtype=float)
    rewards_all = np.asarray(data["rewards"], dtype=float)
    actions_all = np.asarray(data.get("actions", []), dtype=int)
    states_all = np.asarray(data.get("states", []), dtype=int)

    i0 = int(data.get("cycle_start", -1))
    L = int(data.get("cycle_length", -1))
    if i0 < 0 or L <= 0:
        raise ValueError("No valid cycle found in result.")

    angles = angles_all[i0:i0 + L]
    rewards = rewards_all[i0:i0 + L]
    actions = actions_all[i0:i0 + L] if len(actions_all) >= i0 + L else np.array([])
    states = states_all[i0:i0 + L] if len(states_all) >= i0 + L else np.array([])

    return angles, rewards, actions, states


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _wall_and_limits(ax, all_pts):
    xs = all_pts[..., 0].ravel()
    zs = all_pts[..., 1].ravel()

    xrange = xs.max() - xs.min()
    zrange = zs.max() - zs.min()
    pad = 0.15 * max(xrange, zrange, 1e-9)

    ax.axhspan(-1, 0, color="0.88", zorder=0)
    ax.axhline(0, color="0.4", lw=1.2, zorder=1)

    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(-0.05, zs.max() + pad)
    ax.set_aspect("equal", adjustable="box")


def plot_stroke_overlay(angles, outpath, title):
    shapes = all_chain_positions(angles)
    n = len(shapes)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    _wall_and_limits(ax, shapes)

    for i, pts in enumerate(shapes):
        c = cmap(i / max(n - 1, 1))
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.8, alpha=0.85, zorder=3)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", color=c, ms=4, zorder=4)

    ax.plot(0, 0, "ks", ms=8, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("step in cycle")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_tip_path(angles, outpath, title):
    shapes = all_chain_positions(angles)
    tips = shapes[:, -1, :]

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    _wall_and_limits(ax, shapes)

    ax.plot(tips[:, 0], tips[:, 1], "-o", lw=1.8, ms=4)
    ax.plot(tips[0, 0], tips[0, 1], "s", ms=7, label="start")
    ax.plot(tips[-1, 0], tips[-1, 1], "^", ms=7, label="end")
    ax.plot(0, 0, "ks", ms=8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_joint_trajectories(angles, outpath, title, n_cycles=5):
    """
    PPO-style stacked angle plot:
      top: relative joint angles phi_k
      bottom: cumulative segment angles psi_k
    """
    angles = np.asarray(angles, dtype=float)
    nballs = angles.shape[1]
    L = len(angles)

    rep_phi = np.tile(angles, (n_cycles, 1))
    rep_phi_deg = np.degrees(rep_phi)

    psi = np.cumsum(angles, axis=1)
    rep_psi = np.tile(psi, (n_cycles, 1))
    rep_psi_deg = np.degrees(rep_psi)

    steps = np.arange(len(rep_phi_deg))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    for k in range(nballs):
        ax1.plot(steps, rep_phi_deg[:, k], lw=1.5, label=f"$\\phi_{{{k+1}}}$")

    for c in range(1, n_cycles):
        ax1.axvline(c * L - 0.5, color="0.8", lw=0.8, zorder=0)

    ax1.set_title(f"Relative joint angles ({n_cycles} cycles)")
    ax1.set_ylabel("degrees")
    ax1.legend(fontsize=8, ncol=max(1, nballs), loc="upper right")
    ax1.grid(alpha=0.3)

    for k in range(nballs):
        ax2.plot(steps, rep_psi_deg[:, k], lw=1.5, label=f"$\\psi_{{{k+1}}}$")

    for c in range(1, n_cycles):
        ax2.axvline(c * L - 0.5, color="0.8", lw=0.8, zorder=0)

    ax2.set_title("Cumulative segment angles")
    ax2.set_xlabel(f"step ({n_cycles} cycles of length {L})")
    ax2.set_ylabel("degrees")
    ax2.legend(fontsize=8, ncol=max(1, nballs), loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cycle_reward(rewards, outpath, title):
    rewards = np.asarray(rewards, dtype=float)
    steps = np.arange(len(rewards))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(steps, rewards, "-o", lw=1.8, ms=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(
        np.nanmean(rewards),
        color="k",
        ls="--",
        lw=1,
        label=f"mean = {np.nanmean(rewards):.6g}",
    )

    ax.set_xlabel("step in cycle")
    ax.set_ylabel("immediate reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def animate_beat(angles, outpath, title, fps=6):
    shapes = all_chain_positions(angles)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _wall_and_limits(ax, shapes)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    (line,) = ax.plot([], [], "-o", lw=2.5, ms=5)
    ax.plot([0], [0], "ks", ms=7)

    (trail,) = ax.plot([], [], ".", ms=3, alpha=0.6)
    tx, tz = [], []

    def update(frame):
        pts = shapes[frame % len(shapes)]
        line.set_data(pts[:, 0], pts[:, 1])

        tx.append(pts[-1, 0])
        tz.append(pts[-1, 1])
        if len(tx) > 2 * len(shapes):
            tx.pop(0)
            tz.pop(0)

        trail.set_data(tx, tz)
        return line, trail

    anim = FuncAnimation(
        fig,
        update,
        frames=len(shapes) * 2,
        interval=1000 / fps,
        blit=True,
    )
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------

def read_summary_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rr = dict(r)
            rr["seed"] = int(rr["seed"])
            rr["cycle_length"] = int(rr["cycle_length"])
            rr["ppo_avg_reward"] = float(rr["ppo_avg_reward"])
            rr["howard_gain"] = float(rr["howard_gain"])
            rr["howard_cycle_length"] = int(rr["howard_cycle_length"])
            rr["ppo_percent_of_howard"] = float(rr["ppo_percent_of_howard"])
            rows.append(rr)
    return rows


def select_runs(rows):
    """
    For each dtheta_label:
      - best: max percent of Howard
      - median: closest to median percent of Howard
      - long: longest cycle length; tie choose better percent
    """
    selected = []

    labels = sorted(set(r["dtheta_label"] for r in rows))
    for label in labels:
        rr = [r for r in rows if r["dtheta_label"] == label]

        best = max(rr, key=lambda r: r["ppo_percent_of_howard"])

        vals = np.array([r["ppo_percent_of_howard"] for r in rr], dtype=float)
        med_val = float(np.median(vals))
        median = min(rr, key=lambda r: abs(r["ppo_percent_of_howard"] - med_val))

        max_len = max(r["cycle_length"] for r in rr)
        long_candidates = [r for r in rr if r["cycle_length"] == max_len]
        long = max(long_candidates, key=lambda r: r["ppo_percent_of_howard"])

        selected.append(("best", best))
        selected.append(("median", median))
        selected.append(("longcycle", long))

    return selected


# ---------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------

def visualize_one(kind, row, out_root, n_cycles=5, make_animation=True):
    result_path = Path(row["result_path"])
    data = load_ppo_result(result_path)
    angles, rewards, actions, states = get_cycle_from_result(data)

    label = row["dtheta_label"]
    seed = int(row["seed"])
    L = int(row["cycle_length"])
    pct = float(row["ppo_percent_of_howard"])
    avg = float(row["ppo_avg_reward"])
    howard_gain = float(row["howard_gain"])
    howard_L = int(row["howard_cycle_length"])

    outdir = out_root / f"n4_dtheta_{label}" / f"seed_{seed:03d}_{kind}"
    outdir.mkdir(parents=True, exist_ok=True)

    title = (
        f"PPO N=4 {label} | seed={seed} | {kind} | "
        f"L={L}, avg={avg:.6g}, {pct:.2f}% Howard"
    )

    print(f"[viz] {label} seed={seed} {kind}: L={L}, avg={avg:.6g}, %Howard={pct:.2f}")
    print(f"      result: {result_path}")
    print(f"      outdir: {outdir}")

    plot_joint_trajectories(
        angles,
        outdir / "joint_trajectories.png",
        title,
        n_cycles=n_cycles,
    )

    plot_cycle_reward(
        rewards,
        outdir / "cycle_reward.png",
        "Per-step reward over selected PPO cycle\n" + title,
    )

    plot_stroke_overlay(
        angles,
        outdir / "stroke_overlay.png",
        "PPO stroke overlay\n" + title,
    )

    plot_tip_path(
        angles,
        outdir / "tip_path.png",
        "Tip path over selected PPO cycle\n" + title,
    )

    if make_animation:
        animate_beat(
            angles,
            outdir / "beat_animation.gif",
            "PPO beat\n" + title,
            fps=6,
        )

    meta = {
        "kind": kind,
        "dtheta_label": label,
        "seed": seed,
        "cycle_length": L,
        "ppo_avg_reward": avg,
        "howard_gain": howard_gain,
        "howard_cycle_length": howard_L,
        "ppo_percent_of_howard": pct,
        "source_result": str(result_path),
        "n_cycle_states": int(len(angles)),
        "n_cycle_rewards": int(len(rewards)),
    }

    with open(outdir / "viz_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/ppo_sweeps/summary_all.csv"),
        help="summary_all.csv from ppo_sweep_n4_compare_howard.py",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("figures/ppo_sweeps"),
        help="output root for selected-run figures",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="number of repeated cycles in angle trajectory plot",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip GIF animations",
    )
    args = parser.parse_args()

    rows = read_summary_csv(args.summary)
    selected = select_runs(rows)

    print("=" * 72)
    print("Selected runs:")
    for kind, row in selected:
        print(
            f"  {row['dtheta_label']:>4} {kind:>9}: "
            f"seed={row['seed']}, L={row['cycle_length']}, "
            f"avg={row['ppo_avg_reward']:.6g}, "
            f"%Howard={row['ppo_percent_of_howard']:.2f}"
        )

    print("=" * 72)
    outdirs = []
    for kind, row in selected:
        outdirs.append(
            visualize_one(
                kind,
                row,
                out_root=args.out_root,
                n_cycles=args.n_cycles,
                make_animation=not args.no_animation,
            )
        )

    print("=" * 72)
    print("[done] wrote selected PPO visualizations:")
    for d in outdirs:
        print(f"  {d}")


if __name__ == "__main__":
    main()