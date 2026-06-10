#!/usr/bin/env python3
"""
visualize_run.py
================

Generic visualizer for the current PPO result.npz format used in Phase 5/6.

Reads files like:
    results/ppo_sweeps_phase6/N3_dtheta_pi20_factored/seed_004/result.npz

Expected keys:
    phi, tip, rewards, cycle_start, cycle_length

Outputs:
    stroke_overlay.png
    tip_path.png
    joint_trajectories.png
    cycle_reward.png
    beat_animation.gif  (optional)
    viz_metadata.json
"""

from pathlib import Path
import argparse
import json
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_npz_result(path):
    data = np.load(path, allow_pickle=True)

    required = ["phi", "tip", "rewards", "cycle_start", "cycle_length"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required key(s): {missing}. Found {data.files}")

    phi_all = np.asarray(data["phi"], dtype=float)
    tip_all = np.asarray(data["tip"], dtype=float)
    rewards_all = np.asarray(data["rewards"], dtype=float).reshape(-1)

    cycle_start = int(np.asarray(data["cycle_start"]).reshape(-1)[0])
    cycle_length = int(np.asarray(data["cycle_length"]).reshape(-1)[0])

    i0 = cycle_start
    i1 = cycle_start + cycle_length

    phi = phi_all[i0:i1]
    tip = tip_all[i0:i1]
    rewards = rewards_all[i0:i1]

    if len(phi) != cycle_length or len(tip) != cycle_length or len(rewards) != cycle_length:
        raise ValueError(
            f"Bad cycle slice: len(phi)={len(phi)}, len(tip)={len(tip)}, "
            f"len(rewards)={len(rewards)}, cycle_length={cycle_length}, cycle_start={cycle_start}"
        )

    return {
        "phi": phi,
        "tip": tip,
        "rewards": rewards,
        "cycle_start": cycle_start,
        "cycle_length": cycle_length,
        "phi_all": phi_all,
        "tip_all": tip_all,
        "rewards_all": rewards_all,
        "keys": list(data.files),
    }


def chain_positions_from_phi(phi, total_length=1.0):
    phi = np.asarray(phi, dtype=float)
    N = len(phi)
    seg_len = total_length / N

    psi = np.cumsum(phi)

    pts = [np.array([0.0, 0.0])]
    for ang in psi:
        prev = pts[-1]
        pts.append(prev + seg_len * np.array([np.sin(ang), np.cos(ang)]))

    return np.asarray(pts), psi


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


def wall_and_limits(ax, all_pts):
    xs = all_pts[..., 0].ravel()
    zs = all_pts[..., 1].ravel()

    xspan = xs.max() - xs.min()
    zspan = zs.max() - zs.min()

    xpad = 0.15 * (xspan + 1e-9)
    zpad = 0.10 * (zspan + 1e-9)

    ax.axhspan(-1, 0, color="0.85", zorder=0)
    ax.axhline(0, color="0.4", lw=1.2, zorder=1)

    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(-0.05, zs.max() + zpad + 0.05)
    ax.set_aspect("equal", adjustable="box")


def plot_stroke_overlay(phi, outpath, title):
    shapes = [chain_positions_from_phi(p)[0] for p in phi]
    all_pts = np.asarray(shapes)
    n = len(shapes)

    fig, ax = plt.subplots(figsize=(6.2, 6))
    wall_and_limits(ax, all_pts)

    cmap = plt.cm.viridis
    for i, pts in enumerate(shapes):
        c = cmap(i / max(n - 1, 1))
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.8, alpha=0.85)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", color=c, ms=4)

    ax.plot(0, 0, "ks", ms=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("step in cycle")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_tip_path(tip, outpath, title):
    tip = np.asarray(tip, dtype=float)
    A = closed_polygon_area(tip)
    plen = path_length(tip, close=True)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    closed = np.vstack([tip, tip[0]])
    ax.plot(closed[:, 0], closed[:, 1], "-o", lw=1.8, ms=4)
    ax.plot(tip[0, 0], tip[0, 1], "s", ms=7, label="start")
    ax.plot(tip[-1, 0], tip[-1, 1], "x", ms=8, label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip z")
    ax.set_title(title + f"\narea={abs(A):.4f}, path={plen:.4f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_joint_trajectories(phi, outpath, title, n_cycles=5):
    phi = np.asarray(phi, dtype=float)
    L = len(phi)
    rep = np.tile(phi, (n_cycles, 1))

    phis_deg = np.degrees(rep)
    psis_deg = np.degrees(np.cumsum(rep, axis=1))

    steps = np.arange(len(rep))
    N = phi.shape[1]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for k in range(N):
        a1.plot(steps, phis_deg[:, k], "-", lw=1.5, label=f"$\\phi_{{{k+1}}}$")
        a2.plot(steps, psis_deg[:, k], "-", lw=1.5, label=f"$\\psi_{{{k+1}}}$")

    for c in range(1, n_cycles):
        for ax in (a1, a2):
            ax.axvline(c * L - 0.5, color="0.8", lw=0.8)

    a1.set_title(f"relative joint angles ({n_cycles} cycles)")
    a2.set_title("cumulative segment angles")
    a2.set_xlabel(f"step ({n_cycles} cycles of length {L})")

    for ax in (a1, a2):
        ax.set_ylabel("degrees")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=N, loc="upper right")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cycle_reward(rewards, outpath, title):
    rewards = np.asarray(rewards, dtype=float)
    steps = np.arange(len(rewards))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(steps, rewards, "-o", lw=1.8, ms=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(rewards.mean(), color="k", ls="--", lw=1, label=f"mean = {rewards.mean():.3f}")
    ax.set_xlabel("step in cycle")
    ax.set_ylabel("immediate reward / flux")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def animate_beat(phi, outpath, title, fps=6):
    shapes = [chain_positions_from_phi(p)[0] for p in phi]
    all_pts = np.asarray(shapes)

    fig, ax = plt.subplots(figsize=(5, 5))
    wall_and_limits(ax, all_pts)

    ax.set_title(title, fontsize=11)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Path to result.npz")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--ncycles", type=int, default=5)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--fps", type=int, default=6)

    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    data = load_npz_result(result_path)

    phi = data["phi"]
    tip = data["tip"]
    rewards = data["rewards"]
    L = data["cycle_length"]
    N = phi.shape[1]

    seed_name = result_path.parent.name
    case_name = result_path.parent.parent.name

    if args.outdir is None:
        outdir = Path("figures") / case_name / seed_name
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    A = closed_polygon_area(tip)
    plen = path_length(tip, close=True)
    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards))
    rho = total_reward / abs(A) if abs(A) > 1e-12 else np.nan

    tag = (
        f"{case_name}/{seed_name} | "
        f"N={N}, L={L}, mean={mean_reward:.3f}, "
        f"area={abs(A):.3f}, rho={rho:.2f}"
    )

    print(f"[viz] {tag}")
    print(f"[viz] keys={data['keys']}")
    print(f"[viz] writing to {outdir}")

    plot_stroke_overlay(
        phi,
        outdir / "stroke_overlay.png",
        "Learned beat: stroke overlay\n" + tag,
    )

    plot_tip_path(
        tip,
        outdir / "tip_path.png",
        "Tip path\n" + tag,
    )

    plot_joint_trajectories(
        phi,
        outdir / "joint_trajectories.png",
        tag,
        n_cycles=args.ncycles,
    )

    plot_cycle_reward(
        rewards,
        outdir / "cycle_reward.png",
        "Per-step reward over cycle\n" + tag,
    )

    if not args.no_animation:
        animate_beat(
            phi,
            outdir / "beat_animation.gif",
            "Learned beat\n" + tag,
            fps=args.fps,
        )

    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "result_file": str(result_path),
        "npz_keys": data["keys"],
        "cycle_start": data["cycle_start"],
        "cycle_length": data["cycle_length"],
        "N": int(N),
        "cycle_mean_reward": mean_reward,
        "cycle_total_reward": total_reward,
        "tip_signed_area": A,
        "tip_abs_area": abs(A),
        "tip_path_length": plen,
        "reward_per_abs_tip_area": rho,
        "n_cycles_traj": args.ncycles,
    }

    with open(outdir / "viz_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[viz] done")


if __name__ == "__main__":
    main()
