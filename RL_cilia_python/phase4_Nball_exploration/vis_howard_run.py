#!/usr/bin/env python3
"""
visualize_howard_run.py
=======================

Visualizer for Howard policy-iteration cilia outputs.

Expected files from run_howard_pi.py, for example:

    howard_summary_2ball.npz
    howard_stroke_2ball.npy
    howard_states_2ball.npy
    howard_rewards_2ball.npy
    howard_actions_2ball.npy

Preferred input is the summary file because it contains cycle_start and
cycle_length:

    python visualize_howard_run.py --nballs 2
    python visualize_howard_run.py --nballs 3
    python visualize_howard_run.py --nballs 4

or directly:

    python visualize_howard_run.py --file howard_summary_3ball.npz
    python visualize_howard_run.py --file howard_stroke_3ball.npy

Outputs, written to "<basename>_figs" by default:

    stroke_overlay.png
    joint_trajectories.png       stacked relative angles phi_k and cumulative angles psi_k
    cycle_reward.png
    tip_path.png
    beat_animation.gif
    viz_metadata.json
"""

# ====================== CONFIG DEFAULTS ======================
DEFAULT_NBALLS = 4
DEFAULT_RESULT_FILE = None      # None -> howard_summary_{N}ball.npz if present
N_CYCLES_TRAJ = 5
OUTDIR = None                   # None -> "<resultname>_figs"
MAKE_ANIMATION = True
FPS = 6
USE_CYCLE_ONLY = True           # True -> plot detected periodic cycle only
# =============================================================

import argparse
import datetime
import json
import os
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ------------------------------------------------------------------
# Kinematics from relative joint angles phi
# ------------------------------------------------------------------

def chain_positions_from_phi(phi):
    """
    Convert relative joint angles phi = [phi1, ..., phiN] into chain node
    positions in the x-z plane.

    Physical segment angles:
        psi_k = sum_{j=1}^k phi_j

    Segment length:
        L = 1/N

    Returns:
        pts : shape (N+1, 2), including anchor at (0,0)
        psi : shape (N,)
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


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def infer_nballs_from_name(path):
    m = re.search(r"_(\d+)ball", str(path))
    if m:
        return int(m.group(1))
    return None


def scalar_int(x, default=-1):
    if x is None:
        return default
    arr = np.asarray(x)
    if arr.size == 0:
        return default
    return int(arr.reshape(-1)[0])


def default_result_file(nballs):
    summary = Path(f"howard_summary_{nballs}ball.npz")
    stroke = Path(f"howard_stroke_{nballs}ball.npy")

    if summary.exists():
        return summary
    if stroke.exists():
        return stroke

    raise FileNotFoundError(
        f"Could not find {summary} or {stroke}. "
        f"Run Howard first, or pass --file explicitly."
    )


def load_howard_result(path=None, nballs=None):
    """
    Load Howard output from either:

        howard_summary_Nball.npz

    or:

        howard_stroke_Nball.npy

    The summary file is preferred because it stores cycle_start and
    cycle_length. If loading the stroke .npy, this function also looks for
    sibling howard_states_Nball.npy and howard_rewards_Nball.npy.
    """
    if path is None:
        if nballs is None:
            nballs = DEFAULT_NBALLS
        path = default_result_file(nballs)

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    inferred = infer_nballs_from_name(path)
    if nballs is None:
        nballs = inferred

    data = {
        "source_file": str(path),
        "nballs": nballs,
        "angles": None,
        "states": None,
        "actions": None,
        "rewards": None,
        "cycle_start": -1,
        "cycle_length": -1,
        "eta": None,
        "bias": None,
    }

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            keys = list(z.keys())

            if "angles" not in z:
                raise KeyError(
                    f"{path} does not contain key 'angles'. "
                    f"Available keys are: {keys}"
                )

            data["angles"] = np.asarray(z["angles"], dtype=float)

            if "states" in z:
                data["states"] = np.asarray(z["states"], dtype=int)
            if "actions" in z:
                data["actions"] = np.asarray(z["actions"], dtype=int)
            if "rewards" in z:
                data["rewards"] = np.asarray(z["rewards"], dtype=float)
            if "cycle_start" in z:
                data["cycle_start"] = scalar_int(z["cycle_start"], -1)
            if "cycle_length" in z:
                data["cycle_length"] = scalar_int(z["cycle_length"], -1)
            if "eta" in z:
                data["eta"] = np.asarray(z["eta"], dtype=float)
            if "bias" in z:
                data["bias"] = np.asarray(z["bias"], dtype=float)

    elif path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)

        # Howard stroke files should be ordinary arrays of shape (T+1, N).
        if arr.dtype == object and arr.size == 1:
            raise ValueError(
                f"{path} appears to contain an object/dict result, not a "
                f"Howard angle array. Use the PPO visualizer for that file."
            )

        data["angles"] = np.asarray(arr, dtype=float)

        if nballs is None:
            nballs = int(data["angles"].shape[1])
            data["nballs"] = nballs

        # Try to load sibling files if present.
        if nballs is not None:
            states_file = path.with_name(f"howard_states_{nballs}ball.npy")
            actions_file = path.with_name(f"howard_actions_{nballs}ball.npy")
            rewards_file = path.with_name(f"howard_rewards_{nballs}ball.npy")
            summary_file = path.with_name(f"howard_summary_{nballs}ball.npz")

            if states_file.exists():
                data["states"] = np.load(states_file, allow_pickle=True)
            if actions_file.exists():
                data["actions"] = np.load(actions_file, allow_pickle=True)
            if rewards_file.exists():
                data["rewards"] = np.load(rewards_file, allow_pickle=True)

            # If summary exists, borrow cycle_start/cycle_length from it.
            if summary_file.exists():
                with np.load(summary_file, allow_pickle=True) as z:
                    if "cycle_start" in z:
                        data["cycle_start"] = scalar_int(z["cycle_start"], -1)
                    if "cycle_length" in z:
                        data["cycle_length"] = scalar_int(z["cycle_length"], -1)
    else:
        raise ValueError(f"Expected .npz or .npy input, got: {path}")

    if data["angles"] is None:
        raise ValueError("No angle array was loaded.")

    if data["angles"].ndim != 2:
        raise ValueError(
            f"Expected angles to have shape (steps, N), got {data['angles'].shape}."
        )

    if data["nballs"] is None:
        data["nballs"] = int(data["angles"].shape[1])

    if data["angles"].shape[1] != data["nballs"]:
        raise ValueError(
            f"Filename/request says N={data['nballs']}, but angles have "
            f"shape {data['angles'].shape}."
        )

    return data


def extract_plot_segment(data, use_cycle_only=True):
    """
    Select the periodic cycle if cycle_start/cycle_length are known.
    Otherwise use the full saved trajectory.

    For a detected cycle:
        states/angles have length actions+1
        rewards/actions have length actions

    If cycle_start=i0 and cycle_length=L, the periodic states are
        angles[i0 : i0+L]
    and the repeated closing state is angles[i0+L].
    """
    angles = data["angles"]
    states = data["states"]
    rewards = data["rewards"]
    actions = data["actions"]

    i0 = int(data.get("cycle_start", -1))
    L = int(data.get("cycle_length", -1))

    if use_cycle_only and i0 >= 0 and L > 0:
        i1 = i0 + L

        plot_angles = angles[i0:i1]

        plot_states = None
        if states is not None and len(states) >= i1:
            plot_states = states[i0:i1]

        plot_rewards = None
        if rewards is not None and len(rewards) >= i1:
            plot_rewards = rewards[i0:i1]
        elif rewards is not None and len(rewards) >= i0:
            plot_rewards = rewards[i0:]

        plot_actions = None
        if actions is not None and len(actions) >= i1:
            plot_actions = actions[i0:i1]
        elif actions is not None and len(actions) >= i0:
            plot_actions = actions[i0:]

        label = f"cycle start={i0}, length={L}"
        return plot_angles, plot_states, plot_rewards, plot_actions, label

    # Fallback: full trajectory. If last angle repeats first, remove duplicate
    # for cleaner periodic plotting.
    plot_angles = angles
    if len(plot_angles) > 2 and np.allclose(plot_angles[0], plot_angles[-1]):
        plot_angles = plot_angles[:-1]

    label = f"full rollout, length={len(plot_angles)}"
    return plot_angles, states, rewards, actions, label


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

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

    fig, ax = plt.subplots(figsize=(6.2, 6))
    _wall_and_limits(ax, shapes)

    for i, pts in enumerate(shapes):
        c = cmap(i / max(n - 1, 1))
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.8, alpha=0.85, zorder=3)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", color=c, ms=4, zorder=4)

    ax.plot(0, 0, "ks", ms=8, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

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

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_joint_trajectories(angles, outpath, title, n_cycles=5):
    """
    Make a PPO-style stacked angle figure for Howard outputs.

    Top panel:
        relative joint angles phi_k

    Bottom panel:
        cumulative segment angles psi_k = sum_{j<=k} phi_j

    Saved to joint_trajectories.png.
    """
    angles = np.asarray(angles, dtype=float)
    nballs = angles.shape[1]

    # Relative joint angles.
    rep_phi = np.tile(angles, (n_cycles, 1))
    rep_phi_deg = np.degrees(rep_phi)

    # Cumulative segment angles.
    psi = np.cumsum(angles, axis=1)
    rep_psi = np.tile(psi, (n_cycles, 1))
    rep_psi_deg = np.degrees(rep_psi)

    steps = np.arange(len(rep_phi_deg))
    L = len(angles)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True
    )

    # ----------------------------------------------------------
    # Top: relative joint angles phi_k
    # ----------------------------------------------------------
    for k in range(nballs):
        ax1.plot(
            steps,
            rep_phi_deg[:, k],
            "-",
            lw=1.5,
            label=f"$\\phi_{{{k+1}}}$",
        )

    for c in range(1, n_cycles):
        ax1.axvline(c * L - 0.5, color="0.8", lw=0.8, zorder=0)

    ax1.set_title(f"Relative joint angles ({n_cycles} cycles)")
    ax1.set_ylabel("degrees")
    ax1.legend(fontsize=8, ncol=max(1, nballs), loc="upper right")
    ax1.grid(alpha=0.3)

    # ----------------------------------------------------------
    # Bottom: cumulative segment angles psi_k
    # ----------------------------------------------------------
    for k in range(nballs):
        ax2.plot(
            steps,
            rep_psi_deg[:, k],
            "-",
            lw=1.5,
            label=f"$\\psi_{{{k+1}}}$",
        )

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
    if rewards is None:
        return False

    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0 or np.all(~np.isfinite(rewards)):
        return False

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

    ax.set_xlabel("step in selected segment")
    ax.set_ylabel("immediate reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return True


def animate_beat(angles, outpath, title, fps=6):
    shapes = all_chain_positions(angles)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _wall_and_limits(ax, shapes)

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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Howard PI cilia strokes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nballs",
        "--NBALLS",
        type=int,
        default=None,
        help="which N to visualize if --file is not given; inferred from filename when possible",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_RESULT_FILE,
        help="Howard output file: howard_summary_Nball.npz or howard_stroke_Nball.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=OUTDIR,
        help="output directory for figures",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="plot full rollout instead of detected cycle only",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip GIF animation",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="animation frames per second",
    )
    parser.add_argument(
        "--n-cycles-traj",
        type=int,
        default=N_CYCLES_TRAJ,
        help="number of repeated cycles in angle trajectory plots",
    )
    args = parser.parse_args()

    #data = load_howard_result(path=args.file, nballs=args.nballs)
    data = load_howard_result(
    path=args.file,
    nballs=(args.nballs if args.nballs is not None else DEFAULT_NBALLS if args.file is None else None),
    )

    angles, states, rewards, actions, segment_label = extract_plot_segment(
        data,
        use_cycle_only=(USE_CYCLE_ONLY and not args.all),
    )

    if len(angles) == 0:
        raise ValueError("Selected angle segment is empty.")

    source_path = Path(data["source_file"])
    base = source_path.stem
    outdir = Path(args.out_dir) if args.out_dir is not None else Path(f"{base}_figs")
    outdir.mkdir(parents=True, exist_ok=True)

    nballs = int(data["nballs"])
    avg_reward = float(np.nanmean(rewards)) if rewards is not None and len(rewards) else float("nan")

    tag = (
        f"Howard PI | N={nballs} | {segment_label} | "
        f"mean reward={avg_reward:.6g}"
    )

    print(f"[viz] source: {source_path}")
    print(f"[viz] N={nballs}")
    print(f"[viz] selected angles shape: {angles.shape}")
    print(f"[viz] {segment_label}")
    if rewards is not None:
        print(f"[viz] selected rewards length: {len(rewards)}, mean={avg_reward:.8g}")

    plot_stroke_overlay(
        angles,
        outdir / "stroke_overlay.png",
        "Howard gain-optimal stroke overlay\n" + tag,
    )

    plot_tip_path(
        angles,
        outdir / "tip_path.png",
        "Tip path over selected Howard stroke\n" + tag,
    )

    plot_joint_trajectories(
        angles,
        outdir / "joint_trajectories.png",
        tag,
        n_cycles=args.n_cycles_traj,
    )


    wrote_reward = plot_cycle_reward(
        rewards,
        outdir / "cycle_reward.png",
        "Per-step reward over selected Howard stroke\n" + tag,
    )

    if not args.no_animation:
        animate_beat(
            angles,
            outdir / "beat_animation.gif",
            "Howard gain-optimal beat\n" + tag,
            fps=args.fps,
        )

    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_file": str(source_path.resolve()),
        "nballs": nballs,
        "selected_segment": segment_label,
        "selected_angles_shape": list(angles.shape),
        "mean_reward": None if not np.isfinite(avg_reward) else avg_reward,
        "cycle_start": int(data.get("cycle_start", -1)),
        "cycle_length": int(data.get("cycle_length", -1)),
        "used_cycle_only": bool(USE_CYCLE_ONLY and not args.all),
        "n_cycles_traj": int(args.n_cycles_traj),
        "made_animation": bool(not args.no_animation),
        "wrote_reward_plot": bool(wrote_reward),
    }

    if states is not None:
        meta["states"] = np.asarray(states, dtype=int).tolist()
    if actions is not None:
        meta["actions"] = np.asarray(actions, dtype=int).tolist()
    if rewards is not None:
        meta["rewards"] = np.asarray(rewards, dtype=float).tolist()

    with open(outdir / "viz_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[viz] wrote figures to: {outdir}")


if __name__ == "__main__":
    main()