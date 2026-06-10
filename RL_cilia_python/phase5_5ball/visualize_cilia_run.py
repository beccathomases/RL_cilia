#!/usr/bin/env python3
"""
visualize_cilia_run.py
======================

Standalone visualizer for the saved output of the cilia PPO pipeline
(the .npy files: policy + ranked cycles + env_settings + train_settings).

No dependency on gymnasium / torch / stable_baselines3 -- it re-implements
only the kinematics (state -> joint angles -> bead positions). Physics
rewards are read back from the saved cycle records, not recomputed.

Edit the CONFIG block below, then just run:  python visualize_cilia_run.py

Outputs (written to OUTDIR, default "<resultname>_figs"):
    stroke_overlay.png       all chain shapes of the cycle in one figure
    joint_trajectories.png   phi_k and psi_k over N_CYCLES_TRAJ cycles
    cycle_reward.png         per-step immediate reward (line plot)
    beat_animation.gif       animated beat + tip trail   (if MAKE_ANIMATION)
    viz_metadata.json        parameters + provenance for reproducibility
"""


# ====================== CONFIG: edit these ======================
RESULT_FILE    = "results/ppo_sweeps_n5/N5_dtheta_pi20/seed_000/result.npz"
CYCLE_INDEX    = 0
N_CYCLES_TRAJ  = 5
OUTDIR         = "figures/N5_pi20_seed000"
MAKE_ANIMATION = True
FPS            = 6
# ================================================================


import os
import json
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ------------------------------------------------------------------
# Kinematics (mirrors CiliaNBallEnv, no gym dependency)
# ------------------------------------------------------------------
def default_angle_ranges(Nballs):
    mins = np.array([-np.pi / 4] + [-np.pi / 2] * (Nballs - 1), dtype=float)
    maxs = np.array([np.pi / 4] + [np.pi / 2] * (Nballs - 1), dtype=float)
    return mins, maxs


def build_grid(env_settings):
    Nballs = int(env_settings["Nballs"])
    dtheta = float(env_settings.get("dtheta", np.pi / 20))
    mins = env_settings.get("angle_mins", None)
    maxs = env_settings.get("angle_maxs", None)
    if mins is None or maxs is None:
        mins, maxs = default_angle_ranges(Nballs)
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    widths = maxs - mins
    n_bins = np.rint(widths / dtheta).astype(int) + 1
    dangle = widths / (n_bins - 1)
    seg_len = 1.0 / Nballs
    return dict(Nballs=Nballs, n_bins=n_bins, angle_mins=mins,
                dangle=dangle, seg_len=seg_len)


def state_to_angles(state, grid):
    state = np.asarray(state, dtype=float)
    return grid["angle_mins"] + state * grid["dangle"]


def chain_positions(state, grid):
    """Return ((Nballs+1, 2) node positions [x, z] incl. anchor, phi, psi)."""
    phi = state_to_angles(state, grid)
    psi = np.cumsum(phi)
    L = grid["seg_len"]
    pts = [np.array([0.0, 0.0])]  # anchor at origin on the wall
    for k in range(grid["Nballs"]):
        prev = pts[-1]
        pts.append(prev + L * np.array([np.sin(psi[k]), np.cos(psi[k])]))
    return np.array(pts), phi, psi


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_result(path):
    out = np.load(path, allow_pickle=True).item()
    env_settings = dict(out["env_settings"])
    policy = np.asarray(out["policy"])
    cycles = out["cycles"]
    cycles = [c for c in (cycles.tolist() if hasattr(cycles, "tolist") else cycles)]
    return env_settings, policy, cycles


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------
def _wall_and_limits(ax, all_pts):
    xs = all_pts[..., 0].ravel()
    zs = all_pts[..., 1].ravel()
    xpad = 0.15 * (xs.max() - xs.min() + 1e-9)
    ax.axhspan(-1, 0, color="0.85", zorder=0)   # wall region
    ax.axhline(0, color="0.4", lw=1.2, zorder=1)
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(-0.05, zs.max() * 1.15 + 0.05)
    ax.set_aspect("equal", adjustable="box")


# ------------------------------------------------------------------
# 1) Stroke overlay: all shapes in one figure
# ------------------------------------------------------------------
def plot_stroke_overlay(states, grid, outpath, title):
    shapes = [chain_positions(s, grid)[0] for s in states]
    all_pts = np.array(shapes)
    n = len(shapes)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(6.2, 6))
    _wall_and_limits(ax, all_pts)
    for i, pts in enumerate(shapes):
        c = cmap(i / max(n - 1, 1))
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.8, alpha=0.85, zorder=3)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", color=c, ms=4, zorder=4)  # tip
    ax.plot(0, 0, "ks", ms=8, zorder=5)  # anchor
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("step in cycle")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 2) Hinge trajectories over several cycles
# ------------------------------------------------------------------
def plot_joint_trajectories(states, grid, outpath, title, n_cycles=5):
    L = len(states)
    rep = states * n_cycles
    phis, psis = [], []
    for s in rep:
        _, phi, psi = chain_positions(s, grid)
        phis.append(phi); psis.append(psi)
    phis = np.degrees(np.array(phis)); psis = np.degrees(np.array(psis))
    steps = np.arange(len(rep))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k in range(grid["Nballs"]):
        a1.plot(steps, phis[:, k], "-", lw=1.5, label=f"$\\phi_{{{k+1}}}$")
        a2.plot(steps, psis[:, k], "-", lw=1.5, label=f"$\\psi_{{{k+1}}}$")
    for c in range(1, n_cycles):  # faint cycle boundaries
        for a in (a1, a2):
            a.axvline(c * L - 0.5, color="0.8", lw=0.8, zorder=0)
    a1.set_title(f"relative joint angles  ({n_cycles} cycles)")
    a2.set_title("cumulative segment angles")
    a2.set_xlabel(f"step  ({n_cycles} cycles of length {L})")
    for a in (a1, a2):
        a.set_ylabel("degrees")
        a.legend(fontsize=8, ncol=grid["Nballs"], loc="upper right")
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------
# 3) Per-step reward as a line plot
# ------------------------------------------------------------------
def plot_cycle_reward(rewards, outpath, title):
    rewards = np.asarray(rewards, dtype=float)
    steps = np.arange(len(rewards))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(steps, rewards, "-o", color="#2a9d8f", lw=1.8, ms=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(rewards.mean(), color="k", ls="--", lw=1,
               label=f"mean = {rewards.mean():.3f}")
    ax.set_xlabel("step in cycle"); ax.set_ylabel("immediate reward (flux)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------
# Optional: animated beat
# ------------------------------------------------------------------
def animate_beat(states, grid, outpath, title, fps=6):
    shapes = [chain_positions(s, grid)[0] for s in states]
    all_pts = np.array(shapes)
    fig, ax = plt.subplots(figsize=(5, 5))
    _wall_and_limits(ax, all_pts)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    (line,) = ax.plot([], [], "-o", color="#2a6f97", lw=2.5, ms=5)
    ax.plot([0], [0], "ks", ms=7)
    (trail,) = ax.plot([], [], ".", color="#e07a5f", ms=3, alpha=0.6)
    tx, tz = [], []

    def update(frame):
        pts = shapes[frame % len(shapes)]
        line.set_data(pts[:, 0], pts[:, 1])
        tx.append(pts[-1, 0]); tz.append(pts[-1, 1])
        if len(tx) > 2 * len(shapes):
            tx.pop(0); tz.pop(0)
        trail.set_data(tx, tz)
        return line, trail

    anim = FuncAnimation(fig, update, frames=len(shapes) * 2,
                         interval=1000 / fps, blit=True)
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    env_settings, policy, cycles = load_result(RESULT_FILE)
    grid = build_grid(env_settings)
    if not cycles:
        print("No cycles found in result; nothing to visualize.")
        return

    ci = max(0, min(CYCLE_INDEX, len(cycles) - 1))
    rec = cycles[ci]
    states = [tuple(int(v) for v in s) for s in rec["cycle"]]
    rewards = rec.get("rewards", [np.nan] * len(states))

    base = os.path.basename(os.path.dirname(RESULT_FILE))
    outdir = OUTDIR or f"{base}_figs"

    #base = os.path.splitext(os.path.basename(RESULT_FILE))[0]
    #outdir = OUTDIR or f"{base}_figs"
    os.makedirs(outdir, exist_ok=True)

   
    tag = f"N=5 {seed_name}: L={L}, mean={mean_reward:.3f}, rho={rho:.1f}"
    print(f"[viz] N={grid['Nballs']}  n_bins={grid['n_bins'].tolist()}")
    print(f"[viz] cycle #{ci}: {len(states)} states  ({tag})")

    plot_stroke_overlay(states, grid, os.path.join(outdir, "stroke_overlay.png"),
                        "Learned beat (stroke overlay)\n" + tag)
    plot_joint_trajectories(states, grid,
                            os.path.join(outdir, "joint_trajectories.png"),
                            tag, n_cycles=N_CYCLES_TRAJ)
    plot_cycle_reward(rewards, os.path.join(outdir, "cycle_reward.png"),
                      "Per-step reward over cycle\n" + tag)
    if MAKE_ANIMATION:
        animate_beat(states, grid, os.path.join(outdir, "beat_animation.gif"),
                     "Learned beat\n" + tag, fps=FPS)

    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "result_file": os.path.abspath(RESULT_FILE),
        "cycle_index": ci,
        "n_cycles_total": len(cycles),
        "cycle_length": rec.get("length", len(states)),
        "avg_reward": rec.get("avg_reward", None),
        "n_cycles_traj": N_CYCLES_TRAJ,
        "env_settings": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in env_settings.items()},
        "n_bins": grid["n_bins"].tolist(),
        "states": states,
    }
    with open(os.path.join(outdir, "viz_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[viz] wrote figures + viz_metadata.json to: {outdir}")


if __name__ == "__main__":
    main()