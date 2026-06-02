#!/usr/bin/env python3
"""
vis_ppo_sweep_selected_6panel.py
================================

Make 6-panel comparison figures for selected PPO sweep runs.

Selections per dtheta case:
  - best percent of Howard
  - median-ish percent of Howard
  - longest-cycle outlier

Default input:
    results/ppo_sweeps/summary_all.csv

Default output:
    figures/ppo_sweeps/comparison_6panel/

Run:
    python vis_ppo_sweep_selected_6panel.py

Outputs:
    selected_stroke_overlay_6panel.png
    selected_tip_path_6panel.png
    selected_reward_6panel.png
    selected_joint_angles_6panel.png

Optional:
    python vis_ppo_sweep_selected_6panel.py --summary results/ppo_sweeps/summary_all.csv
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


# ---------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------

def chain_positions_from_phi(phi):
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
        return obj.item()

    if isinstance(obj, np.ndarray) and obj.size == 1 and obj.dtype == object:
        return obj.reshape(-1)[0]

    raise ValueError(f"Expected saved dict in {path}, got array shape {obj.shape}")


def get_cycle_from_result(data):
    if "cycles" in data and data["cycles"] is not None and len(data["cycles"]) > 0:
        cyc = data["cycles"][0]

        if isinstance(cyc, np.ndarray) and cyc.shape == ():
            cyc = cyc.item()

        angles = np.asarray(cyc["angles"], dtype=float)
        rewards = np.asarray(cyc["rewards"], dtype=float)
        actions = np.asarray(cyc.get("actions", []), dtype=int)
        states = np.asarray(cyc.get("cycle", []), dtype=int)

        return angles, rewards, actions, states

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
# Summary reading / selection
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
    Return list of (label, kind, row), ordered as:
      pi20 best, median, longcycle
      pi30 best, median, longcycle
    """
    selected = []

    # Keep this order if present.
    labels = []
    for lab in ["pi20", "pi30"]:
        if lab in set(r["dtheta_label"] for r in rows):
            labels.append(lab)

    # Add any extra labels after.
    for lab in sorted(set(r["dtheta_label"] for r in rows)):
        if lab not in labels:
            labels.append(lab)

    for label in labels:
        rr = [r for r in rows if r["dtheta_label"] == label]

        best = max(rr, key=lambda r: r["ppo_percent_of_howard"])

        vals = np.array([r["ppo_percent_of_howard"] for r in rr], dtype=float)
        med_val = float(np.median(vals))
        median = min(rr, key=lambda r: abs(r["ppo_percent_of_howard"] - med_val))

        max_len = max(r["cycle_length"] for r in rr)
        long_candidates = [r for r in rr if r["cycle_length"] == max_len]
        long = max(long_candidates, key=lambda r: r["ppo_percent_of_howard"])

        selected.append((label, "best", best))
        selected.append((label, "median", median))
        selected.append((label, "longcycle", long))

    return selected


def load_selected_data(selected):
    loaded = []
    for label, kind, row in selected:
        data = load_ppo_result(row["result_path"])
        angles, rewards, actions, states = get_cycle_from_result(data)

        loaded.append(
            {
                "label": label,
                "kind": kind,
                "row": row,
                "angles": angles,
                "rewards": rewards,
                "actions": actions,
                "states": states,
                "shapes": all_chain_positions(angles),
            }
        )

    return loaded


def panel_title(item):
    r = item["row"]
    return (
        f"{item['label']} {item['kind']}\n"
        f"seed {r['seed']}, L={r['cycle_length']}, "
        f"{r['ppo_percent_of_howard']:.1f}% Howard"
    )


# ---------------------------------------------------------------------
# Shared axis helpers
# ---------------------------------------------------------------------

def global_shape_limits(items):
    xs = []
    zs = []

    for item in items:
        shapes = item["shapes"]
        xs.append(shapes[..., 0].ravel())
        zs.append(shapes[..., 1].ravel())

    xs = np.concatenate(xs)
    zs = np.concatenate(zs)

    xrange = xs.max() - xs.min()
    zrange = zs.max() - zs.min()
    pad = 0.12 * max(xrange, zrange, 1e-9)

    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (-0.05, zs.max() + pad)

    return xlim, ylim


def setup_wall_axis(ax, xlim, ylim):
    ax.axhspan(-1, 0, color="0.88", zorder=0)
    ax.axhline(0, color="0.4", lw=1.0, zorder=1)
    ax.plot(0, 0, "ks", ms=5, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------
# 6-panel figures
# ---------------------------------------------------------------------

def plot_stroke_overlay_6panel(items, outpath):
    xlim, ylim = global_shape_limits(items)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()

    cmap = plt.cm.viridis

    for ax, item in zip(axes, items):
        shapes = item["shapes"]
        n = len(shapes)

        setup_wall_axis(ax, xlim, ylim)

        for i, pts in enumerate(shapes):
            c = cmap(i / max(n - 1, 1))
            ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.4, alpha=0.85)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", color=c, ms=2.8)

        ax.set_title(panel_title(item), fontsize=10)

    # Shared colorbar based on longest cycle among selected.
    max_len = max(len(item["angles"]) for item in items)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_len - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("step in cycle")

    fig.suptitle("Selected PPO N=4 strokes: best / median / long-cycle by dtheta", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_tip_path_6panel(items, outpath):
    xlim, ylim = global_shape_limits(items)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()

    for ax, item in zip(axes, items):
        shapes = item["shapes"]
        tips = shapes[:, -1, :]

        setup_wall_axis(ax, xlim, ylim)

        ax.plot(tips[:, 0], tips[:, 1], "-o", lw=1.5, ms=3)
        ax.plot(tips[0, 0], tips[0, 1], "s", ms=6, label="start")
        ax.plot(tips[-1, 0], tips[-1, 1], "^", ms=6, label="end")

        ax.set_title(panel_title(item), fontsize=10)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8, loc="upper right")

    fig.suptitle("Selected PPO N=4 tip paths", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_reward_6panel(items, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0))
    axes = axes.ravel()

    all_rewards = np.concatenate([item["rewards"] for item in items])
    ymin = float(np.nanmin(all_rewards))
    ymax = float(np.nanmax(all_rewards))
    pad = 0.08 * max(ymax - ymin, 1e-9)
    ylim = (ymin - pad, ymax + pad)

    for ax, item in zip(axes, items):
        rewards = np.asarray(item["rewards"], dtype=float)
        steps = np.arange(len(rewards))
        mean = float(np.nanmean(rewards))

        ax.plot(steps, rewards, "-o", lw=1.5, ms=3.5)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axhline(mean, color="k", ls="--", lw=1.0, label=f"mean={mean:.3g}")
        ax.set_ylim(*ylim)
        ax.set_title(panel_title(item), fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Selected PPO N=4 per-step rewards", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_joint_angles_6panel(items, outpath):
    """
    One compact 6-panel figure showing relative joint angles only.
    The full stacked phi/psi version is better as individual figures,
    but this gives a quick comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0))
    axes = axes.ravel()

    all_deg = np.concatenate([np.degrees(item["angles"]).ravel() for item in items])
    ymin = float(np.nanmin(all_deg))
    ymax = float(np.nanmax(all_deg))
    pad = 0.08 * max(ymax - ymin, 1e-9)
    ylim = (ymin - pad, ymax + pad)

    for ax, item in zip(axes, items):
        angles = np.asarray(item["angles"], dtype=float)
        deg = np.degrees(angles)
        steps = np.arange(len(deg))
        nballs = deg.shape[1]

        for k in range(nballs):
            ax.plot(steps, deg[:, k], lw=1.4, label=f"$\\phi_{{{k+1}}}$")

        ax.set_ylim(*ylim)
        ax.set_title(panel_title(item), fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("degrees")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle("Selected PPO N=4 relative joint angles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_selected_table(items, outpath):
    rows = []
    for item in items:
        r = item["row"]
        rows.append(
            {
                "dtheta_label": item["label"],
                "kind": item["kind"],
                "seed": int(r["seed"]),
                "cycle_length": int(r["cycle_length"]),
                "ppo_avg_reward": float(r["ppo_avg_reward"]),
                "howard_gain": float(r["howard_gain"]),
                "howard_cycle_length": int(r["howard_cycle_length"]),
                "ppo_percent_of_howard": float(r["ppo_percent_of_howard"]),
                "result_path": r["result_path"],
            }
        )

    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/ppo_sweeps/summary_all.csv"),
        help="summary_all.csv from ppo_sweep_n4_compare_howard.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/ppo_sweeps/comparison_6panel"),
        help="output directory",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary_csv(args.summary)
    selected = select_runs(rows)
    items = load_selected_data(selected)

    print("=" * 72)
    print("Selected runs for 6-panel plots:")
    for item in items:
        r = item["row"]
        print(
            f"  {item['label']:>4} {item['kind']:>9}: "
            f"seed={r['seed']}, L={r['cycle_length']}, "
            f"avg={r['ppo_avg_reward']:.6g}, "
            f"%Howard={r['ppo_percent_of_howard']:.2f}"
        )

    plot_stroke_overlay_6panel(
        items,
        args.out_dir / "selected_stroke_overlay_6panel.png",
    )

    plot_tip_path_6panel(
        items,
        args.out_dir / "selected_tip_path_6panel.png",
    )

    plot_reward_6panel(
        items,
        args.out_dir / "selected_reward_6panel.png",
    )

    plot_joint_angles_6panel(
        items,
        args.out_dir / "selected_joint_angles_6panel.png",
    )

    write_selected_table(
        items,
        args.out_dir / "selected_runs.json",
    )

    print("=" * 72)
    print(f"[done] wrote 6-panel figures to: {args.out_dir}")
    print("Open with:")
    print(f"  open {args.out_dir}")


if __name__ == "__main__":
    main()