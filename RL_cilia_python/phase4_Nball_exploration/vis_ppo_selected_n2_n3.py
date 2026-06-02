#!/usr/bin/env python3
"""
vis_ppo_selected_n2_n3.py
=========================

Make 2x3 comparison figures for selected PPO runs from N=2 and N=3.

Rows:
    top    = N=2
    bottom = N=3

Columns:
    best | median | longest-cycle

Diagnostics produced:
    - selected_stroke_overlay_n2_n3.png
    - selected_tip_paths_n2_n3.png
    - selected_rewards_n2_n3.png
    - selected_joint_angles_n2_n3.png
    - selected_cumulative_angles_n2_n3.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------

ROOT = Path("results/ppo_sweeps_general")
OUTDIR = Path("figures/ppo_sweeps_general/n2_n3_selected")
OUTDIR.mkdir(parents=True, exist_ok=True)

N2_SUMMARY = ROOT / "N2_dtheta_pi20" / "summary.csv"
N3_SUMMARY = ROOT / "N3_dtheta_pi20" / "summary.csv"

FIG_DPI = 160
NREP_ANGLE = 5
NREP_REWARD = 4
SEGLEN = 1.0


# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------

def load_summary(path):
    df = pd.read_csv(path)
    numeric_cols = [
        "nballs", "dtheta", "seed", "timesteps", "elapsed_sec",
        "cycle_start", "cycle_length", "ppo_avg_reward",
        "howard_gain", "howard_cycle_length",
        "ppo_fraction_of_howard", "ppo_percent_of_howard"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_representatives(df):
    out = {}

    # best by percent Howard
    i_best = df["ppo_percent_of_howard"].idxmax()
    out["best"] = df.loc[i_best]

    # median = closest to median %Howard
    med = df["ppo_percent_of_howard"].median()
    i_med = (df["ppo_percent_of_howard"] - med).abs().idxmin()
    out["median"] = df.loc[i_med]

    # longest cycle
    i_long = df["cycle_length"].idxmax()
    out["longest"] = df.loc[i_long]

    return out


def ensure_result_path(row):
    rp = Path(row["result_path"])
    if rp.exists():
        return rp

    seed = int(row["seed"])
    nballs = int(row["nballs"])
    dlabel = row["dtheta_label"]
    candidate = ROOT / f"N{nballs}_dtheta_{dlabel}" / f"seed_{seed:03d}" / "result.npy"
    return candidate


def load_result(result_path):
    obj = np.load(result_path, allow_pickle=True).item()

    angles = np.asarray(obj["angles"], dtype=float)   # stored in radians
    rewards = np.asarray(obj["rewards"], dtype=float)

    cycle_start = int(obj.get("cycle_start", -1))
    cycle_length = int(obj.get("cycle_length", -1))

    if cycle_start >= 0 and cycle_length > 0:
        cycle_angles = angles[cycle_start:cycle_start + cycle_length]
        cycle_rewards = rewards[cycle_start:cycle_start + cycle_length]
    else:
        cycle_angles = angles
        cycle_rewards = rewards

    return {
        "angles_rad": cycle_angles,
        "rewards": cycle_rewards,
        "cycle_start": cycle_start,
        "cycle_length": cycle_length,
        "env_settings": obj.get("env_settings", {}),
        "train_settings": obj.get("train_settings", {}),
    }


def make_title(row, label):
    seed = int(row["seed"])
    L = int(row["cycle_length"])
    pct = float(row["ppo_percent_of_howard"])
    return f"{label}\nseed {seed}, L={L}, {pct:.1f}% Howard"


def repeat_1d(x, nrep):
    return np.tile(np.asarray(x), nrep)


def repeat_2d(x, nrep):
    return np.tile(np.asarray(x), (nrep, 1))


# ---------------------------------------------------------------------
# geometry / angle helpers
# ---------------------------------------------------------------------

def cumulative_angles_rad(phi_rad):
    """
    phi_rad: shape (T, N)
    returns psi_rad: shape (T, N)
    """
    return np.cumsum(phi_rad, axis=1)


def chain_points_from_phi_rad(phi_rad, seglen=1.0):
    """
    phi_rad is one state of relative joint angles in radians, shape (N,).
    Returns chain joint coordinates x,z of length N+1 including base.
    """
    psi = np.cumsum(phi_rad)
    x = [0.0]
    z = [0.0]
    for ang in psi:
        x.append(x[-1] + seglen * np.sin(ang))
        z.append(z[-1] + seglen * np.cos(ang))
    return np.array(x), np.array(z)


def tip_path_from_angles_rad(phi_cycle_rad):
    tips = []
    for phi in phi_cycle_rad:
        x, z = chain_points_from_phi_rad(phi, seglen=SEGLEN)
        tips.append([x[-1], z[-1]])
    return np.asarray(tips)


# ---------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------

def setup_axes_grid(title):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    return fig, axs


def add_row_labels(fig):
    fig.text(0.02, 0.73, "N=2", rotation=90, va="center", ha="center", fontsize=14)
    fig.text(0.02, 0.29, "N=3", rotation=90, va="center", ha="center", fontsize=14)


def plot_rewards_panel(ax, data, subtitle):
    r = repeat_1d(data["rewards"], NREP_REWARD)
    L = len(data["rewards"])
    x = np.arange(len(r))

    ax.plot(x, r, marker="o", ms=2.8, lw=1.3)
    ax.axhline(np.mean(data["rewards"]), ls="--", color="k", lw=1,
               label=f"mean={np.mean(data['rewards']):.3f}")

    for k in range(1, NREP_REWARD):
        ax.axvline(k * L, color="0.8", lw=0.8)

    ax.set_title(subtitle, fontsize=11)
    ax.set_xlabel(f"step ({NREP_REWARD} cycles of length {L})")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_joint_panel(ax, data, subtitle):
    phi_deg = np.rad2deg(data["angles_rad"])
    phi_deg_rep = repeat_2d(phi_deg, NREP_ANGLE)

    T, N = phi_deg_rep.shape
    x = np.arange(T)

    for j in range(N):
        ax.plot(x, phi_deg_rep[:, j], label=rf"$\phi_{{{j+1}}}$")

    L = len(phi_deg)
    for k in range(1, NREP_ANGLE):
        ax.axvline(k * L, color="0.8", lw=0.8)

    ax.set_title(subtitle, fontsize=11)
    ax.set_xlabel(f"step ({NREP_ANGLE} cycles of length {L})")
    ax.set_ylabel("degrees")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=min(N, 4))


def plot_cumulative_panel(ax, data, subtitle):
    phi_rad = data["angles_rad"]
    psi_deg = np.rad2deg(cumulative_angles_rad(phi_rad))
    psi_deg_rep = repeat_2d(psi_deg, NREP_ANGLE)

    T, N = psi_deg_rep.shape
    x = np.arange(T)

    for j in range(N):
        ax.plot(x, psi_deg_rep[:, j], label=rf"$\psi_{{{j+1}}}$")

    L = len(psi_deg)
    for k in range(1, NREP_ANGLE):
        ax.axvline(k * L, color="0.8", lw=0.8)

    ax.set_title(subtitle, fontsize=11)
    ax.set_xlabel(f"step ({NREP_ANGLE} cycles of length {L})")
    ax.set_ylabel("degrees")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=min(N, 4))


def plot_tip_panel(ax, data, subtitle):
    tips = tip_path_from_angles_rad(data["angles_rad"])

    ax.plot(tips[:, 0], tips[:, 1], "-o", ms=3, lw=1.4)
    ax.scatter([tips[0, 0]], [tips[0, 1]], marker="s", s=42, label="start")
    ax.scatter([tips[-1, 0]], [tips[-1, 1]], marker="^", s=42, label="end")
    ax.scatter([0], [0], marker="s", s=45, color="k")
    ax.axhline(0, color="0.5", lw=1)

    ax.set_title(subtitle, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)

    # give a little padding so the loop is visible
    xmin, xmax = np.min(tips[:, 0]), np.max(tips[:, 0])
    zmin, zmax = np.min(tips[:, 1]), np.max(tips[:, 1])
    dx = max(xmax - xmin, 0.2)
    dz = max(zmax - zmin, 0.2)
    ax.set_xlim(xmin - 0.15 * dx, xmax + 0.15 * dx)
    ax.set_ylim(min(-0.05, zmin - 0.10 * dz), zmax + 0.15 * dz)


def plot_stroke_panel(ax, data, subtitle):
    phi = data["angles_rad"]
    T = len(phi)
    cmap = plt.get_cmap("viridis")

    all_x = []
    all_z = []

    for k in range(T):
        x, z = chain_points_from_phi_rad(phi[k], seglen=SEGLEN)
        c = cmap(k / max(T - 1, 1))
        ax.plot(x, z, "-o", ms=3, lw=1.3, color=c)
        all_x.append(x)
        all_z.append(z)

    all_x = np.concatenate(all_x)
    all_z = np.concatenate(all_z)

    ax.scatter([0], [0], marker="s", s=45, color="k")
    ax.axhline(0, color="0.5", lw=1)

    ax.set_title(subtitle, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    xmin, xmax = np.min(all_x), np.max(all_x)
    zmin, zmax = np.min(all_z), np.max(all_z)
    dx = max(xmax - xmin, 0.2)
    dz = max(zmax - zmin, 0.2)
    ax.set_xlim(xmin - 0.15 * dx, xmax + 0.15 * dx)
    ax.set_ylim(min(-0.05, zmin - 0.10 * dz), zmax + 0.15 * dz)


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------

def build_selected_data():
    df2 = load_summary(N2_SUMMARY)
    df3 = load_summary(N3_SUMMARY)

    picks2 = pick_representatives(df2)
    picks3 = pick_representatives(df3)

    selected = {2: {}, 3: {}}

    for N, picks in [(2, picks2), (3, picks3)]:
        for key, row in picks.items():
            rpath = ensure_result_path(row)
            data = load_result(rpath)
            selected[N][key] = {
                "row": row,
                "data": data,
                "subtitle": make_title(row, key)
            }

    return selected


def make_figure(selected, kind):
    titles = {
        "stroke": "Selected PPO comparison: stroke overlays",
        "tip": "Selected PPO comparison: tip paths",
        "reward": "Selected PPO comparison: per-step rewards",
        "joint": "Selected PPO comparison: relative joint angles",
        "cumulative": "Selected PPO comparison: cumulative segment angles",
    }

    fig, axs = setup_axes_grid(titles[kind])
    add_row_labels(fig)

    col_keys = ["best", "median", "longest"]

    for i, N in enumerate([2, 3]):
        for j, key in enumerate(col_keys):
            ax = axs[i, j]
            item = selected[N][key]
            data = item["data"]
            subtitle = item["subtitle"]

            if kind == "stroke":
                plot_stroke_panel(ax, data, subtitle)
            elif kind == "tip":
                plot_tip_panel(ax, data, subtitle)
            elif kind == "reward":
                plot_rewards_panel(ax, data, subtitle)
            elif kind == "joint":
                plot_joint_panel(ax, data, subtitle)
            elif kind == "cumulative":
                plot_cumulative_panel(ax, data, subtitle)

    outname = {
        "stroke": "selected_stroke_overlay_n2_n3.png",
        "tip": "selected_tip_paths_n2_n3.png",
        "reward": "selected_rewards_n2_n3.png",
        "joint": "selected_joint_angles_n2_n3.png",
        "cumulative": "selected_cumulative_angles_n2_n3.png",
    }[kind]

    outpath = OUTDIR / outname
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outpath}")


def main():
    selected = build_selected_data()

    print("\nSelected runs:")
    for N in [2, 3]:
        print(f"\nN={N}")
        for key in ["best", "median", "longest"]:
            row = selected[N][key]["row"]
            print(
                f"  {key:7s}: seed={int(row['seed'])}, "
                f"L={int(row['cycle_length'])}, "
                f"%Howard={float(row['ppo_percent_of_howard']):.2f}"
            )

    for kind in ["stroke", "tip", "reward", "joint", "cumulative"]:
        make_figure(selected, kind)

    print("\nDone.")


if __name__ == "__main__":
    main()