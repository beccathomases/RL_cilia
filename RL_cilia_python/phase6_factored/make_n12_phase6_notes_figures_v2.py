from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/ppo_sweeps_phase6_n12_t6e6_radscale0p4")
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures_notes_v2"
FIG_DIR.mkdir(exist_ok=True, parents=True)

CASES = ["pi20", "pi30", "pi40"]
DENOMS = {"pi20": 20, "pi30": 30, "pi40": 40}
NCYCLES_SHOW = 4

RENAME = {
    "cycle_mean_reward": "mean_reward",
    "cycle_total_reward": "cycle_reward",
    "tip_abs_area": "tip_area",
    "reward_per_abs_tip_area": "rho_reward_per_area",
    "wrong_orientation_flag": "wrong_orientation",
}


def savefig(fig, name):
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


def load_all_summary():
    rows = []
    for case in CASES:
        path = ROOT / f"N12_dtheta_{case}_factored" / "summary.csv"
        df = pd.read_csv(path).rename(columns=RENAME)
        df["case"] = case
        df["dtheta"] = np.pi / DENOMS[case]
        df["mean_reward_over_dtheta"] = df["mean_reward"] / df["dtheta"]
        df["cycle_phase_length"] = df["cycle_length"] * df["dtheta"]
        df["cycle_phase_over_pi2"] = df["cycle_phase_length"] / (np.pi / 2)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_pick_table(all_df):
    pick_rows = []
    for case in CASES:
        df = all_df[all_df["case"] == case].copy()
        dfm = df.sort_values("mean_reward").reset_index(drop=True)

        picks = [
            ("worst_mean", dfm.iloc[0]),
            ("median_mean", dfm.iloc[len(dfm)//2]),
            ("best_mean", dfm.iloc[-1]),
            ("longest_cycle", df.sort_values("cycle_length").iloc[-1]),
        ]

        mode_L = int(df["cycle_length"].mode().iloc[0])
        mode_df = df[df["cycle_length"] == mode_L].sort_values("mean_reward").reset_index(drop=True)
        picks.append(("modal_L_median", mode_df.iloc[len(mode_df)//2]))

        seen = set()
        for label, row in picks:
            seed = int(row["seed"])
            key = (case, seed)
            if key in seen:
                continue
            seen.add(key)
            r = row.copy()
            r["pick_label"] = label
            pick_rows.append(r)

    return pd.DataFrame(pick_rows)


def load_picks(all_df):
    path = TABLE_DIR / "n12_phase6_seed_picks_for_vis.csv"
    if path.exists():
        picks = pd.read_csv(path)
    else:
        picks = make_pick_table(all_df)
        TABLE_DIR.mkdir(exist_ok=True, parents=True)
        picks.to_csv(path, index=False)

    if "dtheta" not in picks.columns:
        picks["dtheta"] = picks["case"].map(lambda c: np.pi / DENOMS[c])
    return picks


def resolve_result_path(row):
    p = Path(str(row["result_path"]))
    if p.exists():
        return p

    case = row["case"]
    seed = int(row["seed"])

    candidates = [
        ROOT / f"N12_dtheta_{case}_factored" / f"seed_{seed:03d}" / "result.npz",
    ]
    candidates += list(Path(".").glob(f"results/**/N12_dtheta_{case}_factored/seed_{seed:03d}/result.npz"))

    for q in candidates:
        if q.exists():
            return q

    raise FileNotFoundError(f"Could not find result.npz for {case} seed {seed}. Original path was {p}")


def cycle_indices(start, L, n):
    return (start + np.arange(L)) % n


def load_cycle(row):
    """
    Load exactly one detected cycle.

    Important: do NOT append the first point to the end for tip plots.
    The saved tip coordinates can have translational drift over a swimming
    cycle, so closing the loop creates artificial diagonal chords.
    """
    p = resolve_result_path(row)
    z = np.load(p, allow_pickle=True)

    s = int(np.asarray(z["cycle_start"]))
    L = int(np.asarray(z["cycle_length"]))

    idx = cycle_indices(s, L, len(z["rewards"]))

    phi = np.asarray(z["phi"])[idx]
    tip = np.asarray(z["tip"])[idx]
    rewards = np.asarray(z["rewards"])[idx]
    actions = np.asarray(z["actions"])[idx]

    return {
        "path": p,
        "cycle_start": s,
        "cycle_length": L,
        "phi": phi,
        "tip": tip,
        "rewards": rewards,
        "actions": actions,
    }


def tile_cycle_array(a, ncycles=NCYCLES_SHOW):
    return np.concatenate([a] * ncycles, axis=0)


def tiled_time(cyc, ncycles=NCYCLES_SHOW):
    L = cyc["cycle_length"]
    return np.arange(ncycles * L), L


def detrend_tip_open_path(tip):
    """
    Remove the straight start-to-end drift from an open tip path.
    This is for visualizing the local stroke shape, not for computing metrics.
    """
    tip = np.asarray(tip, dtype=float)
    if len(tip) <= 1:
        return tip.copy()

    alpha = np.linspace(0.0, 1.0, len(tip))[:, None]
    drift = alpha * (tip[-1] - tip[0])[None, :]
    out = tip - drift
    out = out - out.mean(axis=0, keepdims=True)
    return out


def center_open_path(tip):
    tip = np.asarray(tip, dtype=float)
    return tip - tip.mean(axis=0, keepdims=True)


def plot_tip_path(ax, cyc, title=None, mode="detrended"):
    tip = cyc["tip"]

    if mode == "raw":
        x = tip[:, 0]
        y = tip[:, 1]
        xlabel = "tip x"
        ylabel = "tip y"
    elif mode == "centered":
        tt = center_open_path(tip)
        x = tt[:, 0]
        y = tt[:, 1]
        xlabel = "centered tip x"
        ylabel = "centered tip y"
    else:
        tt = detrend_tip_open_path(tip)
        x = tt[:, 0]
        y = tt[:, 1]
        xlabel = "drift-removed tip x"
        ylabel = "drift-removed tip y"

    ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2)
    ax.scatter(x[0], y[0], marker="s", s=28, label="start")
    ax.scatter(x[-1], y[-1], marker="x", s=40, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_tip_raw_and_detrended(ax_raw, ax_det, cyc, title=None):
    plot_tip_path(ax_raw, cyc, title=f"{title}\nraw open tip path", mode="raw")
    plot_tip_path(ax_det, cyc, title="drift-removed stroke path", mode="detrended")


def plot_phi_heatmap(ax, cyc, title=None, ncycles=NCYCLES_SHOW):
    phi = np.unwrap(cyc["phi"], axis=0)
    phi_rep = tile_cycle_array(phi, ncycles=ncycles)
    im = ax.imshow(phi_rep.T, aspect="auto", origin="lower")
    L = cyc["cycle_length"]
    for k in range(1, ncycles):
        ax.axvline(k * L - 0.5, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("phase index")
    if title:
        ax.set_title(title)
    return im


def plot_action_heatmap(ax, cyc, title=None, ncycles=NCYCLES_SHOW):
    actions = tile_cycle_array(cyc["actions"], ncycles=ncycles)
    im = ax.imshow(actions.T, aspect="auto", origin="lower", interpolation="nearest")
    L = cyc["cycle_length"]
    for k in range(1, ncycles):
        ax.axvline(k * L - 0.5, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("phase index")
    if title:
        ax.set_title(title)
    return im


def plot_rewards(ax, cyc, title=None, ncycles=NCYCLES_SHOW):
    r = tile_cycle_array(cyc["rewards"], ncycles=ncycles)
    t = np.arange(len(r))
    L = cyc["cycle_length"]

    ax.plot(t, r, marker="o", markersize=2.5, linewidth=1)
    ax.axhline(0, linewidth=0.8)
    for k in range(1, ncycles):
        ax.axvline(k * L - 0.5, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    if title:
        ax.set_title(title)


def plot_cumulative_rewards(ax, cyc, title=None, ncycles=NCYCLES_SHOW):
    r = tile_cycle_array(cyc["rewards"], ncycles=ncycles)
    t = np.arange(len(r))
    cr = np.cumsum(r)
    cr = cr - cr[0]
    L = cyc["cycle_length"]

    ax.plot(t, cr, linewidth=1.2)
    for k in range(1, ncycles):
        ax.axvline(k * L - 0.5, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative reward")
    if title:
        ax.set_title(title)


def plot_phi_profiles(ax, cyc, title=None, nsnaps=6):
    phi = np.unwrap(cyc["phi"], axis=0)
    L, N = phi.shape
    inds = np.linspace(0, L, nsnaps, endpoint=False, dtype=int)
    j = np.arange(N)

    for k in inds:
        ax.plot(j, phi[k], marker="o", markersize=2.5, linewidth=1, label=f"k={k}")

    ax.set_xlabel("phase index")
    ax.set_ylabel("unwrapped phi")
    if title:
        ax.set_title(title)


def plot_summary_metrics(all_df):
    metrics = [
        ("mean_reward_over_dtheta", r"$\langle R\rangle/\Delta\theta$"),
        ("cycle_reward", "cycle reward"),
        ("tip_area", "abs tip area"),
        ("tip_path_length", "tip path length"),
        ("rho_reward_per_area", "reward / area"),
        ("cycle_phase_over_pi2", r"cycle phase / $(\pi/2)$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, label) in zip(axes, metrics):
        data = [all_df.loc[all_df["case"] == c, col].to_numpy() for c in CASES]
        ax.boxplot(data, labels=CASES, showmeans=True)

        for i, vals in enumerate(data, start=1):
            x = np.full_like(vals, i, dtype=float)
            x += np.linspace(-0.08, 0.08, len(vals))
            ax.plot(x, vals, "o", markersize=3)

        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle("N=12 Phase 6 factored PPO: discretization summary", y=1.02)
    savefig(fig, "fig01_summary_metrics")


def plot_modal_gallery(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(3, 5, figsize=(17, 9), constrained_layout=True)

    for row_i, case in enumerate(CASES):
        row = modal[modal["case"] == case].iloc[0]
        cyc = load_cycle(row)
        title = f"{case} seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_path(axes[row_i, 0], cyc, title=f"{title}\nraw open tip", mode="raw")
        plot_tip_path(axes[row_i, 1], cyc, title="drift-removed tip stroke", mode="detrended")
        plot_phi_heatmap(axes[row_i, 2], cyc, title=f"phi, {NCYCLES_SHOW} cycles")
        plot_rewards(axes[row_i, 3], cyc, title=f"reward, {NCYCLES_SHOW} cycles")
        plot_cumulative_rewards(axes[row_i, 4], cyc, title=f"cumulative reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("Modal-stroke gallery across angular refinements", y=1.02)
    savefig(fig, "fig02_modal_stroke_gallery_v2")


def plot_pi40_gallery(picks):
    sub = picks[
        (picks["case"] == "pi40")
        & (picks["pick_label"].isin(["worst_mean", "median_mean", "best_mean", "longest_cycle"]))
    ].copy()

    fig, axes = plt.subplots(len(sub), 5, figsize=(17, 3.1 * len(sub)), constrained_layout=True)
    if len(sub) == 1:
        axes = axes[None, :]

    for i, (_, row) in enumerate(sub.iterrows()):
        cyc = load_cycle(row)
        label = f"{row['pick_label']}: seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_path(axes[i, 0], cyc, title=f"{label}\nraw open tip", mode="raw")
        plot_tip_path(axes[i, 1], cyc, title="drift-removed tip stroke", mode="detrended")
        plot_phi_heatmap(axes[i, 2], cyc, title=f"phi, {NCYCLES_SHOW} cycles")
        plot_rewards(axes[i, 3], cyc, title=f"reward, {NCYCLES_SHOW} cycles")
        plot_cumulative_rewards(axes[i, 4], cyc, title=f"cumulative reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("pi40 seed gallery: refined runs", y=1.01)
    savefig(fig, "fig03_pi40_seed_gallery_v2")


def plot_commensurate_gallery(picks):
    targets = [
        ("pi20", "modal_L_median"),
        ("pi20", "longest_cycle"),
        ("pi30", "modal_L_median"),
        ("pi30", "longest_cycle"),
    ]

    fig, axes = plt.subplots(len(targets), 5, figsize=(17, 3.0 * len(targets)), constrained_layout=True)

    for i, (case, pick_label) in enumerate(targets):
        row = picks[(picks["case"] == case) & (picks["pick_label"] == pick_label)].iloc[0]
        cyc = load_cycle(row)
        title = f"{case} {pick_label}, seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_path(axes[i, 0], cyc, title=f"{title}\nraw open tip", mode="raw")
        plot_tip_path(axes[i, 1], cyc, title="drift-removed tip stroke", mode="detrended")
        plot_phi_heatmap(axes[i, 2], cyc, title=f"phi, {NCYCLES_SHOW} cycles")
        plot_rewards(axes[i, 3], cyc, title=f"reward, {NCYCLES_SHOW} cycles")
        plot_cumulative_rewards(axes[i, 4], cyc, title=f"cumulative reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("Commensurate-cycle comparison: modal vs repeated strokes", y=1.01)
    savefig(fig, "fig04_commensurate_cycle_gallery_v2")


def plot_action_maps(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(3, 3, figsize=(13, 8), constrained_layout=True)

    for i, case in enumerate(CASES):
        row = modal[modal["case"] == case].iloc[0]
        cyc = load_cycle(row)

        plot_action_heatmap(
            axes[i, 0],
            cyc,
            title=f"{case} seed {int(row['seed'])}, L={cyc['cycle_length']}\nactions, {NCYCLES_SHOW} cycles"
        )
        plot_phi_heatmap(axes[i, 1], cyc, title=f"phi, {NCYCLES_SHOW} cycles")
        plot_rewards(axes[i, 2], cyc, title=f"reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("Modal policies: factored action maps, phase evolution, and reward rhythm", y=1.02)
    savefig(fig, "fig05_action_phase_reward_maps_v2")


def resample_open_path(y, n=200):
    """
    Resample an open path, not a closed periodic curve.
    This avoids drawing artificial closing chords.
    """
    y = np.asarray(y)
    old = np.linspace(0, 1, len(y))
    new = np.linspace(0, 1, n)

    if y.ndim == 1:
        return np.interp(new, old, y)

    out = []
    for j in range(y.shape[1]):
        out.append(np.interp(new, old, y[:, j]))
    return np.stack(out, axis=1)


def resample_periodic_scalar(y, n=240):
    """
    Use this only for quantities that are cycle-periodic, such as reward sequence.
    """
    y = np.asarray(y)
    old = np.linspace(0, 1, len(y), endpoint=False)
    new = np.linspace(0, 1, n, endpoint=False)
    xp = np.r_[old, 1.0]
    yp = np.r_[y, y[0]]
    return np.interp(new, xp, yp)


def resample_periodic_matrix(y, n=240):
    y = np.asarray(y)
    old = np.linspace(0, 1, len(y), endpoint=False)
    new = np.linspace(0, 1, n, endpoint=False)
    xp = np.r_[old, 1.0]
    out = []
    for j in range(y.shape[1]):
        yp = np.r_[y[:, j], y[0, j]]
        out.append(np.interp(new, xp, yp))
    return np.stack(out, axis=1)


def plot_phase_normalized_overlay(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    # Panel 1: overlay drift-removed open tip strokes.
    ax = axes[0, 0]
    for _, row in modal.iterrows():
        cyc = load_cycle(row)
        tip = detrend_tip_open_path(cyc["tip"])
        tip_rs = resample_open_path(tip, n=220)
        ax.plot(tip_rs[:, 0], tip_rs[:, 1], label=f"{row['case']} seed {int(row['seed'])}")
        ax.scatter(tip_rs[0, 0], tip_rs[0, 1], marker="s", s=20)
        ax.scatter(tip_rs[-1, 0], tip_rs[-1, 1], marker="x", s=28)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("drift-removed tip strokes\n(open paths, no forced closure)")
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip y")
    ax.legend(fontsize=8)

    # Panel 2: reward over 4 cycles, normalized cycle coordinate repeated.
    ax = axes[0, 1]
    for _, row in modal.iterrows():
        cyc = load_cycle(row)
        r = tile_cycle_array(cyc["rewards"], ncycles=NCYCLES_SHOW)
        phase = np.arange(len(r)) / cyc["cycle_length"]
        ax.plot(phase, r, marker="o", markersize=2, linewidth=1, label=row["case"])
    ax.axhline(0, linewidth=0.8)
    for k in range(1, NCYCLES_SHOW):
        ax.axvline(k, linewidth=0.8)
    ax.set_title(f"reward over {NCYCLES_SHOW} cycles")
    ax.set_xlabel("cycle number")
    ax.set_ylabel("reward")

    # Panel 3: cumulative reward over 4 cycles.
    ax = axes[0, 2]
    for _, row in modal.iterrows():
        cyc = load_cycle(row)
        r = tile_cycle_array(cyc["rewards"], ncycles=NCYCLES_SHOW)
        phase = np.arange(len(r)) / cyc["cycle_length"]
        cr = np.cumsum(r)
        cr = cr - cr[0]
        ax.plot(phase, cr, linewidth=1.2, label=row["case"])
    for k in range(1, NCYCLES_SHOW):
        ax.axvline(k, linewidth=0.8)
    ax.set_title(f"cumulative reward over {NCYCLES_SHOW} cycles")
    ax.set_xlabel("cycle number")
    ax.set_ylabel("cumulative reward")

    # Bottom row: phase profiles at three normalized phases.
    for ax, q in zip(axes[1, :], [0.0, 0.25, 0.5]):
        for _, row in modal.iterrows():
            cyc = load_cycle(row)
            phi = np.unwrap(cyc["phi"], axis=0)
            phi_rs = resample_periodic_matrix(phi, n=240)
            k = int(q * 240)
            ax.plot(
                np.arange(phi_rs.shape[1]),
                phi_rs[k],
                marker="o",
                markersize=2.5,
                linewidth=1,
                label=row["case"]
            )
        ax.set_title(f"phase profile at cycle phase {q:.2f}")
        ax.set_xlabel("phase index")
        ax.set_ylabel("unwrapped phi")

    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Modal-stroke overlay after cycle-phase normalization", y=1.02)
    savefig(fig, "fig06_phase_normalized_overlay_v2")


def main():
    all_df = load_all_summary()
    picks = load_picks(all_df)

    manifest = FIG_DIR / "figure_manifest.txt"
    manifest.write_text(
        "\n".join([
            "N=12 Phase 6 notes figures, v2",
            "",
            "Main fixes relative to v1:",
            "- Tip paths are not forcibly closed.",
            "- Tip panels include raw open paths and drift-removed open stroke paths.",
            f"- Reward, cumulative reward, phi heatmaps, and action heatmaps show {NCYCLES_SHOW} repeated cycles.",
            "",
            "Figures:",
            "fig01_summary_metrics",
            "fig02_modal_stroke_gallery_v2",
            "fig03_pi40_seed_gallery_v2",
            "fig04_commensurate_cycle_gallery_v2",
            "fig05_action_phase_reward_maps_v2",
            "fig06_phase_normalized_overlay_v2",
            "",
            "Seed picks used:",
            picks.to_string(index=False),
            "",
        ])
    )

    plot_summary_metrics(all_df)
    plot_modal_gallery(picks)
    plot_pi40_gallery(picks)
    plot_commensurate_gallery(picks)
    plot_action_maps(picks)
    plot_phase_normalized_overlay(picks)

    print("\nAll v2 figures written to:")
    print(FIG_DIR)
    print("\nManifest:")
    print(manifest)


if __name__ == "__main__":
    main()
