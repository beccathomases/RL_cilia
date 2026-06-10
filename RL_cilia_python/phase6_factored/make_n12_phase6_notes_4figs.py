from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/ppo_sweeps_phase6_n12_t6e6_radscale0p4")
FIG_DIR = ROOT / "figures_notes_4figs"
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
    table_path = ROOT / "tables" / "n12_phase6_seed_picks_for_vis.csv"
    if table_path.exists():
        picks = pd.read_csv(table_path)
    else:
        picks = make_pick_table(all_df)

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


def load_cycle(row):
    p = resolve_result_path(row)
    z = np.load(p, allow_pickle=True)

    s = int(np.asarray(z["cycle_start"]))
    L = int(np.asarray(z["cycle_length"]))
    n = len(z["rewards"])
    idx = (s + np.arange(L)) % n

    return {
        "path": p,
        "cycle_start": s,
        "cycle_length": L,
        "phi": np.asarray(z["phi"])[idx],
        "tip": np.asarray(z["tip"])[idx],
        "rewards": np.asarray(z["rewards"])[idx],
        "actions": np.asarray(z["actions"])[idx],
    }


def tile_cycle(a, ncycles=NCYCLES_SHOW):
    return np.concatenate([a] * ncycles, axis=0)


def add_cycle_lines(ax, L, ncycles=NCYCLES_SHOW):
    for k in range(1, ncycles):
        ax.axvline(k * L - 0.5, linewidth=0.8)
    ax.set_xlim(0, ncycles * L - 1)


def detrend_tip_open_path(tip):
    tip = np.asarray(tip, dtype=float)
    if len(tip) <= 1:
        return tip.copy()
    alpha = np.linspace(0.0, 1.0, len(tip))[:, None]
    drift = alpha * (tip[-1] - tip[0])[None, :]
    out = tip - drift
    out = out - out.mean(axis=0, keepdims=True)
    return out


def plot_tip_path(ax, cyc, title=None):
    tip = detrend_tip_open_path(cyc["tip"])
    x = tip[:, 0]
    y = tip[:, 1]

    ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2)
    ax.scatter(x[0], y[0], marker="s", s=28)
    ax.scatter(x[-1], y[-1], marker="x", s=40)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("drift-removed tip x")
    ax.set_ylabel("drift-removed tip y")
    if title:
        ax.set_title(title)


def plot_cumulative_spatial_angles(ax, cyc, title=None):
    phi = tile_cycle(cyc["phi"])
    theta = np.cumsum(phi, axis=1)
    t = np.arange(theta.shape[0])
    L = cyc["cycle_length"]

    for j in range(theta.shape[1]):
        ax.plot(t, theta[:, j], linewidth=1.0, alpha=0.9)

    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Theta_j=\sum_{m\leq j}\phi_m$")
    if title:
        ax.set_title(title)


def plot_reward(ax, cyc, title=None):
    r = tile_cycle(cyc["rewards"])
    t = np.arange(len(r))
    L = cyc["cycle_length"]

    ax.plot(t, r, marker="o", markersize=2.3, linewidth=1.0)
    ax.axhline(0, linewidth=0.8)
    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    if title:
        ax.set_title(title)


def plot_cumulative_reward(ax, cyc, title=None):
    r = tile_cycle(cyc["rewards"])
    cr = np.cumsum(r)
    cr = cr - cr[0]
    t = np.arange(len(cr))
    L = cyc["cycle_length"]

    ax.plot(t, cr, linewidth=1.2)
    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative reward")
    if title:
        ax.set_title(title)


def fig01_summary_metrics(all_df):
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
        ax.boxplot(data, tick_labels=CASES, showmeans=True)

        for i, vals in enumerate(data, start=1):
            x = np.full_like(vals, i, dtype=float)
            if len(vals) > 1:
                x += np.linspace(-0.08, 0.08, len(vals))
            ax.plot(x, vals, "o", markersize=3)

        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle("N=12 Phase 6 factored PPO: summary metrics", y=1.02)
    savefig(fig, "fig01_summary_metrics")


def fig02_modal_strokes_summary(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)

    for i, case in enumerate(CASES):
        row = modal[modal["case"] == case].iloc[0]
        cyc = load_cycle(row)
        label = f"{case} seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_path(axes[i, 0], cyc, title=f"{label}\ndrift-removed tip path")
        plot_cumulative_spatial_angles(axes[i, 1], cyc, title=f"cumulative spatial angles, {NCYCLES_SHOW} cycles")
        plot_reward(axes[i, 2], cyc, title=f"reward, {NCYCLES_SHOW} cycles")
        plot_cumulative_reward(axes[i, 3], cyc, title=f"cumulative reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("Modal strokes across angular refinements", y=1.02)
    savefig(fig, "fig02_modal_strokes_summary")


def fig03_commensurate_summary(picks):
    rowspec = [
        ("pi20", "modal_L_median"),
        ("pi20", "longest_cycle"),
        ("pi30", "modal_L_median"),
        ("pi30", "longest_cycle"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(15, 11), constrained_layout=True)

    for i, (case, pick_label) in enumerate(rowspec):
        row = picks[(picks["case"] == case) & (picks["pick_label"] == pick_label)].iloc[0]
        cyc = load_cycle(row)
        label = f"{case} {pick_label}, seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_path(axes[i, 0], cyc, title=f"{label}\ndrift-removed tip path")
        plot_cumulative_spatial_angles(axes[i, 1], cyc, title=f"cumulative spatial angles, {NCYCLES_SHOW} cycles")
        plot_reward(axes[i, 2], cyc, title=f"reward, {NCYCLES_SHOW} cycles")
        plot_cumulative_reward(axes[i, 3], cyc, title=f"cumulative reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("Commensurate-cycle examples: modal vs repeated strokes", y=1.01)
    savefig(fig, "fig03_commensurate_summary")


def fig04_pi40_robustness_summary(picks):
    labels = ["worst_mean", "median_mean", "best_mean", "longest_cycle"]
    sub = picks[(picks["case"] == "pi40") & (picks["pick_label"].isin(labels))].copy()

    order = {lab: i for i, lab in enumerate(labels)}
    sub["order"] = sub["pick_label"].map(order)
    sub = sub.sort_values("order")

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), constrained_layout=True)

    for i, (_, row) in enumerate(sub.iterrows()):
        cyc = load_cycle(row)
        label = f"{row['pick_label']}, seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_cumulative_spatial_angles(axes[i, 0], cyc, title=f"{label}\ncumulative spatial angles")
        plot_reward(axes[i, 1], cyc, title=f"reward, {NCYCLES_SHOW} cycles")
        plot_cumulative_reward(axes[i, 2], cyc, title=f"cumulative reward, {NCYCLES_SHOW} cycles")

    fig.suptitle("pi40 robustness across selected seeds", y=1.01)
    savefig(fig, "fig04_pi40_robustness_summary")


def main():
    all_df = load_all_summary()
    picks = load_picks(all_df)

    fig01_summary_metrics(all_df)
    fig02_modal_strokes_summary(picks)
    fig03_commensurate_summary(picks)
    fig04_pi40_robustness_summary(picks)

    manifest = FIG_DIR / "figure_manifest.txt"
    manifest.write_text(
        "\n".join([
            "N=12 Phase 6 notes: compact 4-figure set",
            "",
            "fig01_summary_metrics",
            "fig02_modal_strokes_summary",
            "fig03_commensurate_summary",
            "fig04_pi40_robustness_summary",
            "",
            f"All time-series style plots show {NCYCLES_SHOW} tiled cycles.",
            "",
            "Seed picks used:",
            picks.to_string(index=False),
            "",
        ])
    )

    print("\nAll figures written to:")
    print(FIG_DIR)
    print("\nManifest:")
    print(manifest)


if __name__ == "__main__":
    main()
