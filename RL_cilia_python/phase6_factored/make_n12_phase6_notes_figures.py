from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/ppo_sweeps_phase6_n12_t6e6_radscale0p4")
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures_notes"
FIG_DIR.mkdir(exist_ok=True, parents=True)

CASES = ["pi20", "pi30", "pi40"]
DENOMS = {"pi20": 20, "pi30": 30, "pi40": 40}

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
    fig.savefig(png, dpi=200, bbox_inches="tight")
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

    # Make sure dtheta info is present.
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

    raise FileNotFoundError(f"Could not find result.npz for {case} seed {seed}. Tried {p}")


def load_cycle(row, close_tip=True):
    p = resolve_result_path(row)
    z = np.load(p, allow_pickle=True)

    s = int(np.asarray(z["cycle_start"]))
    L = int(np.asarray(z["cycle_length"]))

    def cyc(name, plus_one=False):
        a = np.asarray(z[name])
        n = len(a)
        count = L + (1 if plus_one else 0)
        idx = (s + np.arange(count)) % n
        return a[idx]

    phi = cyc("phi")
    tip = cyc("tip", plus_one=close_tip)
    rewards = cyc("rewards")
    actions = cyc("actions")

    return {
        "path": p,
        "cycle_start": s,
        "cycle_length": L,
        "phi": phi,
        "tip": tip,
        "rewards": rewards,
        "actions": actions,
    }


def centered_tip(tip):
    tip = np.asarray(tip)
    return tip - tip.mean(axis=0, keepdims=True)


def plot_tip_loop(ax, cyc, title=None, center=True):
    tip = centered_tip(cyc["tip"]) if center else cyc["tip"]
    ax.plot(tip[:, 0], tip[:, 1], marker="o", markersize=2, linewidth=1.2)
    ax.scatter(tip[0, 0], tip[0, 1], marker="s", s=25, label="start")
    ax.scatter(tip[-1, 0], tip[-1, 1], marker="x", s=35, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("tip x")
    ax.set_ylabel("tip y")
    if title:
        ax.set_title(title)


def plot_phi_heatmap(ax, cyc, title=None):
    phi = np.unwrap(cyc["phi"], axis=0)
    im = ax.imshow(phi.T, aspect="auto", origin="lower")
    ax.set_xlabel("cycle step")
    ax.set_ylabel("phase index")
    if title:
        ax.set_title(title)
    return im


def plot_action_heatmap(ax, cyc, title=None):
    actions = cyc["actions"]
    im = ax.imshow(actions.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("cycle step")
    ax.set_ylabel("phase index")
    if title:
        ax.set_title(title)
    return im


def plot_rewards(ax, cyc, title=None):
    r = cyc["rewards"]
    ax.plot(np.arange(len(r)), r, marker="o", markersize=3)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("cycle step")
    ax.set_ylabel("reward")
    if title:
        ax.set_title(title)


def plot_phi_profiles(ax, cyc, title=None, nsnaps=6):
    phi = np.unwrap(cyc["phi"], axis=0)
    L, N = phi.shape
    inds = np.linspace(0, L, nsnaps, endpoint=False, dtype=int)
    j = np.arange(N)
    for k in inds:
        ax.plot(j, phi[k], marker="o", markersize=2, linewidth=1)
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
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)

    for row_i, case in enumerate(CASES):
        row = modal[modal["case"] == case].iloc[0]
        cyc = load_cycle(row)
        title = f"{case} seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_loop(axes[row_i, 0], cyc, title=f"{title}\ntip loop")
        plot_phi_profiles(axes[row_i, 1], cyc, title="phase profiles")
        im1 = plot_phi_heatmap(axes[row_i, 2], cyc, title="phi heatmap")
        plot_rewards(axes[row_i, 3], cyc, title="reward over cycle")

    fig.suptitle("Modal-stroke gallery across angular refinements", y=1.02)
    savefig(fig, "fig02_modal_stroke_gallery")


def plot_pi40_gallery(picks):
    sub = picks[(picks["case"] == "pi40") &
                (picks["pick_label"].isin(["worst_mean", "median_mean", "best_mean", "longest_cycle"]))].copy()

    fig, axes = plt.subplots(len(sub), 4, figsize=(14, 3.1 * len(sub)), constrained_layout=True)
    if len(sub) == 1:
        axes = axes[None, :]

    for i, (_, row) in enumerate(sub.iterrows()):
        cyc = load_cycle(row)
        label = f"{row['pick_label']}: seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_loop(axes[i, 0], cyc, title=label)
        plot_phi_profiles(axes[i, 1], cyc, title="phase profiles")
        plot_phi_heatmap(axes[i, 2], cyc, title="phi heatmap")
        plot_rewards(axes[i, 3], cyc, title="reward")

    fig.suptitle("pi40 seed gallery: clean refined runs", y=1.01)
    savefig(fig, "fig03_pi40_seed_gallery")


def plot_commensurate_gallery(picks):
    targets = [
        ("pi20", "modal_L_median"),
        ("pi20", "longest_cycle"),
        ("pi30", "modal_L_median"),
        ("pi30", "longest_cycle"),
    ]

    fig, axes = plt.subplots(len(targets), 4, figsize=(14, 3.0 * len(targets)), constrained_layout=True)

    for i, (case, pick_label) in enumerate(targets):
        row = picks[(picks["case"] == case) & (picks["pick_label"] == pick_label)].iloc[0]
        cyc = load_cycle(row)
        title = f"{case} {pick_label}, seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_tip_loop(axes[i, 0], cyc, title=title)
        plot_phi_profiles(axes[i, 1], cyc, title="phase profiles")
        plot_phi_heatmap(axes[i, 2], cyc, title="phi heatmap")
        plot_rewards(axes[i, 3], cyc, title="reward")

    fig.suptitle("Commensurate-cycle comparison: modal vs repeated strokes", y=1.01)
    savefig(fig, "fig04_commensurate_cycle_gallery")


def plot_action_maps(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

    for i, case in enumerate(CASES):
        row = modal[modal["case"] == case].iloc[0]
        cyc = load_cycle(row)

        plot_action_heatmap(
            axes[i, 0],
            cyc,
            title=f"{case} seed {int(row['seed'])}, L={cyc['cycle_length']} actions"
        )
        plot_phi_heatmap(axes[i, 1], cyc, title=f"{case} phi")

    fig.suptitle("Modal policies: factored action maps and phase evolution", y=1.02)
    savefig(fig, "fig05_action_phase_maps")


def resample_periodic(y, n=200):
    y = np.asarray(y)
    old = np.linspace(0, 1, len(y), endpoint=False)
    new = np.linspace(0, 1, n, endpoint=False)

    if y.ndim == 1:
        yp = np.r_[y, y[0]]
        xp = np.r_[old, 1.0]
        return np.interp(new, xp, yp)

    out = []
    for j in range(y.shape[1]):
        yp = np.r_[y[:, j], y[0, j]]
        xp = np.r_[old, 1.0]
        out.append(np.interp(new, xp, yp))
    return np.stack(out, axis=1)


def plot_phase_normalized_overlay(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    # Panel 1: overlay centered tip loops
    ax = axes[0, 0]
    for _, row in modal.iterrows():
        cyc = load_cycle(row)
        tip = centered_tip(cyc["tip"])
        tip_rs = resample_periodic(tip, n=240)
        ax.plot(tip_rs[:, 0], tip_rs[:, 1], label=f"{row['case']} seed {int(row['seed'])}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("phase-normalized tip loops")
    ax.set_xlabel("centered tip x")
    ax.set_ylabel("centered tip y")
    ax.legend(fontsize=8)

    # Panel 2: reward vs normalized phase
    ax = axes[0, 1]
    phase = np.linspace(0, 1, 240, endpoint=False)
    for _, row in modal.iterrows():
        cyc = load_cycle(row)
        rr = resample_periodic(cyc["rewards"], n=240)
        ax.plot(phase, rr, label=row["case"])
    ax.axhline(0, linewidth=0.8)
    ax.set_title("reward vs normalized cycle phase")
    ax.set_xlabel("cycle phase")
    ax.set_ylabel("reward")

    # Panel 3: cumulative reward vs normalized phase
    ax = axes[0, 2]
    for _, row in modal.iterrows():
        cyc = load_cycle(row)
        rr = resample_periodic(cyc["rewards"], n=240)
        cr = np.cumsum(rr)
        cr = cr - cr[0]
        ax.plot(phase, cr, label=row["case"])
    ax.set_title("cumulative reward")
    ax.set_xlabel("cycle phase")
    ax.set_ylabel("cumulative reward")

    # Bottom row: phi profiles at three normalized phases
    for ax, q in zip(axes[1, :], [0.0, 0.25, 0.5]):
        for _, row in modal.iterrows():
            cyc = load_cycle(row)
            phi = np.unwrap(cyc["phi"], axis=0)
            phi_rs = resample_periodic(phi, n=240)
            k = int(q * 240)
            ax.plot(np.arange(phi_rs.shape[1]), phi_rs[k], marker="o", markersize=2, label=row["case"])
        ax.set_title(f"phase profile at cycle phase {q:.2f}")
        ax.set_xlabel("phase index")
        ax.set_ylabel("unwrapped phi")

    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Modal-stroke overlay after cycle-phase normalization", y=1.02)
    savefig(fig, "fig06_phase_normalized_overlay")


def main():
    all_df = load_all_summary()
    picks = load_picks(all_df)

    # Write a figure manifest for your notes.
    manifest = FIG_DIR / "figure_manifest.txt"
    manifest.write_text(
        "\n".join([
            "N=12 Phase 6 notes figures",
            "",
            "fig01_summary_metrics: distributions of convergence metrics across pi20/pi30/pi40.",
            "fig02_modal_stroke_gallery: modal-L median seed for each discretization: tip loop, phi profiles, phi heatmap, reward.",
            "fig03_pi40_seed_gallery: pi40 worst/median/best/longest-cycle selected seeds.",
            "fig04_commensurate_cycle_gallery: modal vs longest-cycle repeat examples for pi20 and pi30.",
            "fig05_action_phase_maps: modal action heatmaps next to phi heatmaps.",
            "fig06_phase_normalized_overlay: modal tip/reward/phase-profile overlays after cycle normalization.",
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

    print("\nAll figures written to:")
    print(FIG_DIR)
    print("\nManifest:")
    print(manifest)


if __name__ == "__main__":
    main()
