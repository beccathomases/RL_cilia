from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results/ppo_sweeps_phase6_n12_t6e6_radscale0p4")
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures_angles_4cycles"
FIG_DIR.mkdir(exist_ok=True, parents=True)

CASES = ["pi20", "pi30", "pi40"]
DENOMS = {"pi20": 20, "pi30": 30, "pi40": 40}
NCYCLES_SHOW = 4


def savefig(fig, name):
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)
    print("wrote", png)


def load_picks():
    path = TABLE_DIR / "n12_phase6_seed_picks_for_vis.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run make_n12_phase6_tables.py first."
        )
    picks = pd.read_csv(path)
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
    candidates += list(Path(".").glob(
        f"results/**/N12_dtheta_{case}_factored/seed_{seed:03d}/result.npz"
    ))

    for q in candidates:
        if q.exists():
            return q

    raise FileNotFoundError(f"Could not find result.npz for {case} seed {seed}")


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
        "dtheta": float(row["dtheta"]),
    }


def tile_cycle(a, ncycles=NCYCLES_SHOW):
    return np.concatenate([a] * ncycles, axis=0)


def make_time_axis(cyc, ncycles=NCYCLES_SHOW):
    L = cyc["cycle_length"]
    return np.arange(ncycles * L), L


def add_cycle_lines(ax, L, ncycles=NCYCLES_SHOW):
    for k in range(1, ncycles):
        ax.axvline(k * L - 0.5, linewidth=0.8)
    ax.set_xlim(0, ncycles * L - 1)


def plot_local_angles(ax, cyc, title=None):
    phi = tile_cycle(cyc["phi"])
    t, L = make_time_axis(cyc)

    N = phi.shape[1]
    for j in range(N):
        label = f"j={j}" if j in [0, N//4, N//2, 3*N//4, N-1] else None
        ax.plot(t, phi[:, j], linewidth=1.0, alpha=0.85, label=label)

    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel(r"local angle $\phi_j$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")


def plot_cumulative_spatial_angles(ax, cyc, title=None):
    phi = tile_cycle(cyc["phi"])
    theta = np.cumsum(phi, axis=1)
    t, L = make_time_axis(cyc)

    N = theta.shape[1]
    for j in range(N):
        label = f"j={j}" if j in [0, N//4, N//2, 3*N//4, N-1] else None
        ax.plot(t, theta[:, j], linewidth=1.0, alpha=0.85, label=label)

    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel(r"cumulative angle $\Theta_j=\sum_{m\leq j}\phi_m$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")


def plot_cumulative_action_angles(ax, cyc, title=None):
    """
    This is different from cumulative spatial angle.

    It plots the integrated commanded angle changes through time:
        A_j(t) = sum_s action_j(s) * dtheta.

    Useful for seeing what the policy is doing to each coordinate.
    """
    actions = tile_cycle(cyc["actions"])
    dtheta = cyc["dtheta"]
    cumcmd = np.cumsum(actions * dtheta, axis=0)
    t, L = make_time_axis(cyc)

    N = cumcmd.shape[1]
    for j in range(N):
        label = f"j={j}" if j in [0, N//4, N//2, 3*N//4, N-1] else None
        ax.plot(t, cumcmd[:, j], linewidth=1.0, alpha=0.85, label=label)

    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel(r"cumulative command $\sum a_j\Delta\theta$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")


def plot_reward(ax, cyc, title=None):
    r = tile_cycle(cyc["rewards"])
    t, L = make_time_axis(cyc)

    ax.plot(t, r, marker="o", markersize=2.5, linewidth=1.0)
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
    t, L = make_time_axis(cyc)

    ax.plot(t, cr, linewidth=1.2)
    add_cycle_lines(ax, L)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative reward")
    if title:
        ax.set_title(title)


def plot_angle_summary(rows, name, suptitle):
    fig, axes = plt.subplots(len(rows), 5, figsize=(19, 3.3 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]

    for i, (_, row) in enumerate(rows.iterrows()):
        cyc = load_cycle(row)
        label = (
            f"{row['case']} {row['pick_label']}, "
            f"seed {int(row['seed'])}, L={cyc['cycle_length']}"
        )

        plot_local_angles(
            axes[i, 0],
            cyc,
            title=f"{label}\nlocal angles, {NCYCLES_SHOW} cycles"
        )
        plot_cumulative_spatial_angles(
            axes[i, 1],
            cyc,
            title=f"cumulative spatial angles, {NCYCLES_SHOW} cycles"
        )
        plot_cumulative_action_angles(
            axes[i, 2],
            cyc,
            title=f"cumulative action commands, {NCYCLES_SHOW} cycles"
        )
        plot_reward(
            axes[i, 3],
            cyc,
            title=f"reward, {NCYCLES_SHOW} cycles"
        )
        plot_cumulative_reward(
            axes[i, 4],
            cyc,
            title=f"cumulative reward, {NCYCLES_SHOW} cycles"
        )

    fig.suptitle(suptitle, y=1.01)
    savefig(fig, name)


def plot_case_overlay_modal(picks):
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)

    for i, case in enumerate(CASES):
        row = modal[modal["case"] == case].iloc[0]
        cyc = load_cycle(row)
        label = f"{case} seed {int(row['seed'])}, L={cyc['cycle_length']}"

        plot_local_angles(axes[i, 0], cyc, title=f"{label}\nlocal angles")
        plot_cumulative_spatial_angles(axes[i, 1], cyc, title="cumulative spatial angles")
        plot_reward(axes[i, 2], cyc, title="reward")

    fig.suptitle("Modal strokes: angles and reward over four cycles", y=1.02)
    savefig(fig, "fig07_modal_angles_reward_4cycles")


def main():
    picks = load_picks()

    modal = picks[picks["pick_label"] == "modal_L_median"].copy()
    plot_angle_summary(
        modal,
        "fig07_modal_angle_diagnostics_4cycles",
        "N=12 Phase 6 modal strokes: local angles, cumulative angles, actions, reward"
    )

    comm = picks[
        ((picks["case"] == "pi20") & (picks["pick_label"].isin(["modal_L_median", "longest_cycle"])))
        |
        ((picks["case"] == "pi30") & (picks["pick_label"].isin(["modal_L_median", "longest_cycle"])))
    ].copy()
    plot_angle_summary(
        comm,
        "fig08_commensurate_angle_diagnostics_4cycles",
        "N=12 Phase 6 commensurate examples: modal vs repeated cycles"
    )

    pi40 = picks[
        (picks["case"] == "pi40")
        & (picks["pick_label"].isin(["worst_mean", "median_mean", "best_mean", "longest_cycle"]))
    ].copy()
    plot_angle_summary(
        pi40,
        "fig09_pi40_angle_diagnostics_4cycles",
        "N=12 Phase 6 pi40 selected seeds: angle diagnostics"
    )

    plot_case_overlay_modal(picks)

    manifest = FIG_DIR / "figure_manifest.txt"
    manifest.write_text(
        "\n".join([
            "N=12 Phase 6 angle figures over four cycles",
            "",
            "Columns in main diagnostic figures:",
            "1. local angles phi_j(t)",
            "2. cumulative spatial angles Theta_j(t)=sum_{m<=j} phi_m(t)",
            "3. cumulative action commands sum_t action_j(t)*dtheta",
            "4. reward over four cycles",
            "5. cumulative reward over four cycles",
            "",
            "Figures:",
            "fig07_modal_angle_diagnostics_4cycles",
            "fig08_commensurate_angle_diagnostics_4cycles",
            "fig09_pi40_angle_diagnostics_4cycles",
            "fig07_modal_angles_reward_4cycles",
            "",
            "Seed picks:",
            picks.to_string(index=False),
            "",
        ])
    )

    print("\nAll angle figures written to:")
    print(FIG_DIR)
    print("\nManifest:")
    print(manifest)


if __name__ == "__main__":
    main()
