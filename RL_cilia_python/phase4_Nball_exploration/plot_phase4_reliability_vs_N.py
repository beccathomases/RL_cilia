#!/usr/bin/env python3
"""
plot_phase4_reliability_vs_N.py
===============================

Make Phase 4 reliability-vs-N figures using full 1M-only grouped summary.

Input:
    results/ppo_sweeps_general/phase4_group_summary_full_1M_only.csv

Outputs:
    figures/ppo_sweeps_general/reliability_vs_N/reliability_vs_N_full_range.png
    figures/ppo_sweeps_general/reliability_vs_N/reliability_vs_N_full_range.pdf
    figures/ppo_sweeps_general/reliability_vs_N/reliability_vs_N_zoom.png
    figures/ppo_sweeps_general/reliability_vs_N/reliability_vs_N_zoom.pdf

Plot:
    x-axis: N
    y-axis: PPO percent of Howard
    points: median recovery
    whiskers: min to max recovery
    two series: pi20 and pi30
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INFILE = Path("results/ppo_sweeps_general/phase4_group_summary_full_1M_only.csv")
OUTDIR = Path("figures/ppo_sweeps_general/reliability_vs_N")


def load_data():
    if not INFILE.exists():
        raise FileNotFoundError(f"Could not find {INFILE}")

    df = pd.read_csv(INFILE)

    required = [
        "N",
        "dtheta",
        "PPO_percent_median",
        "PPO_percent_min",
        "PPO_percent_max",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INFILE}: {missing}")

    return df


def make_plot(df, out_stem, ylim=None, annotate_outliers=False):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    dtheta_order = ["pi20", "pi30"]
    label_map = {
        "pi20": r"$\Delta\theta=\pi/20$",
        "pi30": r"$\Delta\theta=\pi/30$",
    }
    offsets = {
        "pi20": -0.06,
        "pi30":  0.06,
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for dtheta in dtheta_order:
        g = df[df["dtheta"] == dtheta].copy()
        if g.empty:
            print(f"[warn] no rows for {dtheta}")
            continue

        g = g.sort_values("N")

        x = g["N"].to_numpy(dtype=float) + offsets[dtheta]
        med = g["PPO_percent_median"].to_numpy(dtype=float)
        ymin = g["PPO_percent_min"].to_numpy(dtype=float)
        ymax = g["PPO_percent_max"].to_numpy(dtype=float)

        yerr = np.vstack([med - ymin, ymax - med])

        ax.errorbar(
            x,
            med,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=5,
            linewidth=1.8,
            label=label_map.get(dtheta, dtheta),
        )

    ax.axhline(100, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("PPO recovery of Howard gain (%)")
    ax.set_title("PPO recovery relative to Howard benchmark")
    ax.set_xticks([2, 3, 4])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        bottom = min(-50, float(df["PPO_percent_min"].min()) - 10)
        ax.set_ylim(bottom=bottom, top=105)

    if annotate_outliers:
        ax.text(
            2.15,
            55,
            "N=2 has negative\noutlier seeds",
            fontsize=10,
            va="bottom",
        )

        # Small visual cue that lower whiskers are clipped in zoomed view.
        ax.text(
            2.0,
            51,
            "lower tails clipped",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    png_path = OUTDIR / f"{out_stem}.png"
    pdf_path = OUTDIR / f"{out_stem}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    df = load_data()

    # Honest diagnostic version: includes the large negative N=2 outliers.
    make_plot(
        df,
        out_stem="reliability_vs_N_full_range",
        ylim=None,
        annotate_outliers=False,
    )

    # Main-text version: zooms in on the high-recovery region.
    make_plot(
        df,
        out_stem="reliability_vs_N_zoom",
        ylim=(50, 105),
        annotate_outliers=True,
    )


if __name__ == "__main__":
    main()