from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/ppo_sweeps_phase6_n12_t6e6_radscale0p4")
TABLE_DIR = ROOT / "tables_notes"
TABLE_DIR.mkdir(exist_ok=True, parents=True)

CASES = ["pi20", "pi30", "pi40"]
DENOMS = {"pi20": 20, "pi30": 30, "pi40": 40}

RENAME = {
    "cycle_mean_reward": "mean_reward",
    "cycle_total_reward": "cycle_reward",
    "tip_abs_area": "tip_area",
    "reward_per_abs_tip_area": "rho_reward_per_area",
    "wrong_orientation_flag": "wrong_orientation",
}


def load_all_summary():
    rows = []
    for case in CASES:
        path = ROOT / f"N12_dtheta_{case}_factored" / "summary.csv"
        df = pd.read_csv(path).rename(columns=RENAME)
        df["case"] = case
        df["dtheta"] = np.pi / DENOMS[case]
        df["mean_reward_over_dtheta"] = df["mean_reward"] / df["dtheta"]
        df["cycle_phase"] = df["cycle_length"] * df["dtheta"]
        df["cycle_phase_over_pi2"] = df["cycle_phase"] / (np.pi / 2)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_picks(all_df):
    path = ROOT / "tables" / "n12_phase6_seed_picks_for_vis.csv"
    if path.exists():
        picks = pd.read_csv(path)
    else:
        raise FileNotFoundError("Run the seed-pick/table script first.")

    if "dtheta" not in picks.columns:
        picks["dtheta"] = picks["case"].map(lambda c: np.pi / DENOMS[c])
    if "cycle_phase_over_pi2" not in picks.columns:
        picks["cycle_phase_over_pi2"] = picks["cycle_length"] * picks["dtheta"] / (np.pi / 2)

    return picks


def fmt_range(df, col, nd=3):
    med = df[col].median()
    lo = df[col].min()
    hi = df[col].max()
    return f"{med:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    all_df = load_all_summary()
    picks = load_picks(all_df)

    # Table 1: aggregate summary.
    rows = []
    for case in CASES:
        df = all_df[all_df["case"] == case]
        rows.append({
            "case": case,
            "runs": len(df),
            "cycle lengths": ", ".join(map(str, sorted(df["cycle_length"].astype(int).unique()))),
            "mean reward / dtheta": fmt_range(df, "mean_reward_over_dtheta", 2),
            "cycle reward": fmt_range(df, "cycle_reward", 2),
            "tip area": fmt_range(df, "tip_area", 3),
            "tip path length": fmt_range(df, "tip_path_length", 3),
            "reward / area": fmt_range(df, "rho_reward_per_area", 2),
            "cycle phase/(pi/2)": fmt_range(df, "cycle_phase_over_pi2", 2),
            "wrong orientation": int(df["wrong_orientation"].astype(bool).sum()),
        })
    table1 = pd.DataFrame(rows)

    # Table 2: modal seeds used in Fig. 2.
    modal = picks[picks["pick_label"] == "modal_L_median"].copy()
    table2 = modal[[
        "case", "seed", "cycle_length", "cycle_phase_length",
        "mean_reward", "cycle_reward", "tip_area",
        "tip_path_length", "rho_reward_per_area"
    ]].copy()
    table2 = table2.rename(columns={
        "cycle_length": "L",
        "cycle_phase_length": "phase length",
        "mean_reward": "mean reward",
        "cycle_reward": "cycle reward",
        "tip_area": "tip area",
        "tip_path_length": "tip path",
        "rho_reward_per_area": "reward/area",
    })

    # Table 3: commensurate interpretation.
    comm_specs = [
        ("pi20", "modal_L_median", "modal branch"),
        ("pi20", "longest_cycle", "2x repeat of L=10 branch"),
        ("pi30", "modal_L_median", "modal branch"),
        ("pi30", "longest_cycle", "3x repeat of L=14 branch"),
    ]
    comm_rows = []
    for case, label, interpretation in comm_specs:
        row = picks[(picks["case"] == case) & (picks["pick_label"] == label)].iloc[0]
        comm_rows.append({
            "case": case,
            "seed": int(row["seed"]),
            "pick": label,
            "L": int(row["cycle_length"]),
            "phase length": row["cycle_phase_length"],
            "cycle reward": row["cycle_reward"],
            "tip area": row["tip_area"],
            "tip path": row["tip_path_length"],
            "reward/area": row["rho_reward_per_area"],
            "interpretation": interpretation,
        })
    table3 = pd.DataFrame(comm_rows)

    # Add ratios relative to modal branch within each case.
    for case in ["pi20", "pi30"]:
        base = table3[(table3["case"] == case) & (table3["pick"] == "modal_L_median")].iloc[0]
        mask = table3["case"] == case
        for col in ["L", "cycle reward", "tip area", "tip path"]:
            table3.loc[mask, f"{col} / modal"] = table3.loc[mask, col] / base[col]

    # Table 4: pi40 robustness.
    order = ["worst_mean", "median_mean", "best_mean", "longest_cycle"]
    pi40 = picks[(picks["case"] == "pi40") & (picks["pick_label"].isin(order))].copy()
    pi40["order"] = pi40["pick_label"].map({k: i for i, k in enumerate(order)})
    pi40 = pi40.sort_values("order")
    table4 = pi40[[
        "pick_label", "seed", "cycle_length", "cycle_phase_length",
        "mean_reward", "cycle_reward", "tip_area",
        "tip_path_length", "rho_reward_per_area"
    ]].copy()
    table4 = table4.rename(columns={
        "pick_label": "pick",
        "cycle_length": "L",
        "cycle_phase_length": "phase length",
        "mean_reward": "mean reward",
        "cycle_reward": "cycle reward",
        "tip_area": "tip area",
        "tip_path_length": "tip path",
        "rho_reward_per_area": "reward/area",
    })

    # Save CSVs.
    tables = {
        "table1_aggregate_summary": table1,
        "table2_modal_seed_metrics": table2,
        "table3_commensurate_interpretation": table3,
        "table4_pi40_robustness": table4,
    }

    for name, df in tables.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)

        tex = df.copy()
        for c in tex.columns:
            if pd.api.types.is_numeric_dtype(tex[c]):
                if c in ["runs", "seed", "L", "wrong orientation"]:
                    tex[c] = tex[c].map(lambda x: f"{int(x)}")
                else:
                    tex[c] = tex[c].map(lambda x: f"{x:.3g}")
        tex.to_latex(TABLE_DIR / f"{name}.tex", index=False)

    print("\nWrote tables to:")
    print(TABLE_DIR)

    for name, df in tables.items():
        print("\n" + "="*80)
        print(name)
        print("="*80)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
