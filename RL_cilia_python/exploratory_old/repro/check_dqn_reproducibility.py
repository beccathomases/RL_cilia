import os
import csv
import hashlib
import numpy as np
from itertools import combinations


def load_dqn_result(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not load as a dict-like DQN result.")
    return obj


def canonical_cycle(cycle):
    cyc = [tuple(map(int, s)) for s in cycle]
    if len(cyc) == 0:
        return tuple()
    rots = [tuple(cyc[k:] + cyc[:k]) for k in range(len(cyc))]
    return min(rots)


def cycles_match_up_to_shift(c1, c2):
    return canonical_cycle(c1) == canonical_cycle(c2)


def best_cycle_summary(run):
    cycles = run.get("cycles", [])
    if isinstance(cycles, np.ndarray):
        cycles = list(cycles)
    if len(cycles) == 0:
        return None
    best = cycles[0]
    return {
        "length": int(best["length"]),
        "avg_reward": float(best["avg_reward"]),
        "cycle": list(best["cycle"]),
        "actions": list(best.get("actions", [])),
    }


def array_equal_safe(a, b):
    if a is None or b is None:
        return False
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)


def array_hash(a):
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def diagnostics_equal(diag1, diag2, key):
    if key not in diag1 or key not in diag2:
        return False, None, None
    a = np.asarray(diag1[key])
    b = np.asarray(diag2[key])
    return np.array_equal(a, b), a, b


if __name__ == "__main__":
    # ------------------------------------------------------------
    # EDIT THESE FILENAMES
    # ------------------------------------------------------------
    paths = [
        "dqn_cilia_2_ball_results_01.npy",
        "dqn_cilia_2_ball_results_02.npy",
        "dqn_cilia_2_ball_results_03.npy",
        "dqn_cilia_2_ball_results_04.npy",
    ]

    outdir = "dqn_repro_check"
    os.makedirs(outdir, exist_ok=True)

    summary_txt = os.path.join(outdir, "repro_summary.txt")
    pairwise_csv = os.path.join(outdir, "repro_pairwise.csv")
    filehash_csv = os.path.join(outdir, "repro_hashes.csv")

    # only keep files that exist
    paths = [p for p in paths if os.path.exists(p)]
    labels = [f"run{k+1}" for k in range(len(paths))]

    if len(paths) < 2:
        raise ValueError("Need at least two existing result files to compare.")

    runs = [load_dqn_result(p) for p in paths]
    cycle_summaries = [best_cycle_summary(r) for r in runs]
    policies = [r.get("policy", None) for r in runs]
    diags = [r.get("diagnostics", {}) for r in runs]
    train_settings = [r.get("train_settings", {}) for r in runs]
    env_settings = [r.get("env_settings", {}) for r in runs]

    # ------------------------------------------------------------
    # Save hashes of key arrays
    # ------------------------------------------------------------
    hash_rows = []
    for lab, path, run in zip(labels, paths, runs):
        row = {
            "label": lab,
            "path": os.path.abspath(path),
        }

        if run.get("policy", None) is not None:
            row["policy_hash"] = array_hash(run["policy"])
        else:
            row["policy_hash"] = ""

        for key in ["episode_rewards", "episode_lengths", "loss_history", "q_means"]:
            if key in run.get("diagnostics", {}):
                row[f"{key}_hash"] = array_hash(run["diagnostics"][key])
            else:
                row[f"{key}_hash"] = ""

        hash_rows.append(row)

    with open(filehash_csv, "w", newline="") as f:
        fieldnames = list(hash_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hash_rows)

    # ------------------------------------------------------------
    # Pairwise comparisons
    # ------------------------------------------------------------
    pairwise_rows = []

    for (i, j) in combinations(range(len(runs)), 2):
        row = {
            "run_i": labels[i],
            "run_j": labels[j],
            "path_i": os.path.abspath(paths[i]),
            "path_j": os.path.abspath(paths[j]),
        }

        # settings
        row["train_settings_equal"] = (train_settings[i] == train_settings[j])
        row["env_settings_equal"] = (env_settings[i] == env_settings[j])

        # best cycles
        ci = cycle_summaries[i]
        cj = cycle_summaries[j]

        if ci is None or cj is None:
            row["best_cycle_exists_both"] = False
            row["cycle_match_up_to_shift"] = False
            row["cycle_length_equal"] = False
            row["cycle_avg_reward_equal"] = False
            row["cycle_actions_equal"] = False
        else:
            row["best_cycle_exists_both"] = True
            row["cycle_match_up_to_shift"] = cycles_match_up_to_shift(ci["cycle"], cj["cycle"])
            row["cycle_length_equal"] = (ci["length"] == cj["length"])
            row["cycle_avg_reward_equal"] = np.isclose(ci["avg_reward"], cj["avg_reward"])
            row["cycle_actions_equal"] = (ci["actions"] == cj["actions"])
            row["cycle_length_i"] = ci["length"]
            row["cycle_length_j"] = cj["length"]
            row["cycle_avg_reward_i"] = ci["avg_reward"]
            row["cycle_avg_reward_j"] = cj["avg_reward"]

        # policy
        if policies[i] is None or policies[j] is None:
            row["policy_equal"] = False
            row["policy_n_diff"] = ""
        else:
            pi = np.asarray(policies[i])
            pj = np.asarray(policies[j])
            if pi.shape != pj.shape:
                row["policy_equal"] = False
                row["policy_n_diff"] = "shape_mismatch"
            else:
                diff_mask = (pi != pj)
                n_diff = int(np.sum(diff_mask))
                row["policy_equal"] = (n_diff == 0)
                row["policy_n_diff"] = n_diff

        # diagnostics arrays
        for key in ["episode_rewards", "episode_lengths", "loss_history", "q_means"]:
            eq, _, _ = diagnostics_equal(diags[i], diags[j], key)
            row[f"{key}_equal"] = eq

        pairwise_rows.append(row)

    with open(pairwise_csv, "w", newline="") as f:
        fieldnames = sorted(set().union(*[set(r.keys()) for r in pairwise_rows]))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    # ------------------------------------------------------------
    # Write human-readable summary
    # ------------------------------------------------------------
    with open(summary_txt, "w") as f:
        f.write("DQN reproducibility check\n")
        f.write("=" * 72 + "\n\n")

        f.write("Compared files:\n")
        for lab, path in zip(labels, paths):
            f.write(f"  {lab}: {os.path.abspath(path)}\n")
        f.write("\n")

        # overall settings agreement
        same_train = all(ts == train_settings[0] for ts in train_settings[1:])
        same_env = all(es == env_settings[0] for es in env_settings[1:])

        f.write(f"All train settings identical: {same_train}\n")
        f.write(f"All env settings identical:   {same_env}\n\n")

        f.write("Best cycle summaries:\n")
        for lab, cs in zip(labels, cycle_summaries):
            f.write(f"\n{lab}:\n")
            if cs is None:
                f.write("  No saved cycle.\n")
            else:
                f.write(f"  length     = {cs['length']}\n")
                f.write(f"  avg reward = {cs['avg_reward']}\n")
                f.write(f"  cycle      = {cs['cycle']}\n")

        f.write("\n" + "=" * 72 + "\n")
        f.write("Pairwise summary:\n\n")
        for row in pairwise_rows:
            f.write(f"{row['run_i']} vs {row['run_j']}:\n")
            f.write(f"  cycle_match_up_to_shift = {row.get('cycle_match_up_to_shift', False)}\n")
            f.write(f"  cycle_actions_equal     = {row.get('cycle_actions_equal', False)}\n")
            f.write(f"  policy_equal            = {row.get('policy_equal', False)}\n")
            f.write(f"  policy_n_diff           = {row.get('policy_n_diff', '')}\n")
            f.write(f"  episode_rewards_equal   = {row.get('episode_rewards_equal', False)}\n")
            f.write(f"  episode_lengths_equal   = {row.get('episode_lengths_equal', False)}\n")
            f.write(f"  loss_history_equal      = {row.get('loss_history_equal', False)}\n")
            f.write(f"  q_means_equal           = {row.get('q_means_equal', False)}\n")
            f.write("\n")

        # strongest summary line
        all_policy_equal = all(r.get("policy_equal", False) for r in pairwise_rows)
        all_cycle_equal = all(r.get("cycle_match_up_to_shift", False) for r in pairwise_rows)
        all_diag_equal = all(
            r.get("episode_rewards_equal", False)
            and r.get("episode_lengths_equal", False)
            and r.get("loss_history_equal", False)
            and r.get("q_means_equal", False)
            for r in pairwise_rows
        )

        f.write("=" * 72 + "\n")
        f.write(f"ALL POLICIES IDENTICAL: {all_policy_equal}\n")
        f.write(f"ALL BEST CYCLES MATCH UP TO SHIFT: {all_cycle_equal}\n")
        f.write(f"ALL MAIN DIAGNOSTIC ARRAYS IDENTICAL: {all_diag_equal}\n")

    print("\nSaved:")
    print(" ", summary_txt)
    print(" ", pairwise_csv)
    print(" ", filehash_csv)