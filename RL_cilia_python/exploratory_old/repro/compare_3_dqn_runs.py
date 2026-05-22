import numpy as np
import os
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
        "length": best["length"],
        "avg_reward": best["avg_reward"],
        "cycle": best["cycle"],
        "actions": best.get("actions", None),
    }


def pairwise_policy_diff(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    if p1.shape != p2.shape:
        return None, None
    diff_mask = (p1 != p2)
    n_diff = int(np.sum(diff_mask))
    idx = np.argwhere(diff_mask)
    return n_diff, idx


def print_if_same_dict(name, dicts, labels):
    print(f"\n{name}:")
    keys = sorted(set().union(*[set(d.keys()) for d in dicts]))
    all_same = True
    for k in keys:
        vals = [d.get(k, "<missing>") for d in dicts]
        same = all(v == vals[0] for v in vals[1:])
        flag = "same" if same else "DIFF"
        if not same:
            all_same = False
        print(f"  {k}: {flag}")
        for lab, v in zip(labels, vals):
            print(f"    {lab} = {v}")
    if all_same:
        print(f"  All {name.lower()} entries agree across runs.")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # EDIT THESE PATHS
    # ------------------------------------------------------------
    paths = [
        "dqn_cilia_2_ball_results_01.npy",
        "dqn_cilia_2_ball_results_02.npy",
        "dqn_cilia_2_ball_results_03.npy",
    ]

    labels = [f"run{k+1}" for k in range(len(paths))]
    runs = [load_dqn_result(p) for p in paths]

    print("=" * 80)
    print("Comparing DQN reruns")
    for lab, p in zip(labels, paths):
        print(f"  {lab}: {os.path.abspath(p)}")

    # ------------------------------------------------------------
    # settings
    # ------------------------------------------------------------
    train_settings = [r.get("train_settings", {}) for r in runs]
    env_settings = [r.get("env_settings", {}) for r in runs]

    print_if_same_dict("Train settings", train_settings, labels)
    print_if_same_dict("Env settings", env_settings, labels)

    # ------------------------------------------------------------
    # best cycle summaries
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST CYCLE SUMMARIES")
    cycle_summaries = [best_cycle_summary(r) for r in runs]

    for lab, cs in zip(labels, cycle_summaries):
        print(f"\n{lab}:")
        if cs is None:
            print("  No saved cycles.")
        else:
            print("  length     =", cs["length"])
            print("  avg reward =", cs["avg_reward"])
            print("  cycle      =", cs["cycle"])

    # pairwise cycle comparisons
    print("\n" + "=" * 80)
    print("PAIRWISE CYCLE COMPARISONS")
    for (i, j) in combinations(range(len(runs)), 2):
        cs1 = cycle_summaries[i]
        cs2 = cycle_summaries[j]
        print(f"\n{labels[i]} vs {labels[j]}:")
        if cs1 is None or cs2 is None:
            print("  At least one run has no cycle.")
            continue
        print("  match up to cyclic shift:",
              cycles_match_up_to_shift(cs1["cycle"], cs2["cycle"]))
        print("  exact action sequence equal:",
              cs1["actions"] == cs2["actions"])
        print("  lengths:", cs1["length"], cs2["length"])
        print("  avg rewards:", cs1["avg_reward"], cs2["avg_reward"])

    # ------------------------------------------------------------
    # policy comparisons
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PAIRWISE POLICY DIFFERENCES")
    policies = [r.get("policy", None) for r in runs]

    for (i, j) in combinations(range(len(runs)), 2):
        print(f"\n{labels[i]} vs {labels[j]}:")
        p1 = policies[i]
        p2 = policies[j]
        if p1 is None or p2 is None:
            print("  At least one run has no saved policy.")
            continue

        n_diff, idx = pairwise_policy_diff(p1, p2)
        if n_diff is None:
            print("  Policy shapes differ:",
                  np.asarray(p1).shape, np.asarray(p2).shape)
            continue

        print("  number of differing states:", n_diff)
        if n_diff > 0:
            print("  first 15 differing states:")
            for (ii, jj) in idx[:15]:
                print(f"    state ({ii},{jj}): {labels[i]}={p1[ii,jj]}, {labels[j]}={p2[ii,jj]}")

    # ------------------------------------------------------------
    # simple diagnostic summaries
    # ------------------------------------------------------------
    def diag_summary(diag):
        out = {}
        if "episode_rewards" in diag:
            er = np.asarray(diag["episode_rewards"], dtype=float)
            out["last50_reward_mean"] = float(np.mean(er[-50:])) if len(er) >= 50 else float(np.mean(er))
            out["final_reward"] = float(er[-1]) if len(er) > 0 else np.nan
        if "episode_lengths" in diag:
            el = np.asarray(diag["episode_lengths"], dtype=float)
            out["last50_length_mean"] = float(np.mean(el[-50:])) if len(el) >= 50 else float(np.mean(el))
        if "loss_history" in diag:
            lh = np.asarray(diag["loss_history"], dtype=float)
            out["last100_loss_mean"] = float(np.mean(lh[-100:])) if len(lh) >= 100 else (float(np.mean(lh)) if len(lh) > 0 else np.nan)
        if "q_means" in diag:
            qm = np.asarray(diag["q_means"], dtype=float)
            out["last50_qmean"] = float(np.mean(qm[-50:])) if len(qm) >= 50 else (float(np.mean(qm)) if len(qm) > 0 else np.nan)
        return out

    diag_summaries = [diag_summary(r.get("diagnostics", {})) for r in runs]

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARIES")
    print_if_same_dict("Diagnostics summary", diag_summaries, labels)