import os
import numpy as np


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


def cycle_summary(run):
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
        "rewards": best.get("rewards", None),
    }


def print_dict_diff(name, d1, d2):
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    print(f"\n{name}:")
    for k in keys:
        v1 = d1.get(k, "<missing>")
        v2 = d2.get(k, "<missing>")
        same = (v1 == v2)
        flag = "same" if same else "DIFF"
        print(f"  {k}:")
        print(f"    run1 = {v1}")
        print(f"    run2 = {v2}")
        print(f"    -> {flag}")


def diagnostics_summary(diag):
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
    if "grad_norms" in diag:
        gn = np.asarray(diag["grad_norms"], dtype=float)
        out["last50_gradnorm"] = float(np.mean(gn[-50:])) if len(gn) >= 50 else (float(np.mean(gn)) if len(gn) > 0 else np.nan)
    return out


if __name__ == "__main__":
    # ------------------------------------------------------------
    # EDIT THESE TWO PATHS
    # ------------------------------------------------------------
    run1_path = "rerun_from_current_state_backup/old_outputs/dqn_cilia_2_ball_results.npy"
    run2_path = "dqn_cilia_2_ball_results.npy"

    run1 = load_dqn_result(run1_path)
    run2 = load_dqn_result(run2_path)

    print("=" * 72)
    print("Comparing DQN runs")
    print("run1:", os.path.abspath(run1_path))
    print("run2:", os.path.abspath(run2_path))

    # ------------------------------------------------------------
    # settings
    # ------------------------------------------------------------
    print_dict_diff("Train settings", run1.get("train_settings", {}), run2.get("train_settings", {}))
    print_dict_diff("Env settings", run1.get("env_settings", {}), run2.get("env_settings", {}))

    # ------------------------------------------------------------
    # best cycle
    # ------------------------------------------------------------
    c1 = cycle_summary(run1)
    c2 = cycle_summary(run2)

    print("\n" + "=" * 72)
    print("BEST CYCLE SUMMARY")

    if c1 is None or c2 is None:
        print("At least one run has no cycles saved.")
    else:
        print("\nRun 1:")
        print("  length     =", c1["length"])
        print("  avg reward =", c1["avg_reward"])
        print("  cycle      =", c1["cycle"])

        print("\nRun 2:")
        print("  length     =", c2["length"])
        print("  avg reward =", c2["avg_reward"])
        print("  cycle      =", c2["cycle"])

        print("\nCycle match up to cyclic shift:",
              cycles_match_up_to_shift(c1["cycle"], c2["cycle"]))

        if c1["actions"] is not None and c2["actions"] is not None:
            print("Action sequences exactly equal:", c1["actions"] == c2["actions"])

    # ------------------------------------------------------------
    # policy comparison
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("POLICY COMPARISON")
    p1 = run1.get("policy", None)
    p2 = run2.get("policy", None)

    if p1 is None or p2 is None:
        print("At least one run has no saved policy.")
    else:
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        if p1.shape != p2.shape:
            print("Policy shapes differ:", p1.shape, p2.shape)
        else:
            diff_mask = (p1 != p2)
            n_diff = int(np.sum(diff_mask))
            print("Policy shape:", p1.shape)
            print("Number of differing states:", n_diff)

            if n_diff > 0:
                idx = np.argwhere(diff_mask)
                print("First 20 differing states:")
                for k, (i, j) in enumerate(idx[:20]):
                    print(f"  state ({i},{j}): run1={p1[i,j]}, run2={p2[i,j]}")

    # ------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DIAGNOSTIC SUMMARIES")

    d1 = diagnostics_summary(run1.get("diagnostics", {}))
    d2 = diagnostics_summary(run2.get("diagnostics", {}))

    print_dict_diff("Diagnostics summary", d1, d2)