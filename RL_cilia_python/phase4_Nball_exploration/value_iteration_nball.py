#!/usr/bin/env python3
"""
value_iteration_nball.py
========================

Exact value iteration (VI) for the discrete N-ball cilia MDP defined by
CiliaNBallEnv, using the env's precomputed reward table flux_table = R(s,a)
and deterministic transition table next_state_table = N(s,a).

The MDP is deterministic, so the Bellman update is

    V_{k+1}(s) = max_a [ R(s,a) + gamma * V(N(s,a)) ].

Output is saved in the SAME .npy layout as the PPO pipeline (policy + ranked
cycles + env_settings), so visualize_cilia_run.py runs on it unchanged --
point its RESULT_FILE at the file written here to compare the VI stroke to
your PPO stroke.

Edit the CONFIG block, then run:  python value_iteration_nball.py

NOTE: building the tables with precompute=True solves a Stokeslet system for
every (state, action) pair. For N=4 that is ~8.15M solves and can take a
while (tens of minutes to a couple of hours depending on the machine). The
tables are cached to TABLE_CACHE afterward, so re-running VI -- e.g. a gamma
sweep -- is then fast (seconds per gamma). On a Mac, launch the first run
under  caffeinate -i python value_iteration_nball.py  so it doesn't sleep
mid-precompute.
"""

# ====================== CONFIG: edit these ======================
NBALLS   = 4
GAMMAS   = [0.99]          # single gamma; make this a list to sweep, e.g. [0.95, 0.99, 0.995]
TOL      = 1e-10
MAX_ITERS = 100000

ENV_SETTINGS = {
    "Nballs": NBALLS,
    "max_steps": 2000,          # irrelevant to VI, kept for provenance
    "precompute": True,         # REQUIRED: VI reads the precomputed tables
    "boundary_mode": "clip_penalty",
    "invalid_penalty": -0.1,
    "reward_rescale": 100.0,
    "reset_mode": "uniform_independent",  # irrelevant to VI
    "verbose": True,
}
# dtheta is added below from a single knob so it stays consistent
import numpy as np
ENV_SETTINGS["dtheta"] = np.pi / 20

TABLE_CACHE = f"vi_tables_{NBALLS}ball.npz"   # cache of R + flat next-state index
OUTDIR      = f"vi_{NBALLS}ball_runs"
# ================================================================

import os
import csv
import json
import datetime


# ------------------------------------------------------------------
# Tables: build from env (slow, once) or load from cache (fast)
# ------------------------------------------------------------------
def build_tables_from_env(env_settings):
    """Instantiate the env (with precompute=True) and extract flat tables."""
    from cilia_n_ball_env import CiliaNBallEnv  # imported lazily so tests don't need gym
    env = CiliaNBallEnv(**env_settings)

    shape = tuple(int(b) for b in env.n_bins)
    n_states = int(env.n_states)
    nA = int(env.n_actions)

    R = env.flux_table.reshape(n_states, nA).astype(np.float64)
    NS = env.next_state_table.reshape(n_states, nA, env.Nangles)
    coords = tuple(NS[..., k] for k in range(env.Nangles))
    NSflat = np.ravel_multi_index(coords, dims=shape).astype(np.int64)

    n_bins = np.array(shape, dtype=np.int64)
    del env  # free the big tables
    return R, NSflat, n_bins


def load_or_build_tables(env_settings, cache_path):
    if os.path.exists(cache_path):
        print(f"[vi] loading cached tables from {cache_path}")
        z = np.load(cache_path)
        return z["R"], z["NSflat"], z["n_bins"]
    print("[vi] no cache found -- building tables from env (this is the slow step)")
    R, NSflat, n_bins = build_tables_from_env(env_settings)
    print(f"[vi] caching tables to {cache_path}")
    np.savez_compressed(cache_path, R=R, NSflat=NSflat, n_bins=n_bins)
    return R, NSflat, n_bins


# ------------------------------------------------------------------
# Core: value iteration (deterministic MDP, fully vectorized)
# ------------------------------------------------------------------
def run_value_iteration(R, NSflat, gamma, tol=1e-10, max_iters=100000):
    """R, NSflat: (n_states, nA). Returns V, greedy policy_flat, iters, delta."""
    n_states = R.shape[0]
    V = np.zeros(n_states, dtype=np.float64)
    delta = np.inf
    it = 0
    for it in range(1, max_iters + 1):
        Q = R + gamma * V[NSflat]      # (n_states, nA)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < tol:
            break
    Q = R + gamma * V[NSflat]          # final consistent greedy policy
    policy_flat = Q.argmax(axis=1).astype(np.int64)
    return V, policy_flat, it, delta


# ------------------------------------------------------------------
# Core: all cycles of a deterministic greedy policy (functional graph)
# ------------------------------------------------------------------
def find_all_cycles_flat(succ):
    """succ[s] = greedy next state (flat). Returns list of cycles (flat indices)."""
    n = len(succ)
    status = np.zeros(n, dtype=np.uint8)  # 0 unseen, 1 on current path, 2 finished
    cycles = []
    for start in range(n):
        if status[start]:
            continue
        path = []
        pos = {}
        node = start
        while status[node] == 0:
            status[node] = 1
            pos[node] = len(path)
            path.append(node)
            node = int(succ[node])
        if status[node] == 1:          # closed a loop within this walk -> new cycle
            cycles.append(path[pos[node]:])
        for nd in path:
            status[nd] = 2
    return cycles


def build_cycle_records(cycles, policy_flat, R, shape):
    """Turn flat cycles into ranked records matching the PPO .npy layout."""
    records = []
    for cyc in cycles:
        acts = [int(policy_flat[f]) for f in cyc]
        rews = [float(R[f, policy_flat[f]]) for f in cyc]
        states = [tuple(int(x) for x in np.unravel_index(f, shape)) for f in cyc]
        records.append({
            "cycle": states,
            "actions": acts,
            "rewards": rews,
            "avg_reward": float(np.mean(rews)),
            "length": len(cyc),
        })
    records.sort(key=lambda d: (-d["avg_reward"], -d["length"]))
    return records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(os.path.join(OUTDIR, "results"), exist_ok=True)
    R, NSflat, n_bins = load_or_build_tables(ENV_SETTINGS, TABLE_CACHE)
    shape = tuple(int(b) for b in n_bins)
    n_states, nA = R.shape
    print(f"[vi] N={NBALLS}  n_bins={list(shape)}  states={n_states}  actions={nA}")

    rows = []
    for gamma in GAMMAS:
        print(f"\n[vi] ===== value iteration, gamma={gamma} =====")
        V, policy_flat, iters, delta = run_value_iteration(
            R, NSflat, gamma, tol=TOL, max_iters=MAX_ITERS)
        print(f"[vi] converged in {iters} iters, ||V_k+1 - V_k||_inf = {delta:.3e}")

        succ = NSflat[np.arange(n_states), policy_flat]
        cycles = find_all_cycles_flat(succ)
        records = build_cycle_records(cycles, policy_flat, R, shape)

        if records:
            best = records[0]
            print(f"[vi] {len(records)} recurrent cycles; "
                  f"best: len={best['length']}, avg_reward={best['avg_reward']:.6f}")
        else:
            best = {"length": None, "avg_reward": None}
            print("[vi] no cycles found (unexpected).")

        out = {
            "policy": policy_flat.reshape(shape),
            "value_function": V.reshape(shape),
            "cycles": np.array(records, dtype=object),
            "env_settings": ENV_SETTINGS,
            "vi_settings": {
                "algo": "value_iteration",
                "gamma": gamma,
                "tol": TOL,
                "iterations": iters,
                "final_delta": delta,
            },
        }
        gtag = f"gamma{gamma}".replace(".", "p")
        result_path = os.path.join(OUTDIR, "results", f"vi_{NBALLS}ball_{gtag}.npy")
        np.save(result_path, out, allow_pickle=True)
        print(f"[vi] saved {result_path}")

        rows.append({
            "gamma": gamma,
            "iterations": iters,
            "final_delta": delta,
            "n_cycles": len(records),
            "best_cycle_length": best["length"],
            "best_avg_reward": best["avg_reward"],
            "result_path": result_path,
        })

    summary_csv = os.path.join(OUTDIR, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "Nballs": NBALLS, "n_bins": list(shape),
        "n_states": int(n_states), "n_actions": int(nA),
        "gammas": GAMMAS, "tol": TOL,
        "env_settings": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                         for k, v in ENV_SETTINGS.items()},
        "table_cache": os.path.abspath(TABLE_CACHE),
    }
    with open(os.path.join(OUTDIR, "vi_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[vi] summary -> {summary_csv}")
    print("[vi] visualize a result with: edit RESULT_FILE in visualize_cilia_run.py "
          "to point at the .npy above, then run it.")


if __name__ == "__main__":
    main()