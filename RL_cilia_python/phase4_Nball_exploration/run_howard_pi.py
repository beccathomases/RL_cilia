#!/usr/bin/env python3
"""
Howard policy iteration for the discrete N-ball cilia tables.

Expected input, one file per N:
    vi_tables_{N}ball.npz

The loader is intentionally forgiving about key names.  It looks for a
reward/flux table with shape n_bins + (n_actions,) and a next-state table
with shape n_bins + (n_actions, N).  If an initial VI policy is present, it
uses it; otherwise it starts from the greedy one-step policy.

Outputs, one set per N:
    howard_policy_{N}ball.npy       integer policy table, shape n_bins
    howard_stroke_{N}ball.npy       angle rollout, shape (T+1, N)
    howard_states_{N}ball.npy       state-index rollout, shape (T+1, N)
    howard_actions_{N}ball.npy      action-index rollout, shape (T,)
    howard_rewards_{N}ball.npy      reward rollout, shape (T,)
    howard_summary_{N}ball.npz      bundle with metadata and all arrays

Example:
    python run_howard_pi.py --nballs 2 3 4
    python run_howard_pi.py --nballs 4 --table-dir . --out-dir howard_out
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from cilia_n_ball_env import CiliaNBallEnv

# -----------------------------------------------------------------------------
# Table loading
# -----------------------------------------------------------------------------

REWARD_KEY_HINTS = ("reward", "rewards", "flux", "flux_table", "R")
NEXT_KEY_HINTS = ("next", "next_state", "next_states", "next_state_table", "P")
POLICY_KEY_HINTS = ("policy", "pi", "vi_policy", "optimal_policy")


@dataclass
class Tables:
    rewards: np.ndarray              # flat shape: (n_states, n_actions)
    next_states: Optional[np.ndarray] # coordinate table, if present
    next_flat: Optional[np.ndarray]   # flat deterministic transition table, if present
    initial_policy: Optional[np.ndarray]
    n_bins: Tuple[int, ...]
    n_states: int
    n_actions: int


def _find_key_by_hint(npz: np.lib.npyio.NpzFile, hints: Iterable[str]) -> Optional[str]:
    keys = list(npz.keys())
    lower = {k.lower(): k for k in keys}
    for h in hints:
        if h.lower() in lower:
            return lower[h.lower()]
    for k in keys:
        lk = k.lower()
        if any(h.lower() in lk for h in hints):
            return k
    return None


def load_tables(path: Path, nballs: int) -> Tables:
    """
    Load reward and transition tables from vi_tables_{N}ball.npz.

    Supports the compact cache written by build_table_cache.py:

        R       shape (n_states, n_actions)
        NSflat  shape (n_states, n_actions)
        n_bins  shape (N,)

    Also keeps support for older/full coordinate next-state tables.
    """
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.keys())

        # ------------------------------------------------------------
        # Preferred/current compact cache format:
        #     R, NSflat, n_bins
        # ------------------------------------------------------------
        if "R" in z and "NSflat" in z and "n_bins" in z:
            n_bins = tuple(int(x) for x in np.asarray(z["n_bins"]).ravel())
            if len(n_bins) != nballs:
                raise ValueError(
                    f"For N={nballs}, expected len(n_bins)=={nballs}, "
                    f"but got n_bins={n_bins} from {path}."
                )

            n_states = int(np.prod(n_bins))
            rewards = np.asarray(z["R"], dtype=np.float64)
            next_flat = np.asarray(z["NSflat"], dtype=np.int64)

            if rewards.ndim != 2:
                rewards = rewards.reshape(n_states, -1)
            if next_flat.ndim != 2:
                next_flat = next_flat.reshape(n_states, -1)

            if rewards.shape[0] != n_states:
                raise ValueError(
                    f"R has shape {rewards.shape}, but n_bins={n_bins} "
                    f"implies n_states={n_states}."
                )
            if next_flat.shape != rewards.shape:
                raise ValueError(
                    f"NSflat shape {next_flat.shape} does not match "
                    f"R shape {rewards.shape}."
                )

            if np.any(next_flat < 0) or np.any(next_flat >= n_states):
                raise ValueError(
                    f"NSflat contains indices outside [0,{n_states - 1}]."
                )

            n_actions = int(rewards.shape[1])

            policy_key = _find_key_by_hint(z, POLICY_KEY_HINTS)
            initial_policy = None
            if policy_key is not None:
                initial_policy = np.asarray(z[policy_key], dtype=np.int64).reshape(n_bins)

            print(
                f"Loaded {path.name} compact cache: "
                f"n_bins={n_bins}, states={n_states}, actions={n_actions}"
            )

            return Tables(
                rewards=rewards,
                next_states=None,
                next_flat=next_flat,
                initial_policy=initial_policy,
                n_bins=n_bins,
                n_states=n_states,
                n_actions=n_actions,
            )

        # ------------------------------------------------------------
        # Fallback: older/full table format with coordinate next states.
        # ------------------------------------------------------------
        reward_key = _find_key_by_hint(z, REWARD_KEY_HINTS)
        next_key = _find_key_by_hint(z, NEXT_KEY_HINTS)
        policy_key = _find_key_by_hint(z, POLICY_KEY_HINTS)

        if reward_key is None or next_key is None:
            raise KeyError(
                f"Could not identify reward/next-state tables in {path}.\n"
                f"Available keys: {keys}\n"
                f"Expected compact keys R, NSflat, n_bins, or full keys like "
                f"reward/flux_table and next_state_table."
            )

        rewards_raw = np.asarray(z[reward_key], dtype=np.float64)
        next_states = np.asarray(z[next_key], dtype=np.int64)
        initial_policy = None if policy_key is None else np.asarray(z[policy_key], dtype=np.int64)

    if next_states.ndim < 3:
        raise ValueError(f"next-state table has too few dimensions: {next_states.shape}")
    if next_states.shape[-1] != nballs:
        raise ValueError(
            f"For N={nballs}, expected next_states.shape[-1] == {nballs}, "
            f"got {next_states.shape}."
        )

    n_bins = tuple(int(x) for x in next_states.shape[:-2])
    n_actions = int(next_states.shape[-2])
    n_states = int(np.prod(n_bins))

    expected_reward_shape = n_bins + (n_actions,)
    if rewards_raw.shape == expected_reward_shape:
        rewards = rewards_raw.reshape(n_states, n_actions)
    elif rewards_raw.shape == (n_states, n_actions):
        rewards = rewards_raw
    else:
        raise ValueError(
            f"Reward shape {rewards_raw.shape} is not compatible with next-state "
            f"shape {next_states.shape}; expected {expected_reward_shape} "
            f"or {(n_states, n_actions)}."
        )

    if initial_policy is not None:
        initial_policy = np.asarray(initial_policy).reshape(n_bins)

    print(
        f"Loaded {path.name} full cache: "
        f"n_bins={n_bins}, states={n_states}, actions={n_actions}"
    )

    return Tables(
        rewards=rewards,
        next_states=next_states,
        next_flat=None,
        initial_policy=initial_policy,
        n_bins=n_bins,
        n_states=n_states,
        n_actions=n_actions,
    )


def flatten_tables(tables: Tables) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return rewards[S,A] and next_flat[S,A].

    If the cache already supplied NSflat, use it directly.
    Otherwise convert coordinate next-state table to flat indices.
    """
    R = tables.rewards.reshape(tables.n_states, tables.n_actions)

    if tables.next_flat is not None:
        next_flat = tables.next_flat.reshape(tables.n_states, tables.n_actions)
        return R, next_flat.astype(np.int64, copy=False)

    if tables.next_states is None:
        raise ValueError("No next-state information found in Tables.")

    NS = tables.next_states.reshape(
        tables.n_states,
        tables.n_actions,
        len(tables.n_bins),
    )

    coords = [NS[:, :, k].reshape(-1) for k in range(len(tables.n_bins))]
    next_flat = np.ravel_multi_index(coords, dims=tables.n_bins).reshape(
        tables.n_states,
        tables.n_actions,
    )

    return R, next_flat.astype(np.int64, copy=False)


# -----------------------------------------------------------------------------
# Deterministic average-reward Howard policy iteration
# -----------------------------------------------------------------------------

def evaluate_deterministic_policy(policy: np.ndarray, R: np.ndarray, next_flat: np.ndarray):
    """
    Evaluate a deterministic policy on a deterministic MDP.

    Returns eta and bias, both length S.  eta[s] is the cycle-mean reward of
    the recurrent class reached by following the policy from s.  bias solves
        bias[s] + eta[s] = r_pi[s] + bias[next_pi[s]]
    on trees and on each cycle with one arbitrary zero reference per cycle.
    """
    S = policy.size
    succ = next_flat[np.arange(S), policy]
    rpi = R[np.arange(S), policy]

    eta = np.zeros(S, dtype=float)
    bias = np.zeros(S, dtype=float)
    color = np.zeros(S, dtype=np.int8)  # 0 new, 1 active path, 2 finished

    for start in range(S):
        if color[start] != 0:
            continue

        path = []
        pos: Dict[int, int] = {}
        s = int(start)

        while color[s] == 0:
            pos[s] = len(path)
            path.append(s)
            color[s] = 1
            s = int(succ[s])

        if color[s] == 1 and s in pos:
            # Found a new cycle inside this path.
            c0 = pos[s]
            cycle = path[c0:]
            g = float(np.mean(rpi[cycle]))
            for c in cycle:
                eta[c] = g

            # Fix one bias value to remove the additive constant.
            bias[cycle[0]] = 0.0
            # Walk backward around the cycle, excluding the reference node.
            for c in reversed(cycle[1:]):
                bias[c] = rpi[c] - g + bias[succ[c]]

            tail_end = c0 - 1
        else:
            # Hit an already evaluated component.
            tail_end = len(path) - 1

        # Evaluate the transient tree back toward the start.
        for k in range(tail_end, -1, -1):
            u = path[k]
            v = int(succ[u])
            eta[u] = eta[v]
            bias[u] = rpi[u] - eta[u] + bias[v]

        for u in path:
            color[u] = 2

    return eta, bias


def improve_policy(
    policy: np.ndarray,
    R: np.ndarray,
    next_flat: np.ndarray,
    eta: np.ndarray,
    bias: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, int]:
    """
    Sticky lexicographic Howard improvement.

    We choose actions by:
        1. larger successor gain eta[next]
        2. among tied gains, larger one-step bias score R + bias[next]

    But if the current action is already within tolerance of the best
    lexicographic choice, we KEEP it.  This prevents endless switching among
    equivalent/tied optimal actions.
    """
    S, A = R.shape
    idx = np.arange(S)

    cand_eta = eta[next_flat]
    cand_score = R + bias[next_flat]

    # Best successor gain available from each state.
    best_eta = np.max(cand_eta, axis=1)

    # Among actions with near-best gain, choose best bias score.
    eta_ok = cand_eta >= best_eta[:, None] - tol
    masked_score = np.where(eta_ok, cand_score, -np.inf)
    best_score = np.max(masked_score, axis=1)
    best_action = np.argmax(masked_score, axis=1).astype(np.int64)

    # Current action's lexicographic score.
    curr_eta = cand_eta[idx, policy]
    curr_score = cand_score[idx, policy]

    # Sticky tie rule: if current action is already essentially best, keep it.
    keep_current = (
        (curr_eta >= best_eta - tol) &
        (curr_score >= best_score - tol)
    )

    new_policy = best_action.copy()
    new_policy[keep_current] = policy[keep_current]

    changes = int(np.count_nonzero(new_policy != policy))
    return new_policy, changes

#def improve_policy(
#    policy: np.ndarray,
#    R: np.ndarray,
#    next_flat: np.ndarray,
#    eta: np.ndarray,
#    bias: np.ndarray,
#    tol: float,
#) -> Tuple[np.ndarray, int]:
#    """Lexicographic Howard improvement: higher gain first, then higher bias."""
#    cand_eta = eta[next_flat]
#    cand_bias_score = R + bias[next_flat]
#
#   best_eta = np.max(cand_eta, axis=1)
#    eta_ok = cand_eta >= best_eta[:, None] - tol
#    masked_score = np.where(eta_ok, cand_bias_score, -np.inf)
#    new_policy = np.argmax(masked_score, axis=1).astype(np.int64)
#
#    changes = int(np.count_nonzero(new_policy != policy))
#    return new_policy, changes


def howard_policy_iteration(
    R: np.ndarray,
    next_flat: np.ndarray,
    initial_policy: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-12,
):
    S, A = R.shape
    if initial_policy is None:
        policy = np.argmax(R, axis=1).astype(np.int64)
    else:
        policy = np.asarray(initial_policy, dtype=np.int64).reshape(S).copy()
        if np.any(policy < 0) or np.any(policy >= A):
            raise ValueError("Initial policy contains invalid action indices.")

    history = []
    for it in range(1, max_iter + 1):
        eta, bias = evaluate_deterministic_policy(policy, R, next_flat)
        new_policy, changes = improve_policy(policy, R, next_flat, eta, bias, tol=tol)

        history.append(
            {
                "iter": it,
                "changes": changes,
                "eta_min": float(np.min(eta)),
                "eta_max": float(np.max(eta)),
                "eta_mid": float(eta[S // 2]),
            }
        )
        print(
            f"  Howard iter {it:3d}: changes={changes:7d}, "
            f"gain range=[{np.min(eta): .8g}, {np.max(eta): .8g}]"
        )

        policy = new_policy
        if changes == 0:
            eta, bias = evaluate_deterministic_policy(policy, R, next_flat)
            return policy, eta, bias, history

    print(f"  WARNING: reached max_iter={max_iter} before policy stabilized.")
    eta, bias = evaluate_deterministic_policy(policy, R, next_flat)
    return policy, eta, bias, history


# -----------------------------------------------------------------------------
# Rollout and saving
# -----------------------------------------------------------------------------

def default_angle_ranges_for_tables(nballs: int):
    angle_mins = np.array([-np.pi / 4] + [-np.pi / 2] * (nballs - 1), dtype=float)
    angle_maxs = np.array([ np.pi / 4] + [ np.pi / 2] * (nballs - 1), dtype=float)
    return angle_mins, angle_maxs


def table_state_to_angles(state: np.ndarray, n_bins: Tuple[int, ...], nballs: int):
    """
    Convert table state indices to angles using the table grid itself.

    This is important for dtheta sweeps.  Do NOT instantiate CiliaNBallEnv
    with its default dtheta=pi/20 and use env.state_to_angles, because that
    gives wrong angles for pi/30, pi/40, etc.
    """
    state = np.asarray(state, dtype=float)
    n_bins_arr = np.asarray(n_bins, dtype=float)

    angle_mins, angle_maxs = default_angle_ranges_for_tables(nballs)
    dangle = (angle_maxs - angle_mins) / (n_bins_arr - 1)

    return angle_mins + state * dangle


def rollout_policy(
    policy_table: np.ndarray,
    tables: Tables,
    R: np.ndarray,
    next_flat: np.ndarray,
    nballs: int,
    max_steps: int,
    start: str = "midpoint",
):
    if start != "midpoint":
        raise ValueError("Only start='midpoint' is currently implemented.")

    state = np.array([nb // 2 for nb in tables.n_bins], dtype=np.int64)
    sflat = int(np.ravel_multi_index(tuple(state), dims=tables.n_bins))

    states = [state.copy()]
    angles = [table_state_to_angles(state, tables.n_bins, nballs)]
    actions = []
    rewards = []

    seen = {sflat: 0}
    cycle_start = None
    cycle_length = None

    policy_flat = policy_table.reshape(-1)

    for k in range(max_steps):
        action = int(policy_flat[sflat])
        r = float(R[sflat, action])
        nsflat = int(next_flat[sflat, action])
        nstate = np.array(np.unravel_index(nsflat, tables.n_bins), dtype=np.int64)

        actions.append(action)
        rewards.append(r)
        states.append(nstate.copy())
        angles.append(table_state_to_angles(nstate, tables.n_bins, nballs))

        sflat = nsflat
        state = nstate

        if sflat in seen:
            cycle_start = seen[sflat]
            cycle_length = len(states) - 1 - cycle_start
            break

        seen[sflat] = len(states) - 1

    return {
        "states": np.asarray(states, dtype=np.int64),
        "angles": np.asarray(angles, dtype=float),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=float),
        "cycle_start": -1 if cycle_start is None else int(cycle_start),
        "cycle_length": -1 if cycle_length is None else int(cycle_length),
    }


def save_outputs(
    out_dir: Path,
    nballs: int,
    policy: np.ndarray,
    eta: np.ndarray,
    bias: np.ndarray,
    rollout: Dict[str, np.ndarray],
    history,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"howard_policy_{nballs}ball.npy", policy)
    np.save(out_dir / f"howard_stroke_{nballs}ball.npy", rollout["angles"])
    np.save(out_dir / f"howard_states_{nballs}ball.npy", rollout["states"])
    np.save(out_dir / f"howard_actions_{nballs}ball.npy", rollout["actions"])
    np.save(out_dir / f"howard_rewards_{nballs}ball.npy", rollout["rewards"])

    # Compatibility aliases that are often convenient next to VI/PPO outputs.
    np.save(out_dir / f"gain_optimal_stroke_{nballs}ball.npy", rollout["angles"])
    np.save(out_dir / f"howard_gain_optimal_stroke_{nballs}ball.npy", rollout["angles"])

    np.savez(
        out_dir / f"howard_summary_{nballs}ball.npz",
        policy=policy,
        eta=eta.reshape(policy.shape),
        bias=bias.reshape(policy.shape),
        states=rollout["states"],
        angles=rollout["angles"],
        actions=rollout["actions"],
        rewards=rollout["rewards"],
        cycle_start=np.array(rollout["cycle_start"], dtype=int),
        cycle_length=np.array(rollout["cycle_length"], dtype=int),
        history=np.array(history, dtype=object),
    )

    print(
        f"  saved howard_stroke_{nballs}ball.npy "
        f"with angles shape {rollout['angles'].shape}; "
        f"cycle_start={rollout['cycle_start']}, cycle_length={rollout['cycle_length']}"
    )


def solve_one(nballs: int, table_dir: Path, out_dir: Path, max_iter: int, max_steps: int, tol: float):
    table_path = table_dir / f"vi_tables_{nballs}ball.npz"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing table file: {table_path}")

    print("=" * 72)
    print(f"N={nballs}: Howard average-reward policy iteration")
    tables = load_tables(table_path, nballs)
    R, next_flat = flatten_tables(tables)

    initial = None if tables.initial_policy is None else tables.initial_policy.reshape(-1)
    if initial is None:
        print("  no VI policy key found; starting from greedy immediate-reward policy")
    else:
        print("  using VI policy from npz as Howard initial policy")

    policy_flat, eta, bias, history = howard_policy_iteration(
        R, next_flat, initial_policy=initial, max_iter=max_iter, tol=tol
    )
    policy = policy_flat.reshape(tables.n_bins)

    rollout = rollout_policy(policy, tables, R, next_flat, nballs=nballs, max_steps=max_steps)
    save_outputs(out_dir, nballs, policy, eta, bias, rollout, history)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nballs", type=int, nargs="+", default=[2, 3, 4], help="N values to run")
    p.add_argument("--NBALLS", type=int, nargs="+", dest="nballs_caps", help="alias for --nballs")
    p.add_argument("--table-dir", type=Path, default=Path("."), help="directory containing vi_tables_{N}ball.npz")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="directory for .npy/.npz outputs")
    p.add_argument("--max-iter", type=int, default=200, help="maximum Howard improvement sweeps")
    p.add_argument("--max-steps", type=int, default=5000, help="maximum rollout length before giving up on cycle detection")
    p.add_argument("--tol", type=float, default=1e-12, help="gain tie tolerance")
    args = p.parse_args()

    nballs_list = args.nballs_caps if args.nballs_caps is not None else args.nballs
    for n in nballs_list:
        solve_one(int(n), args.table_dir, args.out_dir, args.max_iter, args.max_steps, args.tol)


if __name__ == "__main__":
    main()
