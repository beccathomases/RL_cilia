#!/usr/bin/env python3
"""
ppo_sweep_n4_compare_howard.py
==============================

Run a PPO seed sweep for the N=4 cilia environment and compare detected
cycles against Howard average-reward benchmarks.

Default sweep:
    dtheta = pi/20 and pi/30
    seeds  = 0,...,9
    timesteps = 1,000,000 per run

Outputs:
    results/ppo_sweeps/n4_dtheta_pi20/seed_000/result.npy
    results/ppo_sweeps/n4_dtheta_pi20/seed_000/model.zip
    results/ppo_sweeps/n4_dtheta_pi20/seed_000/metadata.json
    results/ppo_sweeps/n4_dtheta_pi20/summary.csv

and similarly for dtheta_pi30.

Run:
    caffeinate -i python ppo_sweep_n4_compare_howard.py

Quicker test:
    caffeinate -i python ppo_sweep_n4_compare_howard.py --seeds 0 1 --timesteps 100000
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("This script needs gymnasium.") from e

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    raise ImportError(
        "This script needs stable-baselines3. Try: pip install stable-baselines3"
    ) from e

from cilia_n_ball_env import CiliaNBallEnv


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

NBALLS = 4

DEFAULT_SEEDS = list(range(10))
DEFAULT_TIMESTEPS = 1_000_000

# Use string labels so directory names are clean.
DTHETA_CASES = {
    "pi20": np.pi / 20,
    "pi30": np.pi / 30,
}

HOWARD_SUMMARY_PATHS = {
    "pi20": Path("results/howard/dtheta_pi20/howard_summary_4ball.npz"),
    "pi30": Path("results/howard/dtheta_pi30/howard_summary_4ball.npz"),
}

OUT_ROOT = Path("results/ppo_sweeps")


# ---------------------------------------------------------------------
# Observation wrapper
# ---------------------------------------------------------------------

class NormalizedStateWrapper(gym.ObservationWrapper):
    """
    Convert MultiDiscrete integer state to a Box observation in [-1,1]^N.

    This makes PPO's MLP policy behavior more regular than feeding raw bin
    indices directly.
    """

    def __init__(self, env: CiliaNBallEnv):
        super().__init__(env)
        self.n_bins = np.asarray(env.n_bins, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.Nangles,),
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        # Map index 0 -> -1 and index n_bins-1 -> 1.
        return 2.0 * obs / (self.n_bins - 1.0) - 1.0


def make_env(dtheta: float, seed: int, max_steps: int = 1000):
    def _init():
        env = CiliaNBallEnv(
            Nballs=NBALLS,
            max_steps=max_steps,
            precompute=False,
            boundary_mode="clip_penalty",
            invalid_penalty=-0.1,
            reward_rescale=100.0,
            dtheta=dtheta,
            reset_mode="uniform_independent",
            verbose=False,
        )
        env = NormalizedStateWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------------------
# Howard benchmark loading
# ---------------------------------------------------------------------

def load_howard_gain(summary_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """
    Load Howard mean reward/gain and cycle length from summary file.

    The gain can be read from eta if present, or from the mean of the detected
    cycle rewards.
    """
    if not summary_path.exists():
        print(f"[warn] Howard summary missing: {summary_path}")
        return None, None

    with np.load(summary_path, allow_pickle=True) as z:
        cycle_start = int(np.asarray(z["cycle_start"]).reshape(-1)[0])
        cycle_length = int(np.asarray(z["cycle_length"]).reshape(-1)[0])

        gain = None
        if "eta" in z:
            eta = np.asarray(z["eta"], dtype=float)
            if eta.size:
                gain = float(np.nanmax(eta))

        if gain is None and "rewards" in z:
            rewards = np.asarray(z["rewards"], dtype=float)
            if cycle_start >= 0 and cycle_length > 0:
                gain = float(np.nanmean(rewards[cycle_start:cycle_start + cycle_length]))
            else:
                gain = float(np.nanmean(rewards))

    return gain, cycle_length


# ---------------------------------------------------------------------
# Rollout / cycle detection
# ---------------------------------------------------------------------

def unwrapped_env_from_vec(vec_env):
    """
    Get the underlying CiliaNBallEnv from DummyVecEnv -> Monitor -> ObservationWrapper.
    """
    env = vec_env.envs[0]
    # Monitor
    if hasattr(env, "env"):
        env = env.env
    # NormalizedStateWrapper
    if hasattr(env, "env"):
        env = env.env
    return env


def state_to_angles_from_table_state(env: CiliaNBallEnv, state: np.ndarray):
    return env.state_to_angles(state)


def normalized_obs_from_state(env: CiliaNBallEnv, state: np.ndarray):
    state = np.asarray(state, dtype=np.float32)
    n_bins = np.asarray(env.n_bins, dtype=np.float32)
    return (2.0 * state / (n_bins - 1.0) - 1.0).astype(np.float32)


def detect_cycle_from_midpoint(
    model: PPO,
    dtheta: float,
    seed: int,
    max_rollout_steps: int = 5000,
) -> Dict:
    """
    Roll out deterministic PPO policy from midpoint and detect first repeated state.
    """
    env = CiliaNBallEnv(
        Nballs=NBALLS,
        max_steps=max_rollout_steps + 5,
        precompute=False,
        boundary_mode="clip_penalty",
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        dtheta=dtheta,
        reset_mode="midpoint",
        verbose=False,
    )
    obs, _ = env.reset(seed=seed)

    # Force midpoint explicitly.
    state = np.array([nb // 2 for nb in env.n_bins], dtype=np.int64)
    env.state = state.copy()
    obs_norm = normalized_obs_from_state(env, state)

    states = [state.copy()]
    angles = [env.state_to_angles(state)]
    actions = []
    rewards = []

    seen = {tuple(state.tolist()): 0}
    cycle_start = -1
    cycle_length = -1

    for k in range(max_rollout_steps):
        action, _ = model.predict(obs_norm, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.asarray(next_state, dtype=np.int64)

        actions.append(action)
        rewards.append(float(reward))
        states.append(next_state.copy())
        angles.append(env.state_to_angles(next_state))

        key = tuple(next_state.tolist())
        if key in seen:
            cycle_start = int(seen[key])
            cycle_length = int(len(states) - 1 - cycle_start)
            break

        seen[key] = len(states) - 1
        obs_norm = normalized_obs_from_state(env, next_state)

    states = np.asarray(states, dtype=np.int64)
    angles = np.asarray(angles, dtype=float)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=float)

    if cycle_start >= 0 and cycle_length > 0:
        cycle_rewards = rewards[cycle_start:cycle_start + cycle_length]
        avg_reward = float(np.mean(cycle_rewards))
        total_reward = float(np.sum(cycle_rewards))
    else:
        cycle_rewards = np.array([], dtype=float)
        avg_reward = float("nan")
        total_reward = float("nan")

    cycle_record = {
        "cycle_start": cycle_start,
        "length": cycle_length,
        "avg_reward": avg_reward,
        "total_reward": total_reward,
        "cycle": states[cycle_start:cycle_start + cycle_length].tolist()
        if cycle_start >= 0 else [],
        "angles": angles[cycle_start:cycle_start + cycle_length].tolist()
        if cycle_start >= 0 else [],
        "actions": actions[cycle_start:cycle_start + cycle_length].tolist()
        if cycle_start >= 0 else [],
        "rewards": cycle_rewards.tolist(),
    }

    return {
        "env_settings": {
            "Nballs": NBALLS,
            "max_steps": max_rollout_steps,
            "precompute": False,
            "boundary_mode": "clip_penalty",
            "invalid_penalty": -0.1,
            "reward_rescale": 100.0,
            "dtheta": float(dtheta),
            "reset_mode": "midpoint",
            "n_bins": env.n_bins.tolist(),
        },
        "states": states,
        "angles": angles,
        "actions": actions,
        "rewards": rewards,
        "cycle_start": cycle_start,
        "cycle_length": cycle_length,
        "avg_reward": avg_reward,
        "cycles": np.array([cycle_record], dtype=object),
    }


# ---------------------------------------------------------------------
# Training one run
# ---------------------------------------------------------------------

def train_one(
    label: str,
    dtheta: float,
    seed: int,
    timesteps: int,
    out_dir: Path,
    howard_gain: Optional[float],
    howard_cycle_length: Optional[int],
    force: bool = False,
) -> Dict:
    run_dir = out_dir / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "result.npy"
    summary_path = run_dir / "summary.json"
    model_path = run_dir / "model.zip"

    if result_path.exists() and summary_path.exists() and not force:
        print(f"[skip] {label} seed={seed}: existing result {result_path}")
        with open(summary_path, "r") as f:
            return json.load(f)

    print("=" * 72)
    print(f"[train] dtheta={label}, seed={seed}, timesteps={timesteps}")
    print(f"[train] output: {run_dir}")

    vec_env = DummyVecEnv([make_env(dtheta=dtheta, seed=seed, max_steps=1000)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=0,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        tensorboard_log=None,
    )

    t0 = datetime.datetime.now()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    t1 = datetime.datetime.now()
    elapsed_sec = (t1 - t0).total_seconds()

    model.save(model_path)

    rollout = detect_cycle_from_midpoint(
        model,
        dtheta=dtheta,
        seed=seed,
        max_rollout_steps=5000,
    )

    # Save result in a similar style to the old PPO visualizer expectation:
    # a single dict stored as a .npy object.
    result = {
        "env_settings": rollout["env_settings"],
        "train_settings": {
            "algorithm": "PPO",
            "seed": seed,
            "timesteps": timesteps,
            "dtheta_label": label,
            "elapsed_sec": elapsed_sec,
        },
        "policy": None,
        "cycles": rollout["cycles"],
        "states": rollout["states"],
        "angles": rollout["angles"],
        "actions": rollout["actions"],
        "rewards": rollout["rewards"],
        "cycle_start": rollout["cycle_start"],
        "cycle_length": rollout["cycle_length"],
        "avg_reward": rollout["avg_reward"],
    }
    np.save(result_path, result, allow_pickle=True)

    ppo_avg = rollout["avg_reward"]
    ppo_len = rollout["cycle_length"]

    if howard_gain is not None and np.isfinite(ppo_avg):
        frac_howard = float(ppo_avg / howard_gain)
        pct_howard = 100.0 * frac_howard
    else:
        frac_howard = float("nan")
        pct_howard = float("nan")

    summary = {
        "dtheta_label": label,
        "dtheta": float(dtheta),
        "seed": int(seed),
        "timesteps": int(timesteps),
        "elapsed_sec": elapsed_sec,
        "cycle_start": int(rollout["cycle_start"]),
        "cycle_length": int(ppo_len),
        "ppo_avg_reward": float(ppo_avg),
        "howard_gain": None if howard_gain is None else float(howard_gain),
        "howard_cycle_length": None if howard_cycle_length is None else int(howard_cycle_length),
        "ppo_fraction_of_howard": frac_howard,
        "ppo_percent_of_howard": pct_howard,
        "matches_howard_length": (
            bool(ppo_len == howard_cycle_length)
            if howard_cycle_length is not None and ppo_len > 0
            else False
        ),
        "result_path": str(result_path),
        "model_path": str(model_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] {label} seed={seed}: "
        f"cycle_length={ppo_len}, "
        f"avg_reward={ppo_avg:.8g}, "
        f"Howard={howard_gain}, "
        f"%Howard={pct_howard:.2f}"
    )

    vec_env.close()
    return summary


# ---------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------

def write_summary_csv(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dtheta_label",
        "dtheta",
        "seed",
        "timesteps",
        "elapsed_sec",
        "cycle_start",
        "cycle_length",
        "ppo_avg_reward",
        "howard_gain",
        "howard_cycle_length",
        "ppo_fraction_of_howard",
        "ppo_percent_of_howard",
        "matches_howard_length",
        "result_path",
        "model_path",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[summary] wrote {out_csv}")


def print_group_summary(rows: List[Dict]):
    if not rows:
        return

    labels = sorted(set(r["dtheta_label"] for r in rows))
    print()
    print("=" * 72)
    print("PPO sweep summary")
    for label in labels:
        rr = [r for r in rows if r["dtheta_label"] == label]
        vals = np.array([r["ppo_percent_of_howard"] for r in rr], dtype=float)
        lens = [r["cycle_length"] for r in rr]
        vals = vals[np.isfinite(vals)]

        print(f"\n{label}:")
        print(f"  runs: {len(rr)}")
        if vals.size:
            print(f"  % Howard: mean={vals.mean():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")
        print(f"  cycle lengths: {sorted(set(lens))}")
        nmatch = sum(bool(r["matches_howard_length"]) for r in rr)
        print(f"  Howard-length matches: {nmatch}/{len(rr)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="seeds to run",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="PPO training timesteps per seed",
    )
    parser.add_argument(
        "--cases",
        choices=list(DTHETA_CASES.keys()),
        nargs="+",
        default=list(DTHETA_CASES.keys()),
        help="which dtheta cases to run",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help="root output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even if result.npy and summary.json already exist",
    )
    args = parser.parse_args()

    all_rows = []

    for label in args.cases:
        dtheta = DTHETA_CASES[label]

        howard_gain, howard_len = load_howard_gain(HOWARD_SUMMARY_PATHS[label])
        print("=" * 72)
        print(f"[case] {label}: dtheta={dtheta}")
        print(f"[case] Howard summary: gain={howard_gain}, cycle_length={howard_len}")

        out_dir = args.out_root / f"n4_dtheta_{label}"

        rows = []
        for seed in args.seeds:
            row = train_one(
                label=label,
                dtheta=dtheta,
                seed=seed,
                timesteps=args.timesteps,
                out_dir=out_dir,
                howard_gain=howard_gain,
                howard_cycle_length=howard_len,
                force=args.force,
            )
            rows.append(row)
            all_rows.append(row)

            # Update CSV after every run so partial sweeps are saved.
            write_summary_csv(rows, out_dir / "summary.csv")
            write_summary_csv(all_rows, args.out_root / "summary_all.csv")

        print_group_summary(rows)

    print_group_summary(all_rows)
    print("[done] PPO sweep complete.")


if __name__ == "__main__":
    main()