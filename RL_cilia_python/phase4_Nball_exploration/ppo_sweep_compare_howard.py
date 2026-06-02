#!/usr/bin/env python3
"""
ppo_sweep_compare_howard.py
===========================

General PPO seed sweep for the CiliaNBallEnv, compared against Howard
average-reward benchmarks.

Recommended first run:
    caffeinate -i python ppo_sweep_compare_howard.py \
      --nballs 2 3 \
      --cases pi20 \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --timesteps 1000000

Optional include N=4 too:
    caffeinate -i python ppo_sweep_compare_howard.py \
      --nballs 2 3 4 \
      --cases pi20 \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --timesteps 1000000

Outputs:
    results/ppo_sweeps_general/N2_dtheta_pi20/seed_000/result.npy
    results/ppo_sweeps_general/N2_dtheta_pi20/summary.csv
    results/ppo_sweeps_general/summary_all.csv

Notes:
    - This uses precompute=False for PPO.
    - Howard summaries are expected at:
        results/howard/dtheta_pi20/howard_summary_2ball.npz
        results/howard/dtheta_pi20/howard_summary_3ball.npz
        results/howard/dtheta_pi20/howard_summary_4ball.npz
      and similarly for pi30 if used.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
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

DEFAULT_NBALLS = [2, 3]
DEFAULT_SEEDS = list(range(10))
DEFAULT_TIMESTEPS = 1_000_000

DTHETA_CASES = {
    "pi20": np.pi / 20,
    "pi30": np.pi / 30,
}

OUT_ROOT = Path("results/ppo_sweeps_general")


# ---------------------------------------------------------------------
# Observation wrapper
# ---------------------------------------------------------------------

class NormalizedStateWrapper(gym.ObservationWrapper):
    """
    Convert MultiDiscrete integer state to Box observation in [-1,1]^N.
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
        return 2.0 * obs / (self.n_bins - 1.0) - 1.0


def normalized_obs_from_state(env: CiliaNBallEnv, state: np.ndarray):
    state = np.asarray(state, dtype=np.float32)
    n_bins = np.asarray(env.n_bins, dtype=np.float32)
    return (2.0 * state / (n_bins - 1.0) - 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# Env creation
# ---------------------------------------------------------------------

def make_env(nballs: int, dtheta: float, seed: int, max_steps: int = 1000):
    def _init():
        env = CiliaNBallEnv(
            Nballs=nballs,
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

def howard_summary_path(nballs: int, label: str) -> Path:
    return Path(f"results/howard/dtheta_{label}/howard_summary_{nballs}ball.npz")


def load_howard_gain(summary_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """
    Load Howard gain and cycle length from summary file.

    The gain is inferred from eta if present, otherwise from mean cycle reward.
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

def detect_cycle_from_midpoint(
    model: PPO,
    nballs: int,
    dtheta: float,
    seed: int,
    max_rollout_steps: int = 5000,
) -> Dict:
    """
    Roll out deterministic PPO policy from midpoint and detect first repeated state.
    """
    env = CiliaNBallEnv(
        Nballs=nballs,
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
            "Nballs": nballs,
            "max_steps": max_rollout_steps,
            "precompute": False,
            "boundary_mode": "clip_penalty",
            "invalid_penalty": -0.1,
            "reward_rescale": 100.0,
            "dtheta": float(dtheta),
            "reset_mode": "midpoint",
            "n_bins": env.n_bins.tolist(),
            "n_actions": int(env.n_actions),
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
    nballs: int,
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
        print(f"[skip] N={nballs} {label} seed={seed}: existing result {result_path}")
        with open(summary_path, "r") as f:
            return json.load(f)

    print("=" * 72)
    print(f"[train] N={nballs}, dtheta={label}, seed={seed}, timesteps={timesteps}")
    print(f"[train] output: {run_dir}")

    vec_env = DummyVecEnv([make_env(nballs=nballs, dtheta=dtheta, seed=seed, max_steps=1000)])

    # Use the same PPO hyperparameters as the N=4 sweep for a clean comparison.
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
        nballs=nballs,
        dtheta=dtheta,
        seed=seed,
        max_rollout_steps=5000,
    )

    result = {
        "env_settings": rollout["env_settings"],
        "train_settings": {
            "algorithm": "PPO",
            "Nballs": nballs,
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
        "nballs": int(nballs),
        "dtheta_label": label,
        "dtheta": float(dtheta),
        "seed": int(seed),
        "timesteps": int(timesteps),
        "elapsed_sec": float(elapsed_sec),
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
        f"[done] N={nballs} {label} seed={seed}: "
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
        "nballs",
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

    groups = sorted(set((r["nballs"], r["dtheta_label"]) for r in rows))

    print()
    print("=" * 72)
    print("PPO sweep summary")

    for nballs, label in groups:
        rr = [r for r in rows if r["nballs"] == nballs and r["dtheta_label"] == label]
        vals = np.array([r["ppo_percent_of_howard"] for r in rr], dtype=float)
        lens = [r["cycle_length"] for r in rr]
        vals = vals[np.isfinite(vals)]

        print(f"\nN={nballs}, {label}:")
        print(f"  runs: {len(rr)}")
        if vals.size:
            print(
                f"  % Howard: mean={vals.mean():.2f}, "
                f"min={vals.min():.2f}, max={vals.max():.2f}"
            )
        print(f"  cycle lengths: {sorted(set(lens))}")
        nmatch = sum(bool(r["matches_howard_length"]) for r in rr)
        print(f"  Howard-length matches: {nmatch}/{len(rr)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nballs",
        type=int,
        nargs="+",
        default=DEFAULT_NBALLS,
        help="N values to run",
    )
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
        default=["pi20"],
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

    for nballs in args.nballs:
        for label in args.cases:
            dtheta = DTHETA_CASES[label]
            hpath = howard_summary_path(nballs, label)
            howard_gain, howard_len = load_howard_gain(hpath)

            print("=" * 72)
            print(f"[case] N={nballs}, {label}: dtheta={dtheta}")
            print(f"[case] Howard summary path: {hpath}")
            print(f"[case] Howard gain={howard_gain}, cycle_length={howard_len}")

            out_dir = args.out_root / f"N{nballs}_dtheta_{label}"

            rows = []
            for seed in args.seeds:
                row = train_one(
                    nballs=nballs,
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

                # Write after every run so partial progress is saved.
                write_summary_csv(rows, out_dir / "summary.csv")
                write_summary_csv(all_rows, args.out_root / "summary_all.csv")

            print_group_summary(rows)

    print_group_summary(all_rows)
    print("[done] PPO sweep complete.")


if __name__ == "__main__":
    main()