#!/usr/bin/env python3
"""
ppo_sweep_phase6.py
===================

Phase 6: factored-action PPO comparison.

This script runs controlled PPO sweeps for the generalized N-ball cilia
environment using either

    action_mode="monolithic"
        one categorical action over all nonzero vectors in {-1,0,1}^N,
        with the global no-op excluded;

or

    action_mode="factored"
        one ternary action head per joint, {0,1,2} -> {-1,0,+1},
        with the global no-op included and tracked through noop_fraction.

The intended first Phase 6 tests are N=3 and N=4, where exact Howard policy
iteration benchmarks already exist from Phase 4. This lets us compare:

    Howard optimal gain
    Phase 4 monolithic PPO
    Phase 6 factored PPO

using the same environment physics, state representation, reward convention,
PPO hyperparameters, reset logic, rollout/cycle detector, and diagnostics.

The only intended method change is the action parameterization. In particular,
the PPO configuration is held fixed:

    gamma=0.99
    n_steps=2048
    batch_size=256
    n_epochs=10
    learning_rate=3e-4
    ent_coef=0.0
    clip_range=0.2

Since ent_coef=0.0, the summed entropy of the factored MultiCategorical policy
does not introduce an additional comparison confound.

Outputs:
    results/ppo_sweeps_phase6/N3_dtheta_pi20_factored/seed_000/model.zip
    results/ppo_sweeps_phase6/N3_dtheta_pi20_factored/seed_000/result.npz
    results/ppo_sweeps_phase6/N3_dtheta_pi20_factored/seed_000/summary.json
    results/ppo_sweeps_phase6/N3_dtheta_pi20_factored/summary.csv
    results/ppo_sweeps_phase6/summary_all.csv

Example smoke test:
    caffeinate -i python ppo_sweep_phase6.py \
      --nballs 3 \
      --action-mode factored \
      --cases pi20 \
      --seeds 0 1 \
      --timesteps 300000 \
      --outroot results/ppo_sweeps_phase6 \
      --force

Example full N=3/4 factored comparison:
    caffeinate -i python ppo_sweep_phase6.py \
      --nballs 3 \
      --action-mode factored \
      --cases pi20 pi30 \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --timesteps 1000000 \
      --outroot results/ppo_sweeps_phase6 \
      --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from cilia_n_ball_env import CiliaNBallEnv


# ---------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------

DTHETA_CASES = {
    "pi20": np.pi / 20,
    "pi30": np.pi / 30,
    "pi40": np.pi / 40,
    "pi50": np.pi / 50,
}


# ---------------------------------------------------------------------
# Radius / geometry scaling
# ---------------------------------------------------------------------

def resolve_rad(nballs: int, rad: float | None = None, rad_scale: float | None = None):
    """
    Decide the bead radius used in the environment.

    rad:
        Explicit radius, e.g. --rad 0.04.

    rad_scale:
        Constant relative thickness, e.g. --rad-scale 0.4 gives rad=0.4/N.

    If neither is supplied, return None and let CiliaNBallEnv use its default.
    """
    if rad is not None and rad_scale is not None:
        raise ValueError("Use either --rad or --rad-scale, not both.")

    if rad is not None:
        return float(rad)

    if rad_scale is not None:
        return float(rad_scale) / int(nballs)

    return None


def radius_label(rad_effective: float | None):
    if rad_effective is None:
        return "default"
    return f"{rad_effective:.8g}"


# ---------------------------------------------------------------------
# Gym/Gymnasium compatibility
# ---------------------------------------------------------------------

def reset_env(env, seed=None):
    try:
        if seed is None:
            out = env.reset()
        else:
            out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}

    return obs, info


def step_env(env, action):
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out

    return obs, float(reward), bool(done), info


def clean_action(action):
    """
    Stable-Baselines may return numpy scalars or length-1 arrays for Discrete
    actions. The env should receive a plain int for monolithic Discrete actions.
    """
    arr = np.asarray(action)

    if arr.shape == ():
        return int(arr.item())

    if arr.size == 1:
        return int(arr.reshape(-1)[0])

    return action


# ---------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------

def make_raw_env(nballs: int, dtheta: float, seed: int | None = None,
                 action_mode: str = "monolithic", rad: float | None = None):
    """
    Construct the local CiliaNBallEnv.

    Tries reset_mode='midpoint' and reset_mode='fixed' first. If your env does
    not support these keywords, it falls back to the simpler constructor.
    """
    attempts = [
        dict(Nballs=nballs, dtheta=dtheta, precompute=False, reset_mode="midpoint", action_mode=action_mode),
        dict(Nballs=nballs, dtheta=dtheta, precompute=False, reset_mode="fixed", action_mode=action_mode),
        dict(Nballs=nballs, dtheta=dtheta, precompute=False, seed=seed, action_mode=action_mode),
        dict(Nballs=nballs, dtheta=dtheta, precompute=False, action_mode=action_mode),
        dict(nballs=nballs, dtheta=dtheta, precompute=False, action_mode=action_mode),
        dict(N=nballs, dtheta=dtheta, precompute=False, action_mode=action_mode),
        dict(n_balls=nballs, dtheta=dtheta, precompute=False, action_mode=action_mode),
    ]

    if rad is not None:
        for kwargs in attempts:
            kwargs["rad"] = float(rad)

    last_err = None

    for kwargs in attempts:
        try:
            env = CiliaNBallEnv(**kwargs)

            if seed is not None:
                try:
                    env.reset(seed=seed)
                except TypeError:
                    try:
                        env.seed(seed)
                    except Exception:
                        pass

            return env

        except TypeError as e:
            last_err = e

    raise TypeError(
        "Could not construct CiliaNBallEnv with the attempted keyword sets. "
        "Edit make_raw_env() to match your local constructor."
    ) from last_err


def make_train_env(nballs: int, dtheta: float, seed: int, action_mode: str = "monolithic",
                   rad: float | None = None):
    env = make_raw_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode, rad=rad)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------

def unwrap_candidates(env: Any):
    candidates = [env]
    cur = env

    for _ in range(10):
        if hasattr(cur, "env"):
            cur = cur.env
            candidates.append(cur)
        elif hasattr(cur, "unwrapped"):
            cur = cur.unwrapped
            candidates.append(cur)
            break
        else:
            break

    return candidates


def get_attr_if_exists(env: Any, names: list[str]):
    for obj in unwrap_candidates(env):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
    return None


# ---------------------------------------------------------------------
# Geometry diagnostics
# ---------------------------------------------------------------------

def default_angle_ranges(nballs: int):
    phi_max = np.array([np.pi / 4] + [np.pi / 2] * (nballs - 1), dtype=float)
    phi_min = -phi_max
    return phi_min, phi_max


def get_n_bins(env: Any | None, nballs: int, dtheta: float):
    if env is not None:
        n_bins_attr = get_attr_if_exists(env, ["n_bins", "Nstates", "nbins"])
        if n_bins_attr is not None:
            arr = np.asarray(n_bins_attr, dtype=int).reshape(-1)
            if len(arr) >= nballs:
                return arr[:nballs]

    phi_min, phi_max = default_angle_ranges(nballs)
    widths = phi_max - phi_min
    return np.rint(widths / dtheta).astype(int) + 1


def get_dtheta(env: Any | None, fallback: float):
    if env is not None:
        dtheta_attr = get_attr_if_exists(env, ["dtheta", "dangle", "delta_theta"])
        if dtheta_attr is not None:
            try:
                val = np.asarray(dtheta_attr, dtype=float).reshape(-1)[0]
                return float(val)
            except Exception:
                pass
    return float(fallback)


def segment_lengths(env: Any | None, nballs: int):
    """
    Try to use env segment lengths. If not available, use total length 1.
    This keeps N-to-N area comparisons from being dominated by chain length.
    """
    if env is not None:
        L = get_attr_if_exists(env, ["L", "lengths", "segment_lengths"])
        if L is not None:
            arr = np.asarray(L, dtype=float).reshape(-1)
            if len(arr) >= nballs:
                return arr[:nballs].copy()

    return np.ones(nballs, dtype=float) / nballs


def infer_phi_from_obs(obs, nballs: int, env: Any | None, dtheta_fallback: float):
    """
    Convert observation to relative joint angles phi.

    Handles:
      1. direct current angle attributes if the env exposes them;
      2. normalized observations in [-1,1];
      3. integer bin-index observations.
    """
    if env is not None:
        direct = get_attr_if_exists(
            env,
            ["phi", "phis", "angles", "theta", "thetas", "joint_angles", "rel_angles"],
        )
        if direct is not None:
            arr = np.asarray(direct, dtype=float).reshape(-1)
            if len(arr) >= nballs:
                return arr[:nballs].copy()

    obs_arr = np.asarray(obs, dtype=float).reshape(-1)[:nballs]

    phi_min, phi_max = default_angle_ranges(nballs)

    # Case 1: normalized observation.
    if np.nanmax(np.abs(obs_arr)) <= 1.05:
        return obs_arr * phi_max

    # Case 2: bin indices.
    dtheta = get_dtheta(env, dtheta_fallback)
    n_bins = get_n_bins(env, nballs, dtheta)

    phi = np.zeros(nballs, dtype=float)
    for k in range(nballs):
        idx = int(round(obs_arr[k]))
        idx = max(0, min(idx, int(n_bins[k]) - 1))
        grid = np.linspace(phi_min[k], phi_max[k], int(n_bins[k]))
        phi[k] = grid[idx]

    return phi


def phi_to_tip(phi, L):
    psi = np.cumsum(phi)
    x = np.sum(L * np.sin(psi))
    z = np.sum(L * np.cos(psi))
    return np.array([x, z], dtype=float)


def closed_polygon_area(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    x2 = np.r_[x, x[0]]
    y2 = np.r_[y, y[0]]

    return float(0.5 * np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))


def path_length(points, close=True):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0

    if close:
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ---------------------------------------------------------------------
# Cycle extraction
# ---------------------------------------------------------------------

def obs_key(obs, ndigits=8):
    arr = np.asarray(obs, dtype=float).reshape(-1)
    return tuple(np.round(arr, ndigits))


def reset_to_midpoint_if_possible(env, seed=None):
    """
    Prefer a midpoint/fixed reset if your env has one. Otherwise call reset.
    """
    for name in ["reset_to_midpoint", "reset_midpoint", "set_midpoint_state"]:
        if hasattr(env, name):
            try:
                out = getattr(env, name)()
                if out is None:
                    return reset_env(env, seed=seed)
                if isinstance(out, tuple):
                    return out[0], out[1] if len(out) > 1 else {}
                return out, {}
            except Exception:
                pass

    return reset_env(env, seed=seed)


def rollout_and_extract_cycle(
    model: PPO,
    env,
    nballs: int,
    dtheta: float,
    max_steps: int = 30000,
):
    """
    Roll out deterministic PPO policy and detect first repeated observation.
    """
    obs, info = reset_to_midpoint_if_possible(env)

    seen = {}
    records = []

    Lseg = segment_lengths(env, nballs)

    for t in range(max_steps):
        key = obs_key(obs)

        if key in seen:
            cycle_start = seen[key]
            cycle_end = t
            cycle = records[cycle_start:cycle_end]
            return cycle_start, cycle_end - cycle_start, records, cycle

        seen[key] = t

        phi = infer_phi_from_obs(obs, nballs=nballs, env=env, dtheta_fallback=dtheta)
        tip = phi_to_tip(phi, Lseg)

        action, _ = model.predict(obs, deterministic=True)
        action = clean_action(action)

        next_obs, reward, done, step_info = step_env(env, action)
        is_noop = bool(step_info.get("is_noop", False)) if isinstance(step_info, dict) else False

        records.append(
            {
                "t": int(t),
                "obs": np.asarray(obs, dtype=float).copy(),
                "phi": phi.copy(),
                "tip": tip.copy(),
                "action": action,
                "reward": float(reward),
                "is_noop": is_noop,
                "done": bool(done),
            }
        )

        obs = next_obs

        # If episode terminates before a cycle is found, reset to midpoint.
        # This makes the failure visible as a short/stuck cycle if it repeats.
        if done:
            obs, info = reset_to_midpoint_if_possible(env)

    raise RuntimeError(f"No cycle detected within max_steps={max_steps}")


def summarize_cycle(cycle):
    rewards = np.array([r["reward"] for r in cycle], dtype=float)
    tips = np.array([r["tip"] for r in cycle], dtype=float)

    signed_area = closed_polygon_area(tips)
    abs_area = abs(signed_area)
    plen = path_length(tips, close=True)

    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards)) if len(rewards) else np.nan

    if abs_area > 1e-12:
        rho = total_reward / abs_area
    else:
        rho = np.nan

    noop_flags = np.array([bool(r.get("is_noop", False)) for r in cycle], dtype=bool)
    noop_count = int(noop_flags.sum())
    noop_fraction = float(noop_flags.mean()) if len(noop_flags) else 0.0

    return {
        "cycle_length": int(len(cycle)),
        "cycle_mean_reward": mean_reward,
        "cycle_total_reward": total_reward,
        "tip_signed_area": signed_area,
        "tip_abs_area": abs_area,
        "tip_path_length": plen,
        "reward_per_abs_tip_area": float(rho) if np.isfinite(rho) else np.nan,
        "wrong_orientation_flag": bool(np.isfinite(rho) and rho < 0),
        "noop_count": noop_count,
        "noop_fraction": noop_fraction,
    }


def save_result_npz(out_path: Path, records, cycle_start: int, cycle_length: int):
    obs = np.array([r["obs"] for r in records], dtype=float)
    phi = np.array([r["phi"] for r in records], dtype=float)
    tip = np.array([r["tip"] for r in records], dtype=float)
    rewards = np.array([r["reward"] for r in records], dtype=float)

    try:
        actions = np.array([r["action"] for r in records])
    except Exception:
        actions = np.array([r["action"] for r in records], dtype=object)

    np.savez(
        out_path,
        obs=obs,
        phi=phi,
        tip=tip,
        rewards=rewards,
        actions=actions,
        cycle_start=int(cycle_start),
        cycle_length=int(cycle_length),
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one(
    nballs: int,
    dtheta_label: str,
    dtheta: float,
    seed: int,
    timesteps: int,
    outdir: Path,
    action_mode: str = "monolithic",
    rad: float | None = None,
    force: bool = False,
):
    seed_dir = outdir / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model_path = seed_dir / "model.zip"
    result_path = seed_dir / "result.npz"
    summary_path = seed_dir / "summary.json"

    if summary_path.exists() and model_path.exists() and not force:
        print(f"[skip] {dtheta_label} seed={seed}; summary exists")
        with open(summary_path, "r") as f:
            return json.load(f)

    print("=" * 72)
    print(f"[train] N={nballs}, action_mode={action_mode}, dtheta={dtheta_label}, seed={seed}, timesteps={timesteps}, rad={radius_label(rad)}")
    print(f"[out]   {seed_dir}")

    train_env = make_train_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode, rad=rad)

    #model = PPO(
    #    "MlpPolicy",
    #    train_env,
    #    verbose=0,
    #    seed=seed,
    #    gamma=0.99,
    #    n_steps=2048,
    #    batch_size=256,
    #    learning_rate=3e-4,
    #    ent_coef=0.01,          # small exploration bonus; helpful for large action space
    #    device="auto",
    #)

    model = PPO(
    "MlpPolicy",
    train_env,
    verbose=0,
    seed=seed,
    gamma=0.99,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    ent_coef=0.0,
    clip_range=0.2,
    device="auto",
)






    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - t0

    model.save(model_path)

    eval_env = make_raw_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode, rad=rad)
    cycle_start, cycle_length, records, cycle = rollout_and_extract_cycle(
        model=model,
        env=eval_env,
        nballs=nballs,
        dtheta=dtheta,
        max_steps=30000,
    )

    save_result_npz(result_path, records, cycle_start, cycle_length)

    cyc = summarize_cycle(cycle)

    row = {
        "nballs": int(nballs),
        "action_mode": action_mode,
        "dtheta_label": dtheta_label,
        "dtheta": float(dtheta),
        "seed": int(seed),
        "timesteps": int(timesteps),
        "elapsed_sec": float(elapsed),
        "rad": float(rad) if rad is not None else np.nan,
        "rad_label": radius_label(rad),
        "cycle_start": int(cycle_start),
        **cyc,
        "result_path": str(result_path),
        "model_path": str(model_path),
    }

    with open(summary_path, "w") as f:
        json.dump(row, f, indent=2)

    print(
        f"[done] seed={seed}: "
        f"L={row['cycle_length']}, "
        f"mean={row['cycle_mean_reward']:.6g}, "
        f"total={row['cycle_total_reward']:.6g}, "
        f"area={row['tip_abs_area']:.6g}, "
        f"path={row['tip_path_length']:.6g}, "
        f"rho={row['reward_per_abs_tip_area']:.6g}, "
        f"wrong_orientation={row['wrong_orientation_flag']}, "
        f"noop_frac={row['noop_fraction']:.3f}"
    )

    return row


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def write_case_summary(case_dir: Path):
    rows = []

    for js in sorted(case_dir.glob("seed_*/summary.json")):
        with open(js, "r") as f:
            rows.append(json.load(f))

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("seed").reset_index(drop=True)

    out = case_dir / "summary.csv"
    df.to_csv(out, index=False)
    print(f"[summary] wrote {out}")

    return df


def write_all_summary(root: Path):
    rows = []

    for js in sorted(root.glob("N*_dtheta_*/seed_*/summary.json")):
        with open(js, "r") as f:
            rows.append(json.load(f))

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(["nballs", "dtheta_label", "seed"]).reset_index(drop=True)

    out = root / "summary_all.csv"
    df.to_csv(out, index=False)
    print(f"[summary] wrote {out}")

    return df


def print_group_summary(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("Phase 6 PPO sweep summary")
    print("=" * 72)

    for dtheta_label, g in df.groupby("dtheta_label"):
        print(f"\n{dtheta_label}:")
        print(f"  runs: {len(g)}")
        print(
            "  mean reward: "
            f"median={g['cycle_mean_reward'].median():.6g}, "
            f"min={g['cycle_mean_reward'].min():.6g}, "
            f"max={g['cycle_mean_reward'].max():.6g}"
        )
        print(
            "  cycle reward: "
            f"median={g['cycle_total_reward'].median():.6g}, "
            f"min={g['cycle_total_reward'].min():.6g}, "
            f"max={g['cycle_total_reward'].max():.6g}"
        )
        print(
            "  tip area: "
            f"median={g['tip_abs_area'].median():.6g}, "
            f"min={g['tip_abs_area'].min():.6g}, "
            f"max={g['tip_abs_area'].max():.6g}"
        )
        print(
            "  tip path length: "
            f"median={g['tip_path_length'].median():.6g}, "
            f"min={g['tip_path_length'].min():.6g}, "
            f"max={g['tip_path_length'].max():.6g}"
        )
        print(
            "  rho=reward/area: "
            f"median={g['reward_per_abs_tip_area'].median():.6g}, "
            f"min={g['reward_per_abs_tip_area'].min():.6g}, "
            f"max={g['reward_per_abs_tip_area'].max():.6g}"
        )
        print(f"  cycle lengths: {sorted(g['cycle_length'].astype(int).unique().tolist())}")
        print(f"  wrong-orientation seeds: {g.loc[g['wrong_orientation_flag'], 'seed'].tolist()}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--nballs", type=int, default=5)
    p.add_argument("--action-mode", type=str, default="monolithic",
                   choices=["monolithic", "factored"])
    p.add_argument("--cases", nargs="+", default=["pi20"], choices=sorted(DTHETA_CASES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--outroot", type=str, default="results/ppo_sweeps_phase6")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rad", type=float, default=None,
                   help="Explicit bead radius, e.g. --rad 0.04. Do not combine with --rad-scale.")
    p.add_argument("--rad-scale", type=float, default=None,
                   help="Use rad = RAD_SCALE/Nballs, e.g. --rad-scale 0.4.")

    return p.parse_args()


def main():
    args = parse_args()
    rad_effective = resolve_rad(args.nballs, rad=args.rad, rad_scale=args.rad_scale)

    print("=" * 72)
    print("[geometry]")
    print(f"N={args.nballs}")
    print(f"action_mode={args.action_mode}")
    print(f"rad={radius_label(rad_effective)}")
    if rad_effective is not None:
        seg = 1.0 / args.nballs
        thresh = 2.2 * rad_effective
        print(f"segment length = {seg:.8g}")
        print(f"2.2*rad       = {thresh:.8g}")
        print(f"seg/(2.2rad)  = {seg/thresh:.6g}")
    else:
        print("rad uses CiliaNBallEnv default")
    print("=" * 72)

    root = Path(args.outroot)
    root.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        dtheta = DTHETA_CASES[case]
        case_dir = root / f"N{args.nballs}_dtheta_{case}_{args.action_mode}"
        case_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 72)
        print(f"[case] N={args.nballs}, action_mode={args.action_mode}, {case}: dtheta={dtheta}")
        print("=" * 72)

        for seed in args.seeds:
            train_one(
                nballs=args.nballs,
                dtheta_label=case,
                dtheta=dtheta,
                seed=seed,
                timesteps=args.timesteps,
                outdir=case_dir,
                action_mode=args.action_mode,
                rad=rad_effective,
                force=args.force,
            )

            write_case_summary(case_dir)
            write_all_summary(root)

    df = write_all_summary(root)
    if df is not None:
        print_group_summary(df)

    print("\n[done] Phase 6 PPO sweep complete.")

if __name__ == "__main__":
    main()
