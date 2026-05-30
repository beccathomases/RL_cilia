import os
import csv
import random

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from cilia_3_ball_env import Cilia3BallEnv


# ============================================================
# Utilities
# ============================================================
def deterministic_next_state(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return tuple(trans["next_state"].tolist())


def immediate_reward(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)


def canonical_cycle(cycle):
    cyc = [tuple(map(int, s)) for s in cycle]
    if len(cyc) == 0:
        return tuple()
    rots = [tuple(cyc[k:] + cyc[:k]) for k in range(len(cyc))]
    return min(rots)


def find_cycle(env, policy, start_state):
    visited = {}
    trajectory = []

    state = tuple(map(int, start_state))
    t = 0

    while state not in visited:
        visited[state] = t
        trajectory.append(state)

        i, j, k = state
        a = int(policy[i, j, k])
        state = deterministic_next_state(env, state, a)
        t += 1

    cycle_start = visited[state]
    return trajectory[cycle_start:]


def find_all_cycles(env, policy):
    n0, n1, n2 = map(int, env.n_bins)

    seen = set()
    unique_cycles = []

    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                cyc = find_cycle(env, policy, (i, j, k))
                canon = canonical_cycle(cyc)
                if canon not in seen:
                    seen.add(canon)
                    unique_cycles.append(list(canon))

    return unique_cycles


def rank_cycles(env, policy, cycles):
    results = []

    for cycle in cycles:
        rewards = []
        actions = []

        for state in cycle:
            i, j, k = state
            a = int(policy[i, j, k])
            r = immediate_reward(env, state, a)
            rewards.append(r)
            actions.append(a)

        avg_reward = float(np.mean(rewards))
        results.append(
            {
                "cycle": cycle,
                "actions": actions,
                "rewards": rewards,
                "avg_reward": avg_reward,
                "length": len(cycle),
            }
        )

    results.sort(key=lambda d: (-d["avg_reward"], -d["length"]))
    return results


def extract_policy(model, env):
    n0, n1, n2 = map(int, env.n_bins)
    policy = np.zeros((n0, n1, n2), dtype=int)

    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                obs = np.array([i, j, k], dtype=np.int64)
                action, _ = model.predict(obs, deterministic=True)
                policy[i, j, k] = int(action)

    return policy


# ============================================================
# PPO training
# ============================================================
def make_env_fn(env_settings, seed):
    def _init():
        env = Cilia3BallEnv(**env_settings)
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass
        try:
            env.observation_space.seed(seed)
        except AttributeError:
            pass
        return env
    return _init


def train_ppo(env_settings, total_timesteps=300000, seed=0, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_vec_env(
        make_env_fn(env_settings, seed),
        n_envs=1,
        seed=seed,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    model.learn(total_timesteps=total_timesteps)
    return model


# ============================================================
# Main small seed sweep
# ============================================================
if __name__ == "__main__":
    outdir = "ppo_3ball_runs"
    modeldir = os.path.join(outdir, "models")
    resultdir = os.path.join(outdir, "results")
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(resultdir, exist_ok=True)

    summary_csv = os.path.join(outdir, "summary.csv")

    env_settings = {
        "max_steps": 1500,
        "precompute": True,
        "boundary_mode": "clip_penalty",
        "invalid_penalty": -0.1,
        "reward_rescale": 100.0,
        "n_bins": [11, 21, 21],
        "angle_mins": [-np.pi / 4, -np.pi / 2, -np.pi / 2],
        "angle_maxs": [ np.pi / 4,  np.pi / 2,  np.pi / 2],
    }

    # conservative first PPO sweep
    total_timesteps_list = [300000, 1000000]
    seeds = [0, 1, 2]

    device = "cpu"
    rows = []

    for total_timesteps in total_timesteps_list:
        for seed in seeds:
            run_name = f"ppo_3ball_steps{total_timesteps}_seed{seed}"
            print("\n" + "=" * 72)
            print("Running", run_name)

            model = train_ppo(
                env_settings,
                total_timesteps=total_timesteps,
                seed=seed,
                device=device,
            )

            # analyze with a fresh plain env
            env = Cilia3BallEnv(**env_settings)
            policy = extract_policy(model, env)
            raw_cycles = find_all_cycles(env, policy)
            cycles = rank_cycles(env, policy, raw_cycles)

            if len(cycles) > 0:
                best = cycles[0]
                best_len = best["length"]
                best_avg = best["avg_reward"]
            else:
                best_len = np.nan
                best_avg = np.nan

            model_path = os.path.join(modeldir, run_name + ".zip")
            result_path = os.path.join(resultdir, run_name + ".npy")

            model.save(model_path)

            out = {
                "policy": policy,
                "cycles": np.array(cycles, dtype=object),
                "env_settings": env_settings,
                "train_settings": {
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "algo": "PPO",
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "net_arch": [64, 64],
                },
            }
            np.save(result_path, out, allow_pickle=True)

            rows.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "best_cycle_length": best_len,
                    "best_avg_reward": best_avg,
                    "model_path": model_path,
                    "result_path": result_path,
                }
            )

            print(f"Best cycle length: {best_len}")
            print(f"Best avg reward:   {best_avg}")
            print(f"Saved: {model_path}")
            print(f"Saved: {result_path}")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "seed",
                "total_timesteps",
                "best_cycle_length",
                "best_avg_reward",
                "model_path",
                "result_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved summary to:", summary_csv)