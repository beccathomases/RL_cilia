import os
import csv
import random

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from cilia_2_ball_env import Cilia2BallEnv


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


def run_ppo_and_detect_cycle(model, env, start_state=None, max_steps=2000, deterministic=True):
    obs, _ = env.reset()

    if start_state is not None:
        env.state = np.array(start_state, dtype=int)
        obs = env.state.copy()

    visited = {tuple(map(int, obs)): 0}
    states = [tuple(map(int, obs))]
    actions = []
    rewards = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        s = tuple(map(int, obs))

        actions.append(action)
        rewards.append(float(reward))
        states.append(s)

        if s in visited:
            first = visited[s]
            cycle_states = states[first:-1]
            cycle_actions = actions[first:]
            cycle_rewards = rewards[first:]
            return {
                "cycle": cycle_states,
                "actions": cycle_actions,
                "rewards": cycle_rewards,
                "avg_reward": float(np.mean(cycle_rewards)),
                "length": len(cycle_states),
            }

        visited[s] = len(states) - 1

        if terminated or truncated:
            break

    return None


# ============================================================
# Main PPO sweep
# ============================================================
if __name__ == "__main__":
    outdir = "ppo_seed_sweep_runs"
    modeldir = os.path.join(outdir, "models")
    resultdir = os.path.join(outdir, "results")
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(resultdir, exist_ok=True)

    summary_csv = os.path.join(outdir, "summary.csv")

    env_settings = {
        "max_steps": 500,
        "precompute": True,
        "boundary_mode": "clip_penalty",
        "invalid_penalty": -0.1,
        "reward_rescale": 100.0,
        "n_bins": [11, 21],
        "angle_mins": [-np.pi / 4, -np.pi / 2],
        "angle_maxs": [np.pi / 4, np.pi / 2],
    }

    # sweep choices
    seeds = [1, 2, 3, 4, 5]
    total_timesteps_list = [300_000, 1_000_000]

    # fixed PPO hyperparameters (based on your existing script)
    learning_rate = 3e-4
    n_steps = 4096
    batch_size = 128
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.02
    vf_coef = 0.5
    max_grad_norm = 0.5
    verbose = 1

    rows = []

    for total_timesteps in total_timesteps_list:
        for seed in seeds:
            run_name = f"ppo_steps{total_timesteps}_seed{seed}"
            print("\n" + "=" * 72)
            print("Running", run_name)

            # set seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            train_env = Cilia2BallEnv(**env_settings)
            check_env(train_env)
            train_env = Monitor(train_env)

            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                verbose=verbose,
                seed=seed,
            )

            model.learn(total_timesteps=total_timesteps)

            model_path = os.path.join(modeldir, run_name)
            model.save(model_path)

            eval_env = Cilia2BallEnv(**env_settings)
            center_state = (env_settings["n_bins"][0] // 2, env_settings["n_bins"][1] // 2)

            cycle_info = run_ppo_and_detect_cycle(
                model,
                eval_env,
                start_state=center_state,
                max_steps=3000,
                deterministic=True,
            )

            if cycle_info is None:
                best_len = np.nan
                best_avg = np.nan
                cycle_states = []
                cycle_actions = []
                cycle_rewards = []
            else:
                best_len = cycle_info["length"]
                best_avg = cycle_info["avg_reward"]
                cycle_states = cycle_info["cycle"]
                cycle_actions = cycle_info["actions"]
                cycle_rewards = cycle_info["rewards"]

            result_path = os.path.join(resultdir, run_name + ".npy")
            out = {
                "cycle": cycle_states,
                "actions": cycle_actions,
                "rewards": cycle_rewards,
                "avg_reward": best_avg,
                "length": best_len,
                "env_settings": env_settings,
                "train_settings": {
                    "total_timesteps": total_timesteps,
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "clip_range": clip_range,
                    "ent_coef": ent_coef,
                    "vf_coef": vf_coef,
                    "max_grad_norm": max_grad_norm,
                    "seed": seed,
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
                    "model_path": model_path + ".zip",
                    "result_path": result_path,
                }
            )

            print(f"Best cycle length: {best_len}")
            print(f"Best avg reward:   {best_avg}")
            print(f"Saved: {model_path}.zip")
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