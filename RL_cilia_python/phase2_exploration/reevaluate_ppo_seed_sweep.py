import os
import csv
import numpy as np

from stable_baselines3 import PPO
from cilia_2_ball_env import Cilia2BallEnv


def run_ppo_and_detect_cycle(model, env, start_state=None, max_steps=3000, deterministic=True):
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


if __name__ == "__main__":
    outdir = "ppo_seed_sweep_runs"
    modeldir = os.path.join(outdir, "models")
    resultdir = os.path.join(outdir, "results")
    summary_csv = os.path.join(outdir, "summary.csv")
    reevaluated_csv = os.path.join(outdir, "summary_reevaluated.csv")

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

    seeds = [1, 2, 3, 4, 5]
    total_timesteps_list = [300_000, 1_000_000]

    rows = []

    for total_timesteps in total_timesteps_list:
        for seed in seeds:
            run_name = f"ppo_steps{total_timesteps}_seed{seed}"
            model_path = os.path.join(modeldir, run_name + ".zip")
            result_path = os.path.join(resultdir, run_name + ".npy")

            print("\n" + "=" * 72)
            print("Reevaluating", run_name)

            model = PPO.load(model_path)

            # IMPORTANT: plain env, not Monitor
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

            out = {
                "cycle": cycle_states,
                "actions": cycle_actions,
                "rewards": cycle_rewards,
                "avg_reward": best_avg,
                "length": best_len,
                "env_settings": env_settings,
                "train_settings": {
                    "total_timesteps": total_timesteps,
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
                    "model_path": model_path,
                    "result_path": result_path,
                }
            )

            print(f"Best cycle length: {best_len}")
            print(f"Best avg reward:   {best_avg}")
            print(f"Updated: {result_path}")

    with open(reevaluated_csv, "w", newline="") as f:
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

    print("\nSaved reevaluated summary to:", reevaluated_csv)