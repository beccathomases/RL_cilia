import os
import csv
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# Q network
# ============================================================
class QNet(nn.Module):
    def __init__(self, input_dim=2, hidden=64, n_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Replay buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Utilities
# ============================================================
def update_target(q_net, target_net):
    target_net.load_state_dict(q_net.state_dict())


def state_scale(env):
    return np.array(env.n_bins, dtype=np.float32) - 1.0


def greedy_action(q_net, state_vec, device="cpu"):
    with torch.no_grad():
        s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = q_net(s)
        return int(torch.argmax(qvals, dim=1).item())


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

        i, j = state
        a = int(policy[i, j])
        state = deterministic_next_state(env, state, a)
        t += 1

    cycle_start = visited[state]
    return trajectory[cycle_start:]


def find_all_cycles(env, policy):
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])

    seen = set()
    unique_cycles = []

    for i in range(n0):
        for j in range(n1):
            cyc = find_cycle(env, policy, (i, j))
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
            i, j = state
            a = int(policy[i, j])
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


def extract_policy(q_net, env, device="cpu"):
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])
    policy = np.zeros((n0, n1), dtype=int)

    scale = state_scale(env)

    for i in range(n0):
        for j in range(n1):
            s = np.array([i, j], dtype=np.float32) / scale
            with torch.no_grad():
                inp = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = q_net(inp)
            policy[i, j] = int(torch.argmax(q_vals, dim=1).item())

    return policy


# ============================================================
# DQN training loop
# ============================================================
def train_dqn(
    env,
    episodes=10000,
    gamma=0.99,
    lr=5e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.9995,
    batch_size=64,
    replay_capacity=50000,
    min_replay_size=500,
    target_update_freq=50,
    reward_divisor=None,
    reward_clip=None,
    seed=0,
    device="cpu",
    verbose_every=500,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    try:
        env.observation_space.seed(seed)
    except AttributeError:
        pass

    q_net = QNet(input_dim=2, hidden=64, n_actions=env.action_space.n).to(device)
    target_net = QNet(input_dim=2, hidden=64, n_actions=env.action_space.n).to(device)
    update_target(q_net, target_net)
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    replay = ReplayBuffer(capacity=replay_capacity)
    epsilon = epsilon_start

    episode_rewards = []
    episode_lengths = []
    loss_history = []
    q_means = []
    grad_norms = []

    scale = state_scale(env)

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32) / scale

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(q_net, state, device=device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32) / scale

            reward_used = float(reward)
            if reward_divisor is not None:
                reward_used = reward_used / reward_divisor
            if reward_clip is not None:
                reward_used = float(np.clip(reward_used, reward_clip[0], reward_clip[1]))

            replay.add(state, action, reward_used, next_state, done)

            total_reward += reward_used
            steps += 1
            state = next_state

            if len(replay) >= max(batch_size, min_replay_size):
                s, a, r, s2, d = replay.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.int64, device=device)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q_vals = q_net(s)
                q_sa = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(s2).max(dim=1)[0]
                    target = r + gamma * q_next * (1.0 - d)

                loss = loss_fn(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

                grad_norms.append(float(grad_norm))
                loss_history.append(float(loss.item()))

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if ep % target_update_freq == 0:
            update_target(q_net, target_net)

        with torch.no_grad():
            sample = torch.zeros((1, 2), dtype=torch.float32, device=device)
            q_means.append(float(q_net(sample).mean().item()))

        if verbose_every is not None and ep % verbose_every == 0:
            print(
                f"Episode {ep:5d} | "
                f"reward(last50)={np.mean(episode_rewards[-50:]): .4f} | "
                f"len(last50)={np.mean(episode_lengths[-50:]): .1f} | "
                f"eps={epsilon: .4f}"
            )

    diagnostics = {
        "episode_rewards": np.array(episode_rewards, dtype=float),
        "episode_lengths": np.array(episode_lengths, dtype=float),
        "loss_history": np.array(loss_history, dtype=float),
        "q_means": np.array(q_means, dtype=float),
        "grad_norms": np.array(grad_norms, dtype=float),
    }

    return q_net, diagnostics


# ============================================================
# Main focused follow-up
# ============================================================
if __name__ == "__main__":
    outdir = "dqn_longer_followup_runs"
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

    lr = 5e-4
    episode_list = [15000, 20000]
    seeds = [1, 2, 3]

    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    batch_size = 64
    replay_capacity = 50000
    min_replay_size = 500
    target_update_freq = 50
    reward_divisor = None
    reward_clip = None
    device = "cpu"

    rows = []

    for episodes in episode_list:
        for seed in seeds:
            run_name = f"dqn_lr5e-04_ep{episodes}_seed{seed}"
            print("\n" + "=" * 72)
            print("Running", run_name)

            env = Cilia2BallEnv(**env_settings)

            qnet, diagnostics = train_dqn(
                env,
                episodes=episodes,
                gamma=gamma,
                lr=lr,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                batch_size=batch_size,
                replay_capacity=replay_capacity,
                min_replay_size=min_replay_size,
                target_update_freq=target_update_freq,
                reward_divisor=reward_divisor,
                reward_clip=reward_clip,
                seed=seed,
                device=device,
                verbose_every=500,
            )

            policy = extract_policy(qnet, env, device=device)
            raw_cycles = find_all_cycles(env, policy)
            cycles = rank_cycles(env, policy, raw_cycles)

            if len(cycles) > 0:
                best = cycles[0]
                best_len = best["length"]
                best_avg = best["avg_reward"]
            else:
                best_len = np.nan
                best_avg = np.nan

            model_path = os.path.join(modeldir, run_name + ".pt")
            result_path = os.path.join(resultdir, run_name + ".npy")

            torch.save(qnet.state_dict(), model_path)

            out = {
                "policy": policy,
                "cycles": np.array(cycles, dtype=object),
                "diagnostics": diagnostics,
                "env_settings": env_settings,
                "train_settings": {
                    "episodes": episodes,
                    "gamma": gamma,
                    "lr": lr,
                    "epsilon_start": epsilon_start,
                    "epsilon_end": epsilon_end,
                    "epsilon_decay": epsilon_decay,
                    "batch_size": batch_size,
                    "replay_capacity": replay_capacity,
                    "min_replay_size": min_replay_size,
                    "target_update_freq": target_update_freq,
                    "reward_divisor": reward_divisor,
                    "reward_clip": reward_clip,
                    "seed": seed,
                },
            }
            np.save(result_path, out, allow_pickle=True)

            rows.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "lr": lr,
                    "episodes": episodes,
                    "epsilon_decay": epsilon_decay,
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
                "lr",
                "episodes",
                "epsilon_decay",
                "best_cycle_length",
                "best_avg_reward",
                "model_path",
                "result_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved summary to:", summary_csv)