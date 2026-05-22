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


def normalize_state(state, env):
    return np.array(state, dtype=np.float32) / state_scale(env)


def greedy_action(q_net, state_vec, device="cpu"):
    with torch.no_grad():
        s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = q_net(s)
        return int(torch.argmax(qvals, dim=1).item())


# ============================================================
# Cycle detection for a deterministic policy table
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


# ============================================================
# Extract deterministic policy from trained Q net
# ============================================================
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
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # extra reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # seed env / gym RNGs too
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
    action_counts = np.zeros(env.action_space.n, dtype=int)

    scale = state_scale(env)

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32) / scale

        done = False
        total_reward = 0.0
        steps = 0
        ep_losses = []

        while not done:
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(q_net, state, device=device)

            action_counts[action] += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32) / scale

            # optional reward scaling
            reward_used = float(reward)
            if reward_divisor is not None:
                reward_used = reward_used / reward_divisor
            if reward_clip is not None:
                reward_used = float(np.clip(reward_used, reward_clip[0], reward_clip[1]))

            replay.add(state, action, reward_used, next_state, done)

            total_reward += reward_used
            steps += 1
            state = next_state

            # learning step
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
                ep_losses.append(float(loss.item()))
                loss_history.append(float(loss.item()))

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if ep % target_update_freq == 0:
            update_target(q_net, target_net)

        with torch.no_grad():
            sample = torch.zeros((1, 2), dtype=torch.float32, device=device)
            q_means.append(float(q_net(sample).mean().item()))

        if ep % 50 == 0:
            print("\nEpisode:", ep)
            print("Reward (last 50):", np.mean(episode_rewards[-50:]))
            print("Length (last 50):", np.mean(episode_lengths[-50:]))
            print("Epsilon:", epsilon)
            print("Q mean (last 50):", np.mean(q_means[-50:]))
            if len(grad_norms) > 0:
                print("Grad norm (last 10):", np.mean(grad_norms[-10:]))
            if len(ep_losses) > 0:
                print("Loss (this ep mean):", np.mean(ep_losses))
            print("Action histogram:", action_counts)

    diagnostics = {
        "episode_rewards": np.array(episode_rewards, dtype=float),
        "episode_lengths": np.array(episode_lengths, dtype=float),
        "loss_history": np.array(loss_history, dtype=float),
        "q_means": np.array(q_means, dtype=float),
        "grad_norms": np.array(grad_norms, dtype=float),
        "action_counts": action_counts.copy(),
    }

    return q_net, diagnostics


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------
    env = Cilia2BallEnv(
        max_steps=500,
        precompute=True,
        boundary_mode="clip_penalty",
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        n_bins=[11, 21],
        angle_mins=[-np.pi / 4, -np.pi / 2],
        angle_maxs=[ np.pi / 4,  np.pi / 2],
    )

    # Try reward_divisor=1.0 first if you want raw rewards.
    # If training is unstable, try reward_divisor=1.0 or 10.0 and/or clipping.
    qnet, diagnostics = train_dqn(
        env,
        episodes=5000,
        gamma=0.99,
        lr=1e-3,
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
    )

    policy = extract_policy(qnet, env, device="cpu")
    raw_cycles = find_all_cycles(env, policy)
    cycles = rank_cycles(env, policy, raw_cycles)

    print("\nTop 10 cycles by average reward:")
    for i, C in enumerate(cycles[:10], start=1):
        print(f"\nCycle {i}:")
        print(f"  Average Reward: {C['avg_reward']:.6f}")
        print(f"  Cycle States:   {C['cycle']}")
        print(f"  Cycle Length:   {C['length']}")

    # save model weights + diagnostics + policy + cycles
    torch.save(qnet.state_dict(), "dqn_cilia_2_ball.pt")

    out = {
        "policy": policy,
        "cycles": np.array(cycles, dtype=object),
        "diagnostics": diagnostics,
        "env_settings": {
            "max_steps": 500,
            "precompute": True,
            "boundary_mode": "clip_penalty",
            "invalid_penalty": -0.1,
            "reward_rescale": 100.0,
            "n_bins": [11, 21],
            "angle_mins": [-np.pi / 4, -np.pi / 2],
            "angle_maxs": [ np.pi / 4,  np.pi / 2],
        },
        "train_settings": {
            "episodes": 5000,
            "gamma": 0.99,
            "lr": 1e-3,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.9995,
            "batch_size": 64,
            "replay_capacity": 50000,
            "min_replay_size": 500,
            "target_update_freq": 50,
            "reward_divisor": None,
            "reward_clip": None,
            "seed": 0,
        },
    }

    np.save("dqn_cilia_2_ball_results.npy", out, allow_pickle=True)
    print("\nSaved dqn_cilia_2_ball.pt")
    print("Saved dqn_cilia_2_ball_results.npy")