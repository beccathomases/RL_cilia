import random
from collections import defaultdict

import numpy as np

from cilia_3_ball_env import Cilia3BallEnv


# ============================================================
# User settings
# ============================================================
gamma = 0.99
alpha0 = 0.99
alpha_floor = 0.05
epsilon0 = 0.75
epsilon_floor = 0.05

n_episodes = 10000
max_steps = 1500
seed = 4

verbose_every = 100



save_file = (
    "tabular_q_cilia_3_ball_clip_penalty_bins11x21x21_"
    f"ep{n_episodes}_steps{max_steps}_g{gamma:.3f}_"
    f"eps{epsilon0:.2f}_a{alpha0:.2f}_seed{seed}.npy"
)


# ============================================================
# Helpers
# ============================================================
def alpha_schedule(ep, alpha0=0.99, alpha_floor=0.05):
    """
    Simple decaying learning rate.
    """
    a = alpha0 / (1.0 + 0.002 * ep)
    return max(alpha_floor, a)


def epsilon_schedule(ep, epsilon0=0.75, epsilon_floor=0.05):
    """
    Simple decaying exploration rate.
    """
    e = epsilon0 / (1.0 + 0.002 * ep)
    return max(epsilon_floor, e)


def greedy_policy_from_Q(Q):
    return np.argmax(Q, axis=-1)


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
        if env.precompute and env.next_state_table is not None:
            state = tuple(env.next_state_table[i, j, k, a, :].tolist())
        else:
            trans = env.transition_info(np.array(state, dtype=int), a)
            state = tuple(trans["next_state"].tolist())
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

            if env.precompute and env.flux_table is not None:
                r = float(env.flux_table[i, j, k, a])
            else:
                trans = env.transition_info(np.array(state, dtype=int), a)
                r = float(env.immediate_reward_from_transition(np.array(state, dtype=int), a, trans))

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
# Main
# ============================================================
if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)

    env = Cilia3BallEnv(
        max_steps=max_steps,
        precompute=True,
        boundary_mode="clip_penalty",
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        n_bins=[11, 21, 21],
        angle_mins=[-np.pi / 4, -np.pi / 2, -np.pi / 2],
        angle_maxs=[ np.pi / 4,  np.pi / 2,  np.pi / 2],
    )

    n0, n1, n2 = map(int, env.n_bins)
    nA = env.action_space.n

    Q = np.zeros((n0, n1, n2, nA), dtype=float)

    episode_rewards = []
    episode_lengths = []
    best_cycle_history = []

    print("Starting tabular Q-learning...")
    print(f"State space: {n0} x {n1} x {n2} = {n0*n1*n2}")
    print(f"Actions: {nA}")
    print(f"Episodes: {n_episodes}")
    print(f"Seed: {seed}")

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0

        alpha = alpha_schedule(ep, alpha0=alpha0, alpha_floor=alpha_floor)
        epsilon = epsilon_schedule(ep, epsilon0=epsilon0, epsilon_floor=epsilon_floor)

        for t in range(max_steps):
            i, j, k = state

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[i, j, k, :]))

            next_state, reward, terminated, truncated, info = env.step(action)
            ni, nj, nk = next_state

            target = reward + gamma * np.max(Q[ni, nj, nk, :])
            Q[i, j, k, action] = (1 - alpha) * Q[i, j, k, action] + alpha * target

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(float(total_reward))
        episode_lengths.append(t + 1)

        # occasional diagnostics on greedy policy
        if (ep + 1) % verbose_every == 0 or ep == n_episodes - 1:
            policy = greedy_policy_from_Q(Q)
            raw_cycles = find_all_cycles(env, policy)
            ranked = rank_cycles(env, policy, raw_cycles)
            if len(ranked) > 0:
                best_cycle_history.append((ep + 1, ranked[0]["length"], ranked[0]["avg_reward"]))
                print(
                    f"Episode {ep+1:4d} | "
                    f"reward(last50)={np.mean(episode_rewards[-50:]): .4f} | "
                    f"len(last50)={np.mean(episode_lengths[-50:]): .1f} | "
                    f"eps={epsilon: .4f} | "
                    f"best_cycle_len={ranked[0]['length']} | "
                    f"best_cycle_avg={ranked[0]['avg_reward']:.6f}"
                )
            else:
                print(
                    f"Episode {ep+1:4d} | "
                    f"reward(last50)={np.mean(episode_rewards[-50:]): .4f} | "
                    f"len(last50)={np.mean(episode_lengths[-50:]): .1f} | "
                    f"eps={epsilon: .4f} | "
                    f"no cycle found"
                )

    # --------------------------------------------------------
    # Final policy and cycles
    # --------------------------------------------------------
    policy = greedy_policy_from_Q(Q)
    raw_cycles = find_all_cycles(env, policy)
    cycles = rank_cycles(env, policy, raw_cycles)

    print("\n===================================")
    print(f"Boundary mode: {env.boundary_mode}")
    print(f"Found {len(cycles)} unique cycles")
    print("Top 10 cycles by average reward:\n")

    for m, cyc in enumerate(cycles[:10], start=1):
        print(f"Cycle {m}:")
        print(f"  Average Reward: {cyc['avg_reward']:.6f}")
        print(f"  Cycle Length:   {cyc['length']}")
        print(f"  Cycle States:   {cyc['cycle']}")
        print(f"  Actions:        {cyc['actions']}\n")

    out = {
        "Q": Q,
        "policy": policy,
        "episode_rewards": np.array(episode_rewards, dtype=float),
        "episode_lengths": np.array(episode_lengths, dtype=float),
        "best_cycle_history": np.array(best_cycle_history, dtype=object),
        "cycles": np.array(cycles, dtype=object),
        "train_settings": {
            "gamma": gamma,
            "alpha0": alpha0,
            "alpha_floor": alpha_floor,
            "epsilon0": epsilon0,
            "epsilon_floor": epsilon_floor,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "seed": seed,
        },
        "env_settings": {
            "max_steps": env.max_steps,
            "precompute": env.precompute,
            "boundary_mode": env.boundary_mode,
            "invalid_penalty": env.invalid_penalty,
            "reward_rescale": env.reward_rescale,
            "n_bins": env.n_bins.tolist(),
            "angle_mins": env.angle_mins.tolist(),
            "angle_maxs": env.angle_maxs.tolist(),
        },
    }

    np.save(save_file, out, allow_pickle=True)
    print(f"Saved {save_file}")