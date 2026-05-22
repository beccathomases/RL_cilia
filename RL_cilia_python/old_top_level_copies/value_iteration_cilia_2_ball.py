import numpy as np

from cilia_2_ball_env import Cilia2BallEnv


def deterministic_next_state(env, state, action):
    """
    Deterministic next state, using precomputed tables if available.
    """
    i, j = int(state[0]), int(state[1])

    if env.precompute and env.next_state_table is not None:
        return tuple(env.next_state_table[i, j, action, :].tolist())

    trans = env.transition_info(np.array(state, dtype=int), action)
    return tuple(trans["next_state"].tolist())


def immediate_reward(env, state, action):
    """
    One-step reward, using precomputed tables if available.
    """
    i, j = int(state[0]), int(state[1])

    if env.precompute and env.flux_table is not None:
        return float(env.flux_table[i, j, action])

    trans = env.transition_info(np.array(state, dtype=int), action)
    return env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)


def value_iteration(env, gamma=0.99, tol=1e-10, max_iter=10000, verbose=True):
    """
    Value iteration for the deterministic finite MDP defined by env.

    Returns:
        V       : shape (n0, n1)
        Q       : shape (n0, n1, nA)
        policy  : shape (n0, n1)
        history : Bellman residual history
    """
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])
    nA = env.action_space.n

    V = np.zeros((n0, n1), dtype=float)
    history = []

    for it in range(1, max_iter + 1):
        Vnew = np.zeros_like(V)

        for i in range(n0):
            for j in range(n1):
                qvals = np.empty(nA, dtype=float)

                for a in range(nA):
                    s2 = deterministic_next_state(env, (i, j), a)
                    r = immediate_reward(env, (i, j), a)
                    qvals[a] = r + gamma * V[s2]

                Vnew[i, j] = np.max(qvals)

        err = np.max(np.abs(Vnew - V))
        history.append(err)
        V = Vnew

        if verbose and (it == 1 or it % 50 == 0):
            print(f"Iter {it:4d} | ||Vnew - V||_inf = {err:.3e}")

        if err < tol:
            if verbose:
                print(f"Value iteration converged at iter {it} with err {err:.3e}")
            break
    else:
        print("Warning: reached max_iter before convergence.")

    Q = np.zeros((n0, n1, nA), dtype=float)
    policy = np.zeros((n0, n1), dtype=int)

    for i in range(n0):
        for j in range(n1):
            for a in range(nA):
                s2 = deterministic_next_state(env, (i, j), a)
                r = immediate_reward(env, (i, j), a)
                Q[i, j, a] = r + gamma * V[s2]

            policy[i, j] = int(np.argmax(Q[i, j, :]))

    return V, Q, policy, history


def canonical_cycle(cycle):
    """
    Canonical representation of a cycle, invariant under cyclic rotation.
    """
    cycle = [tuple(map(int, s)) for s in cycle]
    n = len(cycle)
    rots = [tuple(cycle[k:] + cycle[:k]) for k in range(n)]
    return min(rots)


def find_cycle(env, policy, start_state):
    """
    Deterministic cycle under a fixed policy.
    """
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
    """
    Find all unique recurrent cycles under the deterministic policy.
    """
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
    """
    Rank cycles by average one-step reward under the policy.
    """
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


def print_top_cycles(cycles_ranked, top_k=10):
    for k, C in enumerate(cycles_ranked[:top_k], start=1):
        print(f"\nCycle {k}:")
        print(f"  Average Reward: {C['avg_reward']:.6f}")
        print(f"  Cycle Length:   {C['length']}")
        print(f"  Cycle States:   {C['cycle']}")
        print(f"  Actions:        {C['actions']}")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Change this to compare the two boundary models
    # ------------------------------------------------------------
    boundary_mode = "clip_penalty"   # or "stay_penalty"

    env = Cilia2BallEnv(
        max_steps=500,
        precompute=True,
        boundary_mode=boundary_mode,
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        n_bins=[11, 21],
        angle_mins=[-np.pi / 4, -np.pi / 2],
        angle_maxs=[ np.pi / 4,  np.pi / 2],
    )

    gamma = 0.99

    V, Q, policy, history = value_iteration(
        env,
        gamma=gamma,
        tol=1e-10,
        max_iter=10000,
        verbose=True,
    )

    raw_cycles = find_all_cycles(env, policy)
    cycles = rank_cycles(env, policy, raw_cycles)

    print("\n===================================")
    print(f"Boundary mode: {boundary_mode}")
    print(f"Found {len(cycles)} unique cycles")
    print("Top 10 cycles by average reward:")
    print_top_cycles(cycles, top_k=10)

    save_name = (
        f"value_iteration_cilia_2_ball_"
        f"{boundary_mode}_"
        f"bins{env.n_bins[0]}x{env.n_bins[1]}_"
        f"g{gamma:.3f}.npy"
    )

    out = {
        "V": V,
        "Q": Q,
        "policy": policy,
        "history": np.array(history),
        "cycles": np.array(cycles, dtype=object),
        "gamma": gamma,
        "boundary_mode": boundary_mode,
        "invalid_penalty": env.invalid_penalty,
        "reward_rescale": env.reward_rescale,
        "n_bins": np.array(env.n_bins, dtype=int),
        "angle_mins": np.array(env.angle_mins, dtype=float),
        "angle_maxs": np.array(env.angle_maxs, dtype=float),
        "dangle": np.array(env.dangle, dtype=float),
    }

    np.save(save_name, out, allow_pickle=True)
    print(f"\nSaved {save_name}")