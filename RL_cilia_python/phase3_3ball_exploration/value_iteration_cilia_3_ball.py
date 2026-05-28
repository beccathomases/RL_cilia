import numpy as np

from cilia_3_ball_env import Cilia3BallEnv


# ============================================================
# User settings
# ============================================================
gamma = 0.99
tol = 1e-10
max_iters = 10000
verbose_every = 50

save_file = "value_iteration_cilia_3_ball_clip_penalty_bins11x21x21_g0.990.npy"


# ============================================================
# Helpers
# ============================================================
def canonical_cycle(cycle):
    """
    Put a cycle of states into a canonical cyclic-shift-invariant form.
    """
    cyc = [tuple(map(int, s)) for s in cycle]
    if len(cyc) == 0:
        return tuple()
    rots = [tuple(cyc[k:] + cyc[:k]) for k in range(len(cyc))]
    return min(rots)


def find_cycle(env, policy, start_state):
    """
    Follow the deterministic policy until a cycle is reached.
    """
    visited = {}
    trajectory = []

    state = tuple(map(int, start_state))
    t = 0

    while state not in visited:
        visited[state] = t
        trajectory.append(state)

        i, j, k = state
        a = int(policy[i, j, k])
        state = tuple(env.next_state_table[i, j, k, a, :].tolist())
        t += 1

    cycle_start = visited[state]
    return trajectory[cycle_start:]


def find_all_cycles(env, policy):
    """
    Enumerate unique deterministic cycles under the greedy policy.
    """
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
    """
    Compute actions/rewards/avg reward for each cycle and rank them.
    """
    results = []

    for cycle in cycles:
        rewards = []
        actions = []

        for state in cycle:
            i, j, k = state
            a = int(policy[i, j, k])
            r = float(env.flux_table[i, j, k, a])
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
    env = Cilia3BallEnv(
        max_steps=500,
        precompute=True,
        boundary_mode="clip_penalty",
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        n_bins=[11, 21, 21],
        angle_mins=[-np.pi / 4, -np.pi / 2, -np.pi / 2],
        angle_maxs=[ np.pi / 4,  np.pi / 2,  np.pi / 2],
    )

    if env.flux_table is None or env.next_state_table is None:
        raise RuntimeError("This VI script requires precompute=True.")

    n0, n1, n2 = map(int, env.n_bins)
    nA = env.action_space.n

    V = np.zeros((n0, n1, n2), dtype=float)
    history = []

    print("Starting value iteration...")
    print(f"State space: {n0} x {n1} x {n2} = {n0*n1*n2}")
    print(f"Actions: {nA}")

    for it in range(1, max_iters + 1):
        Vnew = np.empty_like(V)

        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    qvals = np.empty(nA, dtype=float)
                    for a in range(nA):
                        ns = env.next_state_table[i, j, k, a, :]
                        r = env.flux_table[i, j, k, a]
                        qvals[a] = r + gamma * V[ns[0], ns[1], ns[2]]

                    Vnew[i, j, k] = np.max(qvals)

        err = np.max(np.abs(Vnew - V))
        history.append(err)

        if it % verbose_every == 0 or err < tol:
            print(f"Iter {it:4d} | ||Vnew - V||_inf = {err:.3e}")

        V = Vnew

        if err < tol:
            print(f"Value iteration converged at iter {it} with err {err:.3e}")
            converged = True
            break
    else:
        print(f"WARNING: did not converge within {max_iters} iterations.")
        converged = False
        it = max_iters
        err = history[-1]

    # --------------------------------------------------------
    # Extract greedy policy and Q table
    # --------------------------------------------------------
    Q = np.empty((n0, n1, n2, nA), dtype=float)
    policy = np.zeros((n0, n1, n2), dtype=int)

    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                qvals = np.empty(nA, dtype=float)
                for a in range(nA):
                    ns = env.next_state_table[i, j, k, a, :]
                    r = env.flux_table[i, j, k, a]
                    qvals[a] = r + gamma * V[ns[0], ns[1], ns[2]]

                Q[i, j, k, :] = qvals
                policy[i, j, k] = int(np.argmax(qvals))

    # --------------------------------------------------------
    # Find cycles
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out = {
        "V": V,
        "Q": Q,
        "policy": policy,
        "history": np.array(history, dtype=float),
        "converged": converged,
        "n_iters": it,
        "final_err": err,
        "gamma": gamma,
        "cycles": np.array(cycles, dtype=object),
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