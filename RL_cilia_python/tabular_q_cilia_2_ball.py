import numpy as np

from cilia_2_ball_env import Cilia2BallEnv


def greedy_action(Q, state):
    i, j = state
    return int(np.argmax(Q[i, j, :]))


def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return greedy_action(Q, state)


def canonical_cycle(cycle):
    cycle = [tuple(map(int, s)) for s in cycle]
    n = len(cycle)
    rots = [tuple(cycle[k:] + cycle[:k]) for k in range(n)]
    return min(rots)


def deterministic_next_state_from_Q(env, Q, start_state, max_steps=5000):
    """
    Follow greedy policy induced by Q from a start state until a state repeats.
    """
    visited = {}
    trajectory = []
    rewards = []
    actions = []

    state = tuple(map(int, start_state))

    for t in range(max_steps):
        if state in visited:
            cyc_start = visited[state]
            return {
                "cycle": trajectory[cyc_start:],
                "actions": actions[cyc_start:],
                "rewards": rewards[cyc_start:],
            }

        visited[state] = t
        trajectory.append(state)

        a = greedy_action(Q, state)
        s2, r = env_step_no_mutation(env, state, a)

        actions.append(a)
        rewards.append(r)
        state = s2

    return None


def find_all_cycles_from_Q(env, Q, max_steps=5000):
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])

    seen = set()
    unique_cycles = []

    for i in range(n0):
        for j in range(n1):
            out = deterministic_next_state_from_Q(env, Q, (i, j), max_steps=max_steps)
            if out is None:
                continue
            canon = canonical_cycle(out["cycle"])
            if canon not in seen:
                seen.add(canon)
                unique_cycles.append(out)

    return unique_cycles


def rank_cycles(cycles):
    ranked = []
    for C in cycles:
        avg_reward = float(np.mean(C["rewards"]))
        ranked.append(
            {
                "cycle": C["cycle"],
                "actions": C["actions"],
                "rewards": C["rewards"],
                "avg_reward": avg_reward,
                "length": len(C["cycle"]),
            }
        )
    ranked.sort(key=lambda d: (-d["avg_reward"], -d["length"]))
    return ranked


def env_step_no_mutation(env, state, action):
    """
    Use env's transition logic without mutating env.state.
    """
    trans = env.transition_info(np.array(state, dtype=int), action)
    reward = env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)
    next_state = tuple(trans["next_state"].tolist())
    return next_state, float(reward)


def q_learning(
    env,
    n_episodes=1000,
    max_steps=500,
    alpha0=0.99,
    gamma=0.99,
    epsilon0=0.75,
    alpha_floor=0.05,
    epsilon_floor=0.02,
    decay=0.999,
    seed=1,
    verbose=True,
):
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])
    nA = env.action_space.n

    rng = np.random.default_rng(seed)
    Q = np.zeros((n0, n1, nA), dtype=float)
    ep_returns = np.zeros(n_episodes, dtype=float)

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep + 1)
        state = tuple(map(int, state))

        alpha = max(alpha_floor, alpha0 * (decay ** ep))
        epsilon = max(epsilon_floor, epsilon0 * (decay ** ep))

        G = 0.0

        for _ in range(max_steps):
            a = epsilon_greedy_action(Q, state, nA, epsilon, rng)
            next_state_arr, r, terminated, truncated, _ = env.step(a)
            next_state = tuple(map(int, next_state_arr))

            i, j = state
            ni, nj = next_state

            td_target = r + gamma * np.max(Q[ni, nj, :])
            td_error = td_target - Q[i, j, a]
            Q[i, j, a] += alpha * td_error

            G += r
            state = next_state

            if terminated or truncated:
                break

        ep_returns[ep] = G

        if verbose and ((ep + 1) % 100 == 0 or ep == 0):
            print(
                f"Episode {ep+1:4d} | return = {G: .6f} | "
                f"alpha = {alpha:.4f} | epsilon = {epsilon:.4f}"
            )

    return Q, ep_returns


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Settings
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

    n_episodes = 1000
    max_steps = 500
    alpha0 = 0.99
    gamma = 0.99
    epsilon0 = 0.75
    seeds = range(1, 11)

    all_results = []

    for seed in seeds:
        print("\n" + "=" * 60)
        print(f"Seed {seed}")

        Q, ep_returns = q_learning(
            env,
            n_episodes=n_episodes,
            max_steps=max_steps,
            alpha0=alpha0,
            gamma=gamma,
            epsilon0=epsilon0,
            alpha_floor=0.05,
            epsilon_floor=0.02,
            decay=0.999,
            seed=seed,
            verbose=False,
        )

        cycles = rank_cycles(find_all_cycles_from_Q(env, Q))

        if cycles:
            best = cycles[0]
            print(f"Best cycle length   = {best['length']}")
            print(f"Best avg reward     = {best['avg_reward']:.6f}")
            print(f"Best cycle states   = {best['cycle']}")
            print(f"Best cycle actions  = {best['actions']}")
        else:
            best = None
            print("No cycle found.")

        all_results.append(
            {
                "seed": seed,
                "Q": Q,
                "ep_returns": ep_returns,
                "cycles": np.array(cycles, dtype=object),
            }
        )

    save_name = (
        f"tabular_q_cilia_2_ball_{boundary_mode}_"
        f"bins{env.n_bins[0]}x{env.n_bins[1]}_"
        f"ep{n_episodes}_steps{max_steps}_"
        f"g{gamma:.3f}_eps{epsilon0:.2f}_a{alpha0:.2f}.npy"
    )

    out = {
        "results": np.array(all_results, dtype=object),
        "boundary_mode": boundary_mode,
        "n_bins": np.array(env.n_bins, dtype=int),
        "angle_mins": np.array(env.angle_mins, dtype=float),
        "angle_maxs": np.array(env.angle_maxs, dtype=float),
        "gamma": gamma,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "alpha0": alpha0,
        "epsilon0": epsilon0,
    }

    np.save(save_name, out, allow_pickle=True)
    print(f"\nSaved {save_name}")