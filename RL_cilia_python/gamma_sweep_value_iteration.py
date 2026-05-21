import numpy as np
import pandas as pd

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# Helpers
# ============================================================
def deterministic_next_state(env, state, action):
    i, j = int(state[0]), int(state[1])

    if env.precompute and env.next_state_table is not None:
        return tuple(env.next_state_table[i, j, action, :].tolist())

    trans = env.transition_info(np.array(state, dtype=int), action)
    return tuple(trans["next_state"].tolist())


def immediate_reward(env, state, action):
    i, j = int(state[0]), int(state[1])

    if env.precompute and env.flux_table is not None:
        return float(env.flux_table[i, j, action])

    trans = env.transition_info(np.array(state, dtype=int), action)
    return env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)


def value_iteration(env, gamma=0.99, tol=1e-10, max_iter=30000, verbose=True):
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])
    nA = env.action_space.n

    V = np.zeros((n0, n1), dtype=float)
    history = []
    converged = False

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

        if verbose and (it == 1 or it % 100 == 0):
            print(f"Iter {it:5d} | ||Vnew - V||_inf = {err:.3e}")

        if err < tol:
            converged = True
            if verbose:
                print(f"Value iteration converged at iter {it} with err {err:.3e}")
            break

    if not converged:
        print(f"Warning: reached max_iter={max_iter} before convergence. Final err = {history[-1]:.3e}")

    Q = np.zeros((n0, n1, nA), dtype=float)
    policy = np.zeros((n0, n1), dtype=int)

    for i in range(n0):
        for j in range(n1):
            for a in range(nA):
                s2 = deterministic_next_state(env, (i, j), a)
                r = immediate_reward(env, (i, j), a)
                Q[i, j, a] = r + gamma * V[s2]

            policy[i, j] = int(np.argmax(Q[i, j, :]))

    return V, Q, policy, history, converged


def canonical_cycle(cycle):
    cycle = [tuple(map(int, s)) for s in cycle]
    n = len(cycle)
    rots = [tuple(cycle[k:] + cycle[:k]) for k in range(n)]
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

    # still rank by average cycle reward for convenience
    results.sort(key=lambda d: (-d["avg_reward"], -d["length"]))
    return results


def discounted_cycle_values(rewards, gamma):
    rewards = np.array(rewards, dtype=float)
    L = len(rewards)
    vals = np.zeros(L, dtype=float)

    denom = 1.0 - gamma**L
    for i in range(L):
        one_lap = 0.0
        for k in range(L):
            one_lap += (gamma**k) * rewards[(i + k) % L]
        vals[i] = one_lap / denom

    return vals


# ============================================================
# Main gamma sweep
# ============================================================
if __name__ == "__main__":
    gammas = [0.90, 0.95, 0.98, 0.99, 0.992, 0.995, 0.997, 0.999]

    env_kwargs = dict(
        max_steps=500,
        precompute=True,
        boundary_mode="clip_penalty",
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        n_bins=[11, 21],
        angle_mins=[-np.pi / 4, -np.pi / 2],
        angle_maxs=[ np.pi / 4,  np.pi / 2],
    )

    tol = 1e-10
    max_iter = 30000

    rows = []
    all_results = {}

    for gamma in gammas:
        print("\n" + "=" * 72)
        print(f"Running value iteration for gamma = {gamma}")

        env = Cilia2BallEnv(**env_kwargs)

        V, Q, policy, history, converged = value_iteration(
            env,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter,
            verbose=True,
        )

        raw_cycles = find_all_cycles(env, policy)
        cycles = rank_cycles(env, policy, raw_cycles)

        if len(cycles) == 0:
            print("No cycles found.")
            rows.append(
                {
                    "gamma": gamma,
                    "cycle_length": np.nan,
                    "avg_reward": np.nan,
                    "disc_best": np.nan,
                    "disc_mean": np.nan,
                    "vi_iters": len(history),
                    "final_err": history[-1],
                    "converged": converged,
                }
            )
            all_results[gamma] = {
                "V": V,
                "Q": Q,
                "policy": policy,
                "history": history,
                "converged": converged,
                "cycles": cycles,
            }
            continue

        best = cycles[0]
        disc_vals = discounted_cycle_values(best["rewards"], gamma)

        print(f"Found {len(cycles)} unique cycles")
        print(f"Best cycle length = {best['length']}")
        print(f"Average reward    = {best['avg_reward']:.6f}")
        print(f"Best disc value   = {np.max(disc_vals):.6f}")
        print(f"Mean disc value   = {np.mean(disc_vals):.6f}")
        print(f"Cycle states      = {best['cycle']}")

        rows.append(
            {
                "gamma": gamma,
                "cycle_length": best["length"],
                "avg_reward": best["avg_reward"],
                "disc_best": np.max(disc_vals),
                "disc_mean": np.mean(disc_vals),
                "vi_iters": len(history),
                "final_err": history[-1],
                "converged": converged,
            }
        )

        all_results[gamma] = {
            "V": V,
            "Q": Q,
            "policy": policy,
            "history": history,
            "converged": converged,
            "cycles": cycles,
        }

    df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print(df.to_string(index=False))

    df.to_csv("gamma_sweep_value_iteration_summary.csv", index=False)
    np.save("gamma_sweep_value_iteration_results.npy", all_results, allow_pickle=True)

    print("\nSaved:")
    print("  gamma_sweep_value_iteration_summary.csv")
    print("  gamma_sweep_value_iteration_results.npy")