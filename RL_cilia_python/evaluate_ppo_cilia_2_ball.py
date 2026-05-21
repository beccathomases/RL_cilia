import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# USER SETTINGS
# ============================================================

model_path = "ppo_model"

# Use the same env settings you trained with
env_kwargs = dict(
    max_steps=500,
    precompute=True,
)

n_rollouts = 5
rollout_max_steps = 1000
deterministic = True

make_animation = True
animate_rollout_index = 0   # which rollout to animate if a cycle is found

# Start modes:
#   "reset"         -> use env.reset()
#   "center"        -> force center state
#   "corners"       -> test a few fixed starts
#   "random_fixed"  -> choose random states by hand
start_mode = "reset"


# ============================================================
# HELPERS
# ============================================================

def state_to_shape_xy(env, state):
    """
    Convert one discrete state into x,z coordinates for plotting.
    Uses cumulative segment angles from the environment.
    """
    psi = env.state_to_segment_angles(np.array(state, dtype=float))

    p0 = np.array(env.X0, dtype=float)
    p1 = p0 + env.len * np.array(
        [np.sin(psi[0]), 0.0, np.cos(psi[0])], dtype=float
    )
    p2 = p1 + env.len * np.array(
        [np.sin(psi[1]), 0.0, np.cos(psi[1])], dtype=float
    )

    x = np.array([p0[0], p1[0], p2[0]])
    z = np.array([p0[2], p1[2], p2[2]])
    return x, z


def run_policy_rollout(model, env, start_state=None, max_steps=1000, deterministic=True):
    """
    Run one rollout and record states/actions/rewards.
    Detects first repeated state and returns the corresponding cycle if found.
    """
    obs, _ = env.reset()

    if start_state is not None:
        env.state = np.array(start_state, dtype=int)
        obs = env.state.copy()

    states = [tuple(map(int, obs))]
    actions = []
    rewards = []

    visited = {tuple(map(int, obs)): 0}
    cycle_info = None

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        state = tuple(map(int, obs))

        actions.append(action)
        rewards.append(float(reward))
        states.append(state)

        if cycle_info is None and state in visited:
            first_idx = visited[state]
            # states[first_idx] repeats at current state
            cycle_states = states[first_idx:-1]
            cycle_actions = actions[first_idx:]
            cycle_rewards = rewards[first_idx:]

            cycle_info = dict(
                start_index=first_idx,
                cycle_states=cycle_states,
                cycle_actions=cycle_actions,
                cycle_rewards=cycle_rewards,
                cycle_length=len(cycle_actions),
                avg_cycle_reward=float(np.mean(cycle_rewards)) if len(cycle_rewards) > 0 else np.nan,
            )

        if state not in visited:
            visited[state] = len(states) - 1

        if terminated or truncated:
            break

    return dict(
        states=states,
        actions=actions,
        rewards=np.array(rewards, dtype=float),
        cycle_info=cycle_info,
    )


def canonical_cycle(cycle_states):
    """
    Canonical representation invariant under cyclic shifts.
    """
    cyc = [tuple(map(int, s)) for s in cycle_states]
    n = len(cyc)
    if n == 0:
        return tuple()
    rots = [tuple(cyc[k:] + cyc[:k]) for k in range(n)]
    return min(rots)


def get_start_states(env, mode, n_rollouts):
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])

    if mode == "reset":
        return [None] * n_rollouts

    if mode == "center":
        s = (n0 // 2, n1 // 2)
        return [s] * n_rollouts

    if mode == "corners":
        starts = [
            (0, 0),
            (0, n1 - 1),
            (n0 - 1, 0),
            (n0 - 1, n1 - 1),
            (n0 // 2, n1 // 2),
        ]
        return starts[:n_rollouts]

    if mode == "random_fixed":
        rng = np.random.default_rng(0)
        starts = []
        for _ in range(n_rollouts):
            starts.append((int(rng.integers(0, n0)), int(rng.integers(0, n1))))
        return starts

    raise ValueError(f"Unknown start_mode: {mode}")


# ============================================================
# LOAD MODEL AND ENV
# ============================================================

env = Cilia2BallEnv(**env_kwargs)
model = PPO.load(model_path)

print("Loaded model:", model_path)
print("Environment settings:", env_kwargs)

starts = get_start_states(env, start_mode, n_rollouts)

all_rollouts = []
cycle_reps = {}

for k, start_state in enumerate(starts):
    env_k = Cilia2BallEnv(**env_kwargs)
    out = run_policy_rollout(
        model,
        env_k,
        start_state=start_state,
        max_steps=rollout_max_steps,
        deterministic=deterministic,
    )
    all_rollouts.append(out)

    print("\n" + "=" * 60)
    print(f"Rollout {k+1}")
    print("Start state:", start_state if start_state is not None else "env.reset()")
    print("Steps taken:", len(out["actions"]))
    print("Total reward:", float(np.sum(out["rewards"])))

    if out["cycle_info"] is None:
        print("No repeated-state cycle detected.")
    else:
        C = out["cycle_info"]
        print("Cycle detected.")
        print("  cycle length      =", C["cycle_length"])
        print("  avg cycle reward  =", C["avg_cycle_reward"])
        print("  cycle states      =", C["cycle_states"])

        canon = canonical_cycle(C["cycle_states"])
        cycle_reps.setdefault(canon, 0)
        cycle_reps[canon] += 1

print("\n" + "=" * 60)
print("Unique detected cycles across rollouts:", len(cycle_reps))
for idx, (cyc, count) in enumerate(cycle_reps.items(), start=1):
    print(f"Cycle {idx}: seen {count} time(s), length {len(cyc)}")

# ============================================================
# PICK ONE ROLLOUT TO PLOT
# ============================================================

plot_rollout = all_rollouts[animate_rollout_index]
states = plot_rollout["states"]
actions = plot_rollout["actions"]
rewards = plot_rollout["rewards"]
cycle_info = plot_rollout["cycle_info"]

phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in states])
phi1 = phis[:, 0]
phi2 = phis[:, 1]

# ------------------------------------------------------------
# Plot 1: reward trace over whole rollout
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(rewards) + 1), rewards, "o-", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title(f"PPO rollout rewards (rollout {animate_rollout_index+1})")
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------
# Plot 2: phase-plane path over whole rollout
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(phi1, phi2, "o-", linewidth=2)
plt.plot(phi1[0], phi2[0], "ro", markersize=9, label="start")
plt.plot(phi1[-1], phi2[-1], "ks", markersize=8, label="end")
plt.xlabel(r"$\phi_1$")
plt.ylabel(r"$\phi_2$")
plt.title(f"PPO rollout in phase plane (rollout {animate_rollout_index+1})")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ------------------------------------------------------------
# Plot 3: if cycle found, stroke overlay for the cycle
# ------------------------------------------------------------
if cycle_info is not None:
    cycle_states = cycle_info["cycle_states"]
    cycle_rewards = np.array(cycle_info["cycle_rewards"], dtype=float)

    shape_list = [state_to_shape_xy(env, s) for s in cycle_states]
    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(6, 6))
    for j, (x, z) in enumerate(shape_list):
        color = cmap(j / max(1, len(shape_list) - 1))
        plt.plot(x, z, "o-", color=color, linewidth=2, markersize=4, alpha=0.9)

    x0, z0 = shape_list[0]
    plt.plot(x0, z0, "o-", color="red", linewidth=2.5, markersize=5, label="start shape")

    xL, zL = shape_list[-1]
    plt.plot(xL, zL, "o-", color="black", linewidth=2.5, markersize=5, label="end shape")

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(
        f"Detected PPO cycle overlay\n"
        f"len={cycle_info['cycle_length']}, avg reward={cycle_info['avg_cycle_reward']:.6f}"
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Plot 4: cycle reward trace
    # --------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(cycle_rewards) + 1), cycle_rewards, "o-", linewidth=2)
    plt.xlabel("Step in detected cycle")
    plt.ylabel("Reward")
    plt.title("Detected PPO cycle rewards")
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------------
    # Optional animation of detected cycle
    # --------------------------------------------------------
    if make_animation:
        allx = np.concatenate([xy[0] for xy in shape_list])
        allz = np.concatenate([xy[1] for xy in shape_list])

        padx = 0.1 * max(1e-6, allx.max() - allx.min())
        padz = 0.1 * max(1e-6, allz.max() - allz.min())

        fig, ax = plt.subplots(figsize=(6, 5))
        line, = ax.plot([], [], "o-", lw=3)

        ax.set_xlim(allx.min() - padx, allx.max() + padx)
        ax.set_ylim(allz.min() - padz, allz.max() + padz)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True)

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            x, z = shape_list[frame]
            line.set_data(x, z)
            ax.set_title(
                f"PPO detected cycle: frame {frame+1}/{len(shape_list)}\n"
                f"reward={cycle_rewards[frame]:.6f}"
            )
            return (line,)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(shape_list),
            init_func=init,
            interval=500,
            blit=True,
            repeat=True,
        )

plt.show()