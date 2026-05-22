import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# USER SETTINGS
# ============================================================

results_file = "tabular_q_cilia_2_ball_stay_penalty_bins11x21_ep1000_steps500_g0.990_eps0.75_a0.99.npy"

seed_to_plot = 1          # <- change this
cycle_id = 0              # 0 = best cycle for that seed
make_animation = True     # set False if you only want static plots

# ============================================================
# LOAD RESULTS
# ============================================================

data = np.load(results_file, allow_pickle=True).item()
results = list(data["results"])

print("Loaded:", results_file)
print("Available seeds:", [r["seed"] for r in results])

# find requested seed
match = None
for r in results:
    if int(r["seed"]) == seed_to_plot:
        match = r
        break

if match is None:
    raise ValueError(f"Could not find seed {seed_to_plot} in file.")

cycles = list(match["cycles"])
if len(cycles) == 0:
    raise ValueError(f"Seed {seed_to_plot} has no saved cycles.")

cyc = cycles[cycle_id]

cycle_states = cyc["cycle"]
cycle_actions = cyc["actions"]
cycle_rewards = np.array(cyc["rewards"], dtype=float)
avg_reward = float(cyc["avg_reward"])
cycle_len = int(cyc["length"])

print(f"Seed {seed_to_plot}")
print(f"  cycle_id    = {cycle_id}")
print(f"  cycle_len   = {cycle_len}")
print(f"  avg_reward  = {avg_reward:.6f}")

# ============================================================
# REBUILD ENV
# ============================================================

env = Cilia2BallEnv(
    max_steps=int(data["max_steps"]),
    precompute=False,
    boundary_mode=data["boundary_mode"],
    invalid_penalty=-0.1,          # adjust if you later save this at top level
    reward_rescale=100.0,          # adjust if you later save this at top level
    n_bins=data["n_bins"].tolist(),
    angle_mins=data["angle_mins"].tolist(),
    angle_maxs=data["angle_maxs"].tolist(),
)

# ============================================================
# HELPERS
# ============================================================

def state_to_shape_xy(env, state):
    """
    Convert one discrete state into x,z coordinates for plotting.
    Uses cumulative segment angles psi = cumsum(phi).
    """
    s = np.array(state, dtype=float)
    phi = env.state_to_angles(s)
    psi = np.cumsum(phi)

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


def state_to_angles(env, state):
    s = np.array(state, dtype=float)
    return env.state_to_angles(s)


# ============================================================
# PRECOMPUTE PHASE / SHAPE DATA
# ============================================================

phis = np.array([state_to_angles(env, s) for s in cycle_states])
phi1 = phis[:, 0]
phi2 = phis[:, 1]

shape_list = [state_to_shape_xy(env, s) for s in cycle_states]

# ============================================================
# PLOT 1: PHASE PLANE
# ============================================================

plt.figure(figsize=(6, 6))
plt.plot(phi1, phi2, "o-", linewidth=2)
plt.plot(phi1[0], phi2[0], "ro", markersize=9, label="start")
plt.plot(phi1[-1], phi2[-1], "ks", markersize=8, label="end")
plt.xlabel(r"$\phi_1$")
plt.ylabel(r"$\phi_2$")
plt.title(f"Seed {seed_to_plot}: phase plane\nlen={cycle_len}, avg reward={avg_reward:.6f}")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# PLOT 2: STROKES OVERLAID
# ============================================================

plt.figure(figsize=(6, 6))
cmap = plt.get_cmap("viridis")

for j, (x, z) in enumerate(shape_list):
    color = cmap(j / max(1, cycle_len - 1))
    plt.plot(x, z, "o-", color=color, linewidth=2, markersize=4, alpha=0.9)

# highlight first and last
x0, z0 = shape_list[0]
plt.plot(x0, z0, "o-", color="red", linewidth=2.5, markersize=5, label="start shape")

xL, zL = shape_list[-1]
plt.plot(xL, zL, "o-", color="black", linewidth=2.5, markersize=5, label="end shape")

plt.xlabel("x")
plt.ylabel("z")
plt.title(f"Seed {seed_to_plot}: stroke overlay")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# PLOT 3: REWARD TRACE
# ============================================================

plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, cycle_len + 1), cycle_rewards, "o-", linewidth=2)
plt.xlabel("Step in cycle")
plt.ylabel("Reward")
plt.title(f"Seed {seed_to_plot}: reward trace")
plt.grid(True)
plt.tight_layout()

# ============================================================
# OPTIONAL ANIMATION
# ============================================================

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
            f"Seed {seed_to_plot}: frame {frame+1}/{cycle_len}\n"
            f"reward={cycle_rewards[frame]:.6f}"
        )
        return (line,)

    ani = FuncAnimation(
        fig,
        update,
        frames=cycle_len,
        init_func=init,
        interval=500,
        blit=True,
        repeat=True,
    )

plt.show()