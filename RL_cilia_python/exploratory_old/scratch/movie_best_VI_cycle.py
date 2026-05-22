import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# USER SETTINGS
# ============================================================

vi_file = "value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy"
cycle_id = 0                  # 0 = best cycle
save_gif = True
gif_name = "vi_cycle_overlay_with_rewards.gif"
interval_ms = 500
show_tip_trace = True

# Build env consistent with saved VI file
data = np.load(vi_file, allow_pickle=True).item()
print("Loading:", vi_file)
print("Boundary mode in file:", data.get("boundary_mode"))
print("n_bins in file:", data.get("n_bins"))

env = Cilia2BallEnv(
    max_steps=500,
    precompute=False,
    boundary_mode=data["boundary_mode"],
    invalid_penalty=data["invalid_penalty"],
    reward_rescale=data["reward_rescale"],
    n_bins=data["n_bins"].tolist(),
    angle_mins=data["angle_mins"].tolist(),
    angle_maxs=data["angle_maxs"].tolist(),
)

cycles = data["cycles"]
cyc = cycles[cycle_id]
print("Cycle keys:", cyc.keys())

cycle_states = cyc["cycle"]
cycle_actions = cyc.get("actions", None)
cycle_rewards = np.array(cyc.get("rewards", []), dtype=float)
avg_reward = cyc.get("avg_reward", np.nan)

print(f"Loaded cycle {cycle_id+1}")
print(f"  length = {len(cycle_states)}")
print(f"  avg reward = {avg_reward}")


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

# ============================================================
# PRECOMPUTE SHAPES / AXES
# ============================================================

shape_list = []
tip_x = []
tip_y = []

for s in cycle_states:
    x, y = state_to_shape_xy(env, s)
    shape_list.append((x, y))
    tip_x.append(x[-1])
    tip_y.append(y[-1])

tip_x = np.array(tip_x)
tip_y = np.array(tip_y)

allx = np.concatenate([xy[0] for xy in shape_list])
ally = np.concatenate([xy[1] for xy in shape_list])

pad_x = 0.1 * max(1e-6, allx.max() - allx.min())
pad_y = 0.1 * max(1e-6, ally.max() - ally.min())

xmin, xmax = allx.min() - pad_x, allx.max() + pad_x
ymin, ymax = ally.min() - pad_y, ally.max() + pad_y

nframes = len(shape_list)
cmap = plt.get_cmap("viridis")

if len(cycle_rewards) == 0:
    cycle_rewards = np.zeros(nframes)

rpad = 0.1 * max(1e-6, cycle_rewards.max() - cycle_rewards.min()) if len(cycle_rewards) > 1 else 0.1
rmin = cycle_rewards.min() - rpad
rmax = cycle_rewards.max() + rpad


# ============================================================
# FIGURE WITH TWO PANELS
# ============================================================

fig, (ax_shape, ax_reward) = plt.subplots(1, 2, figsize=(11, 5))


def animate(k):
    ax_shape.clear()
    ax_reward.clear()

    # ------------------------
    # LEFT: overlayed shapes
    # ------------------------
    for j in range(k + 1):
        color = cmap(j / max(1, nframes - 1))
        x, y = shape_list[j]
        ax_shape.plot(x, y, "-", color=color, linewidth=2, alpha=0.8)
        ax_shape.plot(x, y, "o", color=color, markersize=4, alpha=0.8)

    # highlight current shape
    xk, yk = shape_list[k]
    ax_shape.plot(xk, yk, "-k", linewidth=3)
    ax_shape.plot(xk, yk, "ok", markersize=6)

    # mark start shape
    x0, y0 = shape_list[0]
    ax_shape.plot(x0, y0, "-", color="red", linewidth=2, alpha=0.7)
    ax_shape.plot(x0, y0, "o", color="red", markersize=5, alpha=0.7)

    if show_tip_trace:
        ax_shape.plot(tip_x[:k+1], tip_y[:k+1], "--", color="gray", linewidth=1.5, alpha=0.8)

    ax_shape.set_xlim(xmin, xmax)
    ax_shape.set_ylim(ymin, ymax)
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.grid(True)
    ax_shape.set_xlabel("x")
    ax_shape.set_ylabel("z")
    ax_shape.set_title(
        f"Cycle overlay\nframe {k+1}/{nframes}"
    )

    # ------------------------
    # RIGHT: reward trace
    # ------------------------
    steps = np.arange(1, len(cycle_rewards) + 1)
    colors = [cmap(j / max(1, nframes - 1)) for j in range(len(cycle_rewards))]

    ax_reward.plot(steps, cycle_rewards, "-", color="0.7", linewidth=1.5)
    ax_reward.scatter(steps, cycle_rewards, c=colors, s=35, zorder=3)
    ax_reward.plot(steps[k], cycle_rewards[k], "ko", markersize=9, zorder=4)

    ax_reward.set_xlim(1, len(cycle_rewards))
    ax_reward.set_ylim(rmin, rmax)
    ax_reward.grid(True)
    ax_reward.set_xlabel("Step in cycle")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title(f"Reward trace\navg reward = {avg_reward:.6f}")

    fig.suptitle("Best VI cycle", fontsize=14)
    fig.tight_layout()


anim = FuncAnimation(fig, animate, frames=nframes, interval=interval_ms, repeat=True)

if save_gif:
    anim.save(gif_name, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))
    print(f"Saved {gif_name}")

plt.show()