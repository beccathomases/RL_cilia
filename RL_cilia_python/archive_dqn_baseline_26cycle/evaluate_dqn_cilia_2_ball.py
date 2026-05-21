import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn

from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# SAME QNet ARCHITECTURE AS TRAINING
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
# USER SETTINGS
# ============================================================
results_file = "dqn_cilia_2_ball_results.npy"
weights_file = "dqn_cilia_2_ball.pt"

cycle_id = 0          # 0 = best cycle
make_animation = True


# ============================================================
# HELPERS
# ============================================================
def state_scale(env):
    return np.array(env.n_bins, dtype=np.float32) - 1.0


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


def deterministic_next_state(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return tuple(trans["next_state"].tolist())


def immediate_reward(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)


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


# ============================================================
# LOAD SAVED RESULTS
# ============================================================
data = np.load(results_file, allow_pickle=True).item()
diagnostics = data["diagnostics"]
saved_cycles = list(data["cycles"])
env_settings = data["env_settings"]

print("Loaded results from:", results_file)
print("Loaded weights from:", weights_file)

# rebuild env
env = Cilia2BallEnv(**env_settings)

# rebuild qnet and load weights
qnet = QNet(input_dim=2, hidden=64, n_actions=env.action_space.n)
qnet.load_state_dict(torch.load(weights_file, map_location="cpu"))
qnet.eval()

# extract fresh greedy policy
policy = extract_policy(qnet, env, device="cpu")

# choose cycle
if len(saved_cycles) == 0:
    raise ValueError("No saved cycles found in results file.")

cyc = saved_cycles[cycle_id]
cycle_states = cyc["cycle"]
cycle_actions = cyc["actions"]
cycle_rewards = np.array(cyc["rewards"], dtype=float)
avg_reward = float(cyc["avg_reward"])
cycle_len = int(cyc["length"])

print(f"\nSelected cycle {cycle_id}")
print(f"  length      = {cycle_len}")
print(f"  avg reward  = {avg_reward:.6f}")

# ============================================================
# PLOT 1: TRAINING REWARD CURVE
# ============================================================
episode_rewards = diagnostics["episode_rewards"]
episode_lengths = diagnostics["episode_lengths"]
loss_history = diagnostics["loss_history"]
q_means = diagnostics["q_means"]

plt.figure(figsize=(7, 4))
plt.plot(episode_rewards, linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("DQN training rewards")
plt.grid(True)
plt.tight_layout()

# ============================================================
# PLOT 2: TRAINING LENGTH CURVE
# ============================================================
plt.figure(figsize=(7, 4))
plt.plot(episode_lengths, linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Episode length")
plt.title("DQN training episode lengths")
plt.grid(True)
plt.tight_layout()

# ============================================================
# OPTIONAL: loss curve
# ============================================================
if len(loss_history) > 0:
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, linewidth=1.0)
    plt.xlabel("Optimization step")
    plt.ylabel("Loss")
    plt.title("DQN loss history")
    plt.grid(True)
    plt.tight_layout()

# ============================================================
# PRECOMPUTE CYCLE GEOMETRY
# ============================================================
phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])
phi1 = phis[:, 0]
phi2 = phis[:, 1]

shape_list = [state_to_shape_xy(env, s) for s in cycle_states]

# ============================================================
# PLOT 3: PHASE PLANE
# ============================================================
plt.figure(figsize=(6, 6))
plt.plot(phi1, phi2, "o-", linewidth=2)
plt.plot(phi1[0], phi2[0], "ro", markersize=9, label="start")
plt.plot(phi1[-1], phi2[-1], "ks", markersize=8, label="end")
plt.xlabel(r"$\phi_1$")
plt.ylabel(r"$\phi_2$")
plt.title(f"DQN best cycle in phase plane\nlen={cycle_len}, avg reward={avg_reward:.6f}")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# PLOT 4: REWARD TRACE ON CYCLE
# ============================================================
plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, cycle_len + 1), cycle_rewards, "o-", linewidth=2)
plt.xlabel("Step in cycle")
plt.ylabel("Reward")
plt.title("DQN best cycle rewards")
plt.grid(True)
plt.tight_layout()

# ============================================================
# PLOT 5: STROKE OVERLAY
# ============================================================
plt.figure(figsize=(6, 6))
cmap = plt.get_cmap("viridis")

for j, (x, z) in enumerate(shape_list):
    color = cmap(j / max(1, cycle_len - 1))
    plt.plot(x, z, "o-", color=color, linewidth=2, markersize=4, alpha=0.9)

x0, z0 = shape_list[0]
plt.plot(x0, z0, "o-", color="red", linewidth=2.5, markersize=5, label="start shape")

xL, zL = shape_list[-1]
plt.plot(xL, zL, "o-", color="black", linewidth=2.5, markersize=5, label="end shape")

plt.xlabel("x")
plt.ylabel("z")
plt.title("DQN best cycle stroke overlay")
plt.axis("equal")
plt.grid(True)
plt.legend()
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
            f"DQN best cycle: frame {frame+1}/{cycle_len}\n"
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