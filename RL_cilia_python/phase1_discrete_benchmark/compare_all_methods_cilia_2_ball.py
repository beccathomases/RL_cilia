import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from cilia_2_ball_env import Cilia2BallEnv


# ============================================================
# USER SETTINGS
# ============================================================

vi_file = "value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy"
tabq_file = "tabular_q_cilia_2_ball_clip_penalty_bins11x21_ep1000_steps500_g0.990_eps0.75_a0.99.npy"
dqn_results_file = "dqn_cilia_2_ball_results.npy"
dqn_weights_file = "dqn_cilia_2_ball.pt"
ppo_model_file = "ppo_model.zip"

tabq_seed = 1
ppo_rollout_steps = 1000
ppo_start_mode = "center"   # "center" or "reset"

save_figures = True
fig_dir = "figures"

phase_fig_name   = "all_methods_phase_plane_tiled.png"
reward_fig_name  = "all_methods_reward_traces.png"
stroke_fig_name  = "all_methods_stroke_overlays_tiled.png"
summary_csv_name = "all_methods_cycle_summary.csv"


# ============================================================
# DQN NETWORK
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
# HELPERS
# ============================================================
def ensure_fig_dir(path):
    if save_figures:
        os.makedirs(path, exist_ok=True)


def load_npy_dict(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def state_scale(env):
    return np.array(env.n_bins, dtype=np.float32) - 1.0


def canonical_cycle(cycle):
    cyc = [tuple(map(int, s)) for s in cycle]
    if len(cyc) == 0:
        return tuple()
    rots = [tuple(cyc[k:] + cyc[:k]) for k in range(len(cyc))]
    return min(rots)


def cycles_match(cycle_a, cycle_b):
    return canonical_cycle(cycle_a) == canonical_cycle(cycle_b)


def deterministic_next_state(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return tuple(trans["next_state"].tolist())


def immediate_reward(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)


def state_to_shape_xy(env, state):
    psi = env.state_to_segment_angles(np.array(state, dtype=float))

    p0 = np.array(env.X0, dtype=float)
    p1 = p0 + env.len * np.array([np.sin(psi[0]), 0.0, np.cos(psi[0])], dtype=float)
    p2 = p1 + env.len * np.array([np.sin(psi[1]), 0.0, np.cos(psi[1])], dtype=float)

    x = np.array([p0[0], p1[0], p2[0]])
    z = np.array([p0[2], p1[2], p2[2]])
    return x, z


def cycle_to_phase(env, cycle_states):
    phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])
    return phis[:, 0], phis[:, 1]


def extract_dqn_policy(q_net, env, device="cpu"):
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


def find_cycle_from_policy(env, policy, start_state):
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

    cyc_start = visited[state]
    cycle_states = trajectory[cyc_start:]

    cycle_actions = []
    cycle_rewards = []
    for s in cycle_states:
        i, j = s
        a = int(policy[i, j])
        r = immediate_reward(env, s, a)
        cycle_actions.append(a)
        cycle_rewards.append(r)

    return {
        "cycle": cycle_states,
        "actions": cycle_actions,
        "rewards": cycle_rewards,
        "avg_reward": float(np.mean(cycle_rewards)),
        "length": len(cycle_states),
    }


def run_ppo_and_detect_cycle(model, env, start_state=None, max_steps=1000, deterministic=True):
    obs, _ = env.reset()

    if start_state is not None:
        env.state = np.array(start_state, dtype=int)
        obs = env.state.copy()

    visited = {tuple(map(int, obs)): 0}
    states = [tuple(map(int, obs))]
    actions = []
    rewards = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        s = tuple(map(int, obs))

        actions.append(action)
        rewards.append(float(reward))
        states.append(s)

        if s in visited:
            first = visited[s]
            cycle_states = states[first:-1]
            cycle_actions = actions[first:]
            cycle_rewards = rewards[first:]
            return {
                "cycle": cycle_states,
                "actions": cycle_actions,
                "rewards": cycle_rewards,
                "avg_reward": float(np.mean(cycle_rewards)),
                "length": len(cycle_states),
            }

        visited[s] = len(states) - 1

        if terminated or truncated:
            break

    return None


def print_summary(name, cyc):
    print(f"\n{name}")
    print(f"  length      = {cyc['length']}")
    print(f"  avg reward  = {cyc['avg_reward']:.6f}")
    print(f"  cycle       = {cyc['cycle']}")


def save_summary_csv(methods, outpath, vi_cycle):
    rows = []
    for name, cyc, _ in methods:
        rows.append({
            "method": name,
            "cycle_length": cyc["length"],
            "avg_reward": cyc["avg_reward"],
            "matches_vi_up_to_shift": cycles_match(cyc["cycle"], vi_cycle["cycle"]),
        })

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "cycle_length", "avg_reward", "matches_vi_up_to_shift"]
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# LOAD VI
# ============================================================
vi_data = load_npy_dict(vi_file)
vi_env = Cilia2BallEnv(
    max_steps=500,
    precompute=True,
    boundary_mode=vi_data["boundary_mode"],
    invalid_penalty=vi_data["invalid_penalty"],
    reward_rescale=vi_data["reward_rescale"],
    n_bins=vi_data["n_bins"].tolist(),
    angle_mins=vi_data["angle_mins"].tolist(),
    angle_maxs=vi_data["angle_maxs"].tolist(),
)
vi_cycle = vi_data["cycles"][0]


# ============================================================
# LOAD TABULAR Q
# ============================================================
tabq_data = load_npy_dict(tabq_file)
tabq_results = list(tabq_data["results"])

tabq_match = None
for r in tabq_results:
    if int(r["seed"]) == tabq_seed:
        tabq_match = r
        break
if tabq_match is None:
    raise ValueError(f"Seed {tabq_seed} not found in tabular-Q file.")

tabq_cycle = list(tabq_match["cycles"])[0]

# Safe reconstruction: these keys are not stored in the tabq file
tabq_env = Cilia2BallEnv(
    max_steps=int(tabq_data["max_steps"]),
    precompute=True,
    boundary_mode=tabq_data["boundary_mode"],
    invalid_penalty=-0.1,
    reward_rescale=100.0,
    n_bins=tabq_data["n_bins"].tolist(),
    angle_mins=tabq_data["angle_mins"].tolist(),
    angle_maxs=tabq_data["angle_maxs"].tolist(),
)


# ============================================================
# LOAD DQN
# ============================================================
dqn_data = load_npy_dict(dqn_results_file)
dqn_env_settings = dqn_data["env_settings"]

dqn_env = Cilia2BallEnv(**dqn_env_settings)

dqn_net = QNet(input_dim=2, hidden=64, n_actions=dqn_env.action_space.n)
dqn_net.load_state_dict(torch.load(dqn_weights_file, map_location="cpu"))
dqn_net.eval()

dqn_policy = extract_dqn_policy(dqn_net, dqn_env, device="cpu")
dqn_cycle = find_cycle_from_policy(
    dqn_env,
    dqn_policy,
    start_state=(int(dqn_env.n_bins[0] // 2), int(dqn_env.n_bins[1] // 2)),
)


# ============================================================
# LOAD PPO
# ============================================================
ppo_model = PPO.load(ppo_model_file)

# Use same settings as VI for consistency
ppo_env = Cilia2BallEnv(
    max_steps=500,
    precompute=True,
    boundary_mode=vi_data["boundary_mode"],
    invalid_penalty=vi_data["invalid_penalty"],
    reward_rescale=vi_data["reward_rescale"],
    n_bins=vi_data["n_bins"].tolist(),
    angle_mins=vi_data["angle_mins"].tolist(),
    angle_maxs=vi_data["angle_maxs"].tolist(),
)

if ppo_start_mode == "center":
    ppo_start = (int(ppo_env.n_bins[0] // 2), int(ppo_env.n_bins[1] // 2))
else:
    ppo_start = None

ppo_cycle = run_ppo_and_detect_cycle(
    ppo_model,
    ppo_env,
    start_state=ppo_start,
    max_steps=ppo_rollout_steps,
    deterministic=True,
)

if ppo_cycle is None:
    raise ValueError("PPO rollout did not detect a repeated-state cycle.")


# ============================================================
# PRINT SUMMARIES
# ============================================================
print_summary("VI", vi_cycle)
print_summary(f"Tabular Q (seed {tabq_seed})", tabq_cycle)
print_summary("DQN", dqn_cycle)
print_summary("PPO", ppo_cycle)

print("\nCycle matches up to cyclic shift:")
print("  TabQ vs VI :", cycles_match(tabq_cycle["cycle"], vi_cycle["cycle"]))
print("  DQN  vs VI :", cycles_match(dqn_cycle["cycle"], vi_cycle["cycle"]))
print("  PPO  vs VI :", cycles_match(ppo_cycle["cycle"], vi_cycle["cycle"]))


# ============================================================
# PACK METHODS
# ============================================================
methods = [
    ("VI", vi_cycle, vi_env),
    (f"TabQ seed {tabq_seed}", tabq_cycle, tabq_env),
    ("DQN", dqn_cycle, dqn_env),
    ("PPO", ppo_cycle, ppo_env),
]

if save_figures:
    ensure_fig_dir(fig_dir)
    save_summary_csv(methods, os.path.join(fig_dir, summary_csv_name), vi_cycle)


# ============================================================
# FIGURE 1: TILED PHASE PLANE
# ============================================================
fig1, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (name, cyc, env) in zip(axes, methods):
    phi1, phi2 = cycle_to_phase(env, cyc["cycle"])
    ax.plot(phi1, phi2, "o-", linewidth=2, markersize=5)
    ax.plot(phi1[0], phi2[0], "o", markersize=8)
    ax.set_title(f"{name}\nlen={cyc['length']}, avg={cyc['avg_reward']:.6f}")
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.grid(True)

fig1.suptitle("Recurrent cycles in phase plane", fontsize=14)
fig1.tight_layout(rect=[0, 0, 1, 0.97])

if save_figures:
    fig1.savefig(os.path.join(fig_dir, phase_fig_name), dpi=300, bbox_inches="tight")


# ============================================================
# FIGURE 2: REWARD TRACES
# ============================================================
fig2 = plt.figure(figsize=(8, 5))
for name, cyc, _ in methods:
    rewards = np.array(cyc["rewards"], dtype=float)
    plt.plot(np.arange(1, len(rewards) + 1), rewards, "o-", linewidth=2, markersize=4, label=name)

plt.xlabel("Step in cycle")
plt.ylabel("Reward")
plt.title("Cycle reward traces")
plt.grid(True)
plt.legend()
plt.tight_layout()

if save_figures:
    fig2.savefig(os.path.join(fig_dir, reward_fig_name), dpi=300, bbox_inches="tight")


# ============================================================
# FIGURE 3: TILED STROKE OVERLAYS
# ============================================================
fig3, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.ravel()
cmap = plt.get_cmap("viridis")

for ax, (name, cyc, env) in zip(axes, methods):
    shape_list = [state_to_shape_xy(env, s) for s in cyc["cycle"]]
    n = len(shape_list)

    for j, (x, z) in enumerate(shape_list):
        color = cmap(j / max(1, n - 1))
        ax.plot(x, z, "o-", color=color, linewidth=2, markersize=4, alpha=0.9)

    x0, z0 = shape_list[0]
    ax.plot(x0, z0, "o-", color="red", linewidth=2.5, markersize=5)

    xL, zL = shape_list[-1]
    ax.plot(xL, zL, "o-", color="black", linewidth=2.5, markersize=5)

    ax.set_title(f"{name}\nlen={cyc['length']}, avg={cyc['avg_reward']:.6f}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.axis("equal")
    ax.grid(True)

fig3.suptitle("Stroke overlays in physical space", fontsize=14)
fig3.tight_layout(rect=[0, 0, 1, 0.97])

if save_figures:
    fig3.savefig(os.path.join(fig_dir, stroke_fig_name), dpi=300, bbox_inches="tight")


plt.show()

if save_figures:
    print("\nSaved:")
    print(" ", os.path.join(fig_dir, phase_fig_name))
    print(" ", os.path.join(fig_dir, reward_fig_name))
    print(" ", os.path.join(fig_dir, stroke_fig_name))
    print(" ", os.path.join(fig_dir, summary_csv_name))