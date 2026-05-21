import numpy as np
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
# HELPERS
# ============================================================
def state_scale(env):
    return np.array(env.n_bins, dtype=np.float32) - 1.0


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


def deterministic_next_state(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return tuple(trans["next_state"].tolist())


def immediate_reward(env, state, action):
    trans = env.transition_info(np.array(state, dtype=int), action)
    return env.immediate_reward_from_transition(np.array(state, dtype=int), action, trans)


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
        "rewards": np.array(cycle_rewards, dtype=float),
        "avg_reward": float(np.mean(cycle_rewards)),
        "length": len(cycle_states),
    }


def discounted_cycle_values(rewards, gamma):
    """
    Returns discounted infinite-horizon value starting from each phase
    of the periodic cycle.
    """
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


def summarize_cycle(name, cycle_dict, gamma):
    rewards = np.array(cycle_dict["rewards"], dtype=float)
    vals = discounted_cycle_values(rewards, gamma)

    print(f"\n{name}")
    print(f"  length             = {cycle_dict['length']}")
    print(f"  average reward     = {np.mean(rewards):.6f}")
    print(f"  discounted min     = {np.min(vals):.6f}")
    print(f"  discounted max     = {np.max(vals):.6f}")
    print(f"  discounted mean    = {np.mean(vals):.6f}")
    print(f"  best start index   = {np.argmax(vals)}")
    print(f"  worst start index  = {np.argmin(vals)}")

    return vals


# ============================================================
# SETTINGS
# ============================================================
gamma = 0.99

vi_file = "value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy"
dqn_results_file = "dqn_cilia_2_ball_results.npy"
dqn_weights_file = "dqn_cilia_2_ball.pt"


# ============================================================
# LOAD VI
# ============================================================
vi_data = np.load(vi_file, allow_pickle=True).item()

env = Cilia2BallEnv(
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
# LOAD DQN AND EXTRACT ITS GREEDY CYCLE
# ============================================================
dqn_data = np.load(dqn_results_file, allow_pickle=True).item()
dqn_env = Cilia2BallEnv(**dqn_data["env_settings"])

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
# SUMMARIZE
# ============================================================
vi_vals = summarize_cycle("VI cycle", vi_cycle, gamma)
dqn_vals = summarize_cycle("DQN cycle", dqn_cycle, gamma)

print("\nComparison")
print(f"  VI  avg reward   = {vi_cycle['avg_reward']:.6f}")
print(f"  DQN avg reward   = {dqn_cycle['avg_reward']:.6f}")
print(f"  VI  best disc V  = {np.max(vi_vals):.6f}")
print(f"  DQN best disc V  = {np.max(dqn_vals):.6f}")
print(f"  VI  mean disc V  = {np.mean(vi_vals):.6f}")
print(f"  DQN mean disc V  = {np.mean(dqn_vals):.6f}")

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1. Heatmap of V_{pi_DQN}
# -------------------------------------------------
plt.figure(figsize=(7, 5))
im = plt.imshow(V_dqn, origin="lower", aspect="auto")
plt.colorbar(im, label=r"$V_{\pi_{\mathrm{DQN}}}(s)$")
plt.xlabel("phi2 index")
plt.ylabel("phi1 index")
plt.title(r"Heatmap of $V_{\pi_{\mathrm{DQN}}}$")
plt.tight_layout()
plt.savefig("VpiDQN_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# -------------------------------------------------
# 2. Heatmap of V*
# -------------------------------------------------
plt.figure(figsize=(7, 5))
im = plt.imshow(Vstar, origin="lower", aspect="auto")
plt.colorbar(im, label=r"$V^*(s)$")
plt.xlabel("phi2 index")
plt.ylabel("phi1 index")
plt.title(r"Heatmap of $V^*$ from Value Iteration")
plt.tight_layout()
plt.savefig("Vstar_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# -------------------------------------------------
# 3. Heatmap of difference V_{pi_DQN} - V*
# -------------------------------------------------
Vdiff = V_dqn - Vstar

plt.figure(figsize=(7, 5))
im = plt.imshow(Vdiff, origin="lower", aspect="auto", cmap="coolwarm")
plt.colorbar(im, label=r"$V_{\pi_{\mathrm{DQN}}}(s) - V^*(s)$")
plt.xlabel("phi2 index")
plt.ylabel("phi1 index")
plt.title(r"Difference Heatmap: $V_{\pi_{\mathrm{DQN}}} - V^*$")
plt.tight_layout()
plt.savefig("VpiDQN_minus_Vstar_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()