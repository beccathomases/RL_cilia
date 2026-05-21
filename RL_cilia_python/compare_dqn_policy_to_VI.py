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


def policy_evaluation_deterministic(env, policy, gamma=0.99, tol=1e-12, max_iter=100000, verbose=True):
    """
    Exact iterative policy evaluation for a deterministic policy on a finite deterministic MDP:
        V_pi(s) = r(s,pi(s)) + gamma * V_pi(s')
    """
    n0, n1 = int(env.n_bins[0]), int(env.n_bins[1])
    V = np.zeros((n0, n1), dtype=float)

    for it in range(1, max_iter + 1):
        Vnew = np.zeros_like(V)

        for i in range(n0):
            for j in range(n1):
                a = int(policy[i, j])
                s2 = deterministic_next_state(env, (i, j), a)
                r = immediate_reward(env, (i, j), a)
                Vnew[i, j] = r + gamma * V[s2]

        err = np.max(np.abs(Vnew - V))
        V = Vnew

        if verbose and (it == 1 or it % 100 == 0):
            print(f"Policy eval iter {it:5d} | ||Vnew-V||_inf = {err:.3e}")

        if err < tol:
            if verbose:
                print(f"Policy evaluation converged at iter {it} with err {err:.3e}")
            break
    else:
        print("Warning: policy evaluation hit max_iter")

    return V


def greedy_policy_from_Vstar(vi_data):
    """
    If the VI file already saved policy, use it.
    """
    if "policy" in vi_data:
        return vi_data["policy"]
    return None


# ============================================================
# SETTINGS
# ============================================================
gamma = 0.99

vi_file = "value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy"
dqn_results_file = "dqn_cilia_2_ball_results.npy"
dqn_weights_file = "dqn_cilia_2_ball.pt"


# ============================================================
# LOAD VI DATA
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

V_star = vi_data["V"]

print("Loaded VI file:", vi_file)
print("V* shape:", V_star.shape)


# ============================================================
# LOAD DQN
# ============================================================
dqn_data = np.load(dqn_results_file, allow_pickle=True).item()

dqn_net = QNet(input_dim=2, hidden=64, n_actions=env.action_space.n)
dqn_net.load_state_dict(torch.load(dqn_weights_file, map_location="cpu"))
dqn_net.eval()

dqn_policy = extract_dqn_policy(dqn_net, env, device="cpu")
print("Extracted greedy DQN policy.")


# ============================================================
# EVALUATE DQN POLICY EXACTLY
# ============================================================
V_dqn = policy_evaluation_deterministic(env, dqn_policy, gamma=gamma, tol=1e-12, verbose=True)


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
im = plt.imshow(V_star, origin="lower", aspect="auto")
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
Vdiff = V_dqn - V_star

plt.figure(figsize=(7, 5))
im = plt.imshow(Vdiff, origin="lower", aspect="auto", cmap="coolwarm")
plt.colorbar(im, label=r"$V_{\pi_{\mathrm{DQN}}}(s) - V^*(s)$")
plt.xlabel("phi2 index")
plt.ylabel("phi1 index")
plt.title(r"Difference Heatmap: $V_{\pi_{\mathrm{DQN}}} - V^*$")
plt.tight_layout()
plt.savefig("VpiDQN_minus_Vstar_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()


# ============================================================
# COMPARE TO VI OPTIMAL VALUE
# ============================================================
diff = V_dqn - V_star

print("\n================ VALUE COMPARISON ================")
print(f"max(V_dqn - V*)   = {np.max(diff):.12f}")
print(f"min(V_dqn - V*)   = {np.min(diff):.12f}")
print(f"mean(V_dqn - V*)  = {np.mean(diff):.12f}")
print(f"max abs diff      = {np.max(np.abs(diff)):.12f}")

imax = np.unravel_index(np.argmax(diff), diff.shape)
imin = np.unravel_index(np.argmin(diff), diff.shape)

print("\nState with largest positive difference:")
print("  state =", imax)
print("  V_dqn =", V_dqn[imax])
print("  V*    =", V_star[imax])
print("  diff  =", diff[imax])

print("\nState with largest negative difference:")
print("  state =", imin)
print("  V_dqn =", V_dqn[imin])
print("  V*    =", V_star[imin])
print("  diff  =", diff[imin])

# Check whether DQN ever beats V* beyond numerical tolerance
tol_check = 1e-8
num_violations = np.sum(diff > tol_check)
print(f"\nNumber of states with V_dqn > V* + {tol_check}: {num_violations}")

if num_violations == 0:
    print("Good: no violation of optimality detected (up to tolerance).")
else:
    print("There are apparent violations. This suggests stale files, evaluation mismatch, or a bug.")


# ============================================================
# OPTIONAL: compare actions where DQN differs from VI greedy policy
# ============================================================
vi_policy = greedy_policy_from_Vstar(vi_data)
if vi_policy is not None:
    action_diff = (dqn_policy != vi_policy)
    print(f"\nNumber of states where DQN greedy action differs from VI greedy action: {np.sum(action_diff)}")
else:
    print("\nVI policy not found in saved file, so action-by-action comparison skipped.")


    