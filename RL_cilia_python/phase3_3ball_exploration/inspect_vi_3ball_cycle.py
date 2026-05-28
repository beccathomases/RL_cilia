import os
import numpy as np
import matplotlib.pyplot as plt

from cilia_3_ball_env import Cilia3BallEnv


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
RESULT_FILE = "value_iteration_cilia_3_ball_clip_penalty_bins11x21x21_g0.990.npy"
OUTDIR = "figures_vi_3ball"
TMAX = 30   # repeated time horizon for angle traces

os.makedirs(OUTDIR, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_npy_dict(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def repeat_to_length(arr, length_needed):
    arr = np.asarray(arr)
    reps = int(np.ceil(length_needed / len(arr)))
    out = np.tile(arr, reps)
    return out[:length_needed]


def state_to_shape_xy(env, state):
    psi = env.state_to_segment_angles(np.array(state, dtype=float))

    p0 = np.array(env.X0, dtype=float)
    pts = [p0.copy()]
    curr = p0.copy()

    for k in range(env.Nballs):
        curr = curr + env.len * np.array(
            [np.sin(psi[k]), 0.0, np.cos(psi[k])], dtype=float
        )
        pts.append(curr.copy())

    pts = np.array(pts)
    x = pts[:, 0]
    z = pts[:, 2]
    return x, z


# ------------------------------------------------------------
# Load result and env
# ------------------------------------------------------------
data = load_npy_dict(RESULT_FILE)
env_settings = data["env_settings"]

print("RESULT_FILE:", RESULT_FILE)
print("saved env_settings:", env_settings)

env = Cilia3BallEnv(
    max_steps=env_settings["max_steps"],
    precompute=True,
    boundary_mode=env_settings["boundary_mode"],
    invalid_penalty=env_settings["invalid_penalty"],
    reward_rescale=env_settings["reward_rescale"],
    n_bins=env_settings["n_bins"],
    angle_mins=env_settings["angle_mins"],
    angle_maxs=env_settings["angle_maxs"],
)

cycles = list(data["cycles"])
if len(cycles) == 0:
    raise ValueError("No cycles found in saved VI result.")

best = cycles[0]
cycle_states = list(best["cycle"])
cycle_actions = list(best["actions"])
cycle_rewards = list(best["rewards"])
cycle_len = int(best["length"])
avg_reward = float(best["avg_reward"])

print("\n===================================")
print("Best VI cycle summary")
print(f"Length       : {cycle_len}")
print(f"Avg reward   : {avg_reward:.6f}")
print(f"States       : {cycle_states}")
print(f"Actions      : {cycle_actions}")
print(f"Rewards      : {cycle_rewards}")
print("===================================\n")

# print transition details
print("Step-by-step transitions:")
for m, state in enumerate(cycle_states):
    a = cycle_actions[m]
    r = cycle_rewards[m]
    trans = env.transition_info(np.array(state, dtype=int), a)
    print(
        f"step {m}: state={state}, action={a}, reward={r:.6f}, "
        f"next={tuple(trans['next_state'].tolist())}, clipped={trans['was_clipped']}"
    )

# ------------------------------------------------------------
# Angles
# ------------------------------------------------------------
phis = np.array([env.state_to_angles(np.array(s, dtype=float)) for s in cycle_states])
phi1 = phis[:, 0]
phi2 = phis[:, 1]
phi3 = phis[:, 2]

t = np.arange(TMAX + 1)
phi1_rep = repeat_to_length(phi1, TMAX + 1)
phi2_rep = repeat_to_length(phi2, TMAX + 1)
phi3_rep = repeat_to_length(phi3, TMAX + 1)

plt.figure(figsize=(8, 4.8))
plt.plot(t, phi1_rep, "o-", linewidth=2, markersize=4, label=r"$\phi_1$")
plt.plot(t, phi2_rep, "s--", linewidth=2, markersize=4, label=r"$\phi_2$")
plt.plot(t, phi3_rep, "d-.", linewidth=2, markersize=4, label=r"$\phi_3$")
plt.xlabel("t")
plt.ylabel("angle")
plt.title(f"3-ball VI cycle: repeated angle traces to t={TMAX}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "vi_3ball_cycle_angle_traces.png"), dpi=200, bbox_inches="tight")

# ------------------------------------------------------------
# Physical stroke overlay
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))
cmap = plt.get_cmap("viridis")

for j, state in enumerate(cycle_states):
    x, z = state_to_shape_xy(env, state)
    color = cmap(j / max(1, cycle_len - 1))
    plt.plot(x, z, "o-", linewidth=2, markersize=5, color=color, alpha=0.9)

# mark first and last
x0, z0 = state_to_shape_xy(env, cycle_states[0])
plt.plot(x0, z0, "o-", linewidth=2.8, markersize=6, color="red")

xL, zL = state_to_shape_xy(env, cycle_states[-1])
plt.plot(xL, zL, "o-", linewidth=2.8, markersize=6, color="black")

plt.xlabel("x")
plt.ylabel("z")
plt.title("3-ball VI cycle: physical stroke overlay")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "vi_3ball_cycle_stroke_overlay.png"), dpi=200, bbox_inches="tight")

# ------------------------------------------------------------
# Pairwise phase-plane projections
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=False, sharey=False)

axes[0].plot(phi1, phi2, "o-", linewidth=2, markersize=5)
axes[0].plot(phi1[0], phi2[0], "o", markersize=8)
axes[0].set_xlabel(r"$\phi_1$")
axes[0].set_ylabel(r"$\phi_2$")
axes[0].set_title(r"$(\phi_1,\phi_2)$")
axes[0].grid(True)

axes[1].plot(phi1, phi3, "o-", linewidth=2, markersize=5)
axes[1].plot(phi1[0], phi3[0], "o", markersize=8)
axes[1].set_xlabel(r"$\phi_1$")
axes[1].set_ylabel(r"$\phi_3$")
axes[1].set_title(r"$(\phi_1,\phi_3)$")
axes[1].grid(True)

axes[2].plot(phi2, phi3, "o-", linewidth=2, markersize=5)
axes[2].plot(phi2[0], phi3[0], "o", markersize=8)
axes[2].set_xlabel(r"$\phi_2$")
axes[2].set_ylabel(r"$\phi_3$")
axes[2].set_title(r"$(\phi_2,\phi_3)$")
axes[2].grid(True)

fig.suptitle("3-ball VI cycle: pairwise phase projections", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, "vi_3ball_cycle_phase_projections.png"), dpi=200, bbox_inches="tight")

print("\nSaved figures:")
print(" ", os.path.join(OUTDIR, "vi_3ball_cycle_angle_traces.png"))
print(" ", os.path.join(OUTDIR, "vi_3ball_cycle_stroke_overlay.png"))
print(" ", os.path.join(OUTDIR, "vi_3ball_cycle_phase_projections.png"))

plt.show()