import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ============================================================
# USER SETTINGS
# ============================================================

vi_file = "value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy"
cycle_id = 0   # 0 = best cycle
save_png = True
png_name = "vi_phase_plane.png"


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(vi_file, allow_pickle=True).item()
cycles = data["cycles"]
cyc = cycles[cycle_id]

cycle_states = cyc["cycle"]
avg_reward = cyc.get("avg_reward", np.nan)

angle_mins = np.array(data["angle_mins"], dtype=float)
angle_maxs = np.array(data["angle_maxs"], dtype=float)
n_bins = np.array(data["n_bins"], dtype=int)

dangle = (angle_maxs - angle_mins) / (n_bins - 1)


# ============================================================
# STATE -> ANGLES
# ============================================================

def state_to_angles(state, angle_mins, dangle):
    s = np.array(state, dtype=float)
    return angle_mins + s * dangle


phis = np.array([state_to_angles(s, angle_mins, dangle) for s in cycle_states])
phi1 = phis[:, 0]
phi2 = phis[:, 1]

n = len(cycle_states)

# close the curve visually
phi1_closed = np.r_[phi1, phi1[0]]
phi2_closed = np.r_[phi2, phi2[0]]

# segments for colored line
points = np.array([phi1_closed, phi2_closed]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))

lc = LineCollection(segments, cmap="viridis", linewidth=3)
lc.set_array(np.arange(len(segments)))
ax.add_collection(lc)

sc = ax.scatter(phi1, phi2, c=np.arange(n), cmap="viridis", s=50, zorder=3)

# mark start and end
ax.plot(phi1[0], phi2[0], "ro", markersize=9, label="start")
ax.plot(phi1[-1], phi2[-1], "ks", markersize=8, label="end")

ax.set_xlabel(r"$\phi_1$")
ax.set_ylabel(r"$\phi_2$")
ax.set_title(f"VI cycle in phase plane\nlength = {n}, avg reward = {avg_reward:.6f}")
ax.grid(True)
ax.legend()

pad1 = 0.08 * max(1e-6, phi1.max() - phi1.min())
pad2 = 0.08 * max(1e-6, phi2.max() - phi2.min())
ax.set_xlim(phi1.min() - pad1, phi1.max() + pad1)
ax.set_ylim(phi2.min() - pad2, phi2.max() + pad2)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Step index")

plt.tight_layout()

if save_png:
    plt.savefig(png_name, dpi=200, bbox_inches="tight")
    print(f"Saved {png_name}")

plt.show()