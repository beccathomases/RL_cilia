import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

summary_csv = "gamma_sweep_value_iteration_summary.csv"
results_npy = "gamma_sweep_value_iteration_results.npy"

gammas_to_plot = [0.90, 0.95, 0.98, 0.99, 0.992, 0.995, 0.997, 0.999]
gamma_tol = 1e-12

figdir = "figures"
os.makedirs(figdir, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_results_object(fname):
    obj = np.load(fname, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def entry_best_cycle(entry):
    if isinstance(entry, dict):
        if "bestCycle" in entry:
            return entry["bestCycle"]
        if "rankedCycles" in entry and len(entry["rankedCycles"]) > 0:
            return entry["rankedCycles"][0]
        if "cycles" in entry and len(entry["cycles"]) > 0:
            return entry["cycles"][0]
    raise KeyError("Could not find best cycle in entry.")


def cycle_states_to_arrays(cycle_states):
    arr = np.array(cycle_states)
    return arr[:, 0], arr[:, 1]


def find_entry_for_gamma(results, gamma):
    if isinstance(results, dict):
        # maybe keyed directly by gamma
        if gamma in results:
            return results[gamma]

        # maybe keys are strings
        for k, v in results.items():
            try:
                g = float(k)
                if abs(g - gamma) < gamma_tol:
                    return v
            except Exception:
                pass

    if isinstance(results, (list, tuple)):
        for entry in results:
            if isinstance(entry, dict) and "gamma" in entry:
                if abs(entry["gamma"] - gamma) < gamma_tol:
                    return entry

    if isinstance(results, np.ndarray):
        for entry in results:
            if isinstance(entry, dict) and "gamma" in entry:
                if abs(entry["gamma"] - gamma) < gamma_tol:
                    return entry

    raise KeyError(f"Could not find gamma={gamma} in results.")


# ============================================================
# Load summary and results
# ============================================================

summary = pd.read_csv(summary_csv)
summary = summary.sort_values("gamma").reset_index(drop=True)

print("\nSUMMARY TABLE")
print(summary.to_string(index=False))

results = load_results_object(results_npy)


# ============================================================
# Figure 1: cycle length vs gamma
# ============================================================

plt.figure(figsize=(7, 5))
plt.plot(summary["gamma"], summary["cycle_length"], "o-", linewidth=2, markersize=6)
plt.xlabel(r"$\gamma$")
plt.ylabel("Cycle length")
plt.title("Optimal cycle length vs discount factor")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "gamma_sweep_cycle_length.png"), dpi=200, bbox_inches="tight")


# ============================================================
# Figure 2: average reward vs gamma
# ============================================================

plt.figure(figsize=(7, 5))
plt.plot(summary["gamma"], summary["avg_reward"], "o-", linewidth=2, markersize=6)
plt.xlabel(r"$\gamma$")
plt.ylabel("Average reward per cycle")
plt.title("Average reward of optimal cycle vs discount factor")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "gamma_sweep_avg_reward.png"), dpi=200, bbox_inches="tight")


# ============================================================
# Figure 3: discounted values vs gamma
# ============================================================

plt.figure(figsize=(7, 5))
plt.plot(summary["gamma"], summary["disc_mean"], "o-", linewidth=2, label="discounted mean")
plt.plot(summary["gamma"], summary["disc_best"], "s--", linewidth=2, label="discounted best")
plt.xlabel(r"$\gamma$")
plt.ylabel("Discounted cycle value")
plt.title("Discounted cycle value vs discount factor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "gamma_sweep_discounted_values.png"), dpi=200, bbox_inches="tight")


# ============================================================
# Figure 4: phase-plane cycles for selected gammas
# ============================================================

plt.figure(figsize=(8, 6))

for gamma in gammas_to_plot:
    try:
        entry = find_entry_for_gamma(results, gamma)
        cyc = entry_best_cycle(entry)

        if "states" in cyc:
            cycle_states = cyc["states"]
        elif "cycle" in cyc:
            cycle_states = cyc["cycle"]
        else:
            raise KeyError("Cycle dict has neither 'states' nor 'cycle'.")

        x, y = cycle_states_to_arrays(cycle_states)
        plt.plot(x, y, "o-", linewidth=2, markersize=4, label=fr"$\gamma={gamma:.3f}$")

    except Exception as e:
        print(f"Skipping gamma={gamma}: {e}")

plt.xlabel(r"state index for $\phi_1$")
plt.ylabel(r"state index for $\phi_2$")
plt.title("Optimal recurrent cycles in phase plane")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "gamma_sweep_phase_plane_cycles.png"), dpi=200, bbox_inches="tight")


# ============================================================
# Figure 5: reward traces for selected gammas
# ============================================================

plt.figure(figsize=(8, 6))

for gamma in gammas_to_plot:
    try:
        entry = find_entry_for_gamma(results, gamma)
        cyc = entry_best_cycle(entry)

        if "rewards" not in cyc:
            raise KeyError("Cycle dict does not contain 'rewards'.")

        rewards = np.array(cyc["rewards"])
        plt.plot(
            np.arange(1, len(rewards) + 1),
            rewards,
            "o-",
            linewidth=2,
            markersize=4,
            label=fr"$\gamma={gamma:.3f}$",
        )

    except Exception as e:
        print(f"Skipping reward plot for gamma={gamma}: {e}")

plt.xlabel("Step in cycle")
plt.ylabel("Reward")
plt.title("Reward traces along optimal cycles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figdir, "gamma_sweep_reward_traces.png"), dpi=200, bbox_inches="tight")


print(f"\nSaved figures to: {figdir}/")
print("  gamma_sweep_cycle_length.png")
print("  gamma_sweep_avg_reward.png")
print("  gamma_sweep_discounted_values.png")
print("  gamma_sweep_phase_plane_cycles.png")
print("  gamma_sweep_reward_traces.png")

plt.show()