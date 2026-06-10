#!/usr/bin/env python3
"""
flux_optimal_stroke.py

Continuum gain-optimal pumping stroke for the N-ball cilia PUMP, computed by
classical optimization over a temporal Fourier loop, compared head-to-head
with the Howard average-reward optimum.

Model reminder (from cilia_n_ball_env.py): the base is pinned at the origin,
so this is a wall-anchored pump, NOT a free swimmer. The per-step reward is the
pumped flux  ~  sum_i z_i F_{x,i}  with  F = M^{-1} U, M the regularized-
Stokeslet + wall-image mobility. That reward is linear in the shape increment,
so flux is a 1-form on shape space and the cycle flux is a geometric phase.

Physics is reused VERBATIM from CiliaNBallEnv:
  - pos_vel_from_two_states  (accepts continuous bin-coords -> arbitrary,
    non-lattice shape increments are fine; positions use the midpoint shape)
  - form_stokes_image_system_3D_cm  (same kernel, same epsilon = 1.5*rad)

Objective = GAIN (average reward per step), which is what Howard's eta and PPO's
<R> actually are. In the continuum this is the L-infinity isoperimetric ratio
    g = Q_cyc / L_steps,     L_steps = (Chebyshev arc length) / dtheta,
because one monolithic step advances each active joint by dtheta. g is rate-
independent; g/dtheta is the dtheta->0 ceiling Howard approaches from below.

Expected ladder at fixed dtheta:  RL  <=  Howard  <=  g_continuum.

Also reports flux-per-beat Q_cyc and rho = Q_cyc/|A_tip| for both the continuum
optimum and the Howard cycle, plus (N=2) the curl field-strength map.

Built-in validation: walks Howard's own lattice cycle through this same flux
routine and checks the recovered gain / rho against howard_summary_{N}ball.npz.

The objective lives in one function (`metrics_of_loop`); swapping it for an
efficiency / Pareto objective later is a one-function change.

Outputs to OUT_DIR:
  flux_optimal_{N}ball.json   all numbers + Fourier coefficients + validation
  flux_optimal_{N}ball.npy    optimal loop angles, shape (N_SAMPLES, N)
  flux_optimal_{N}ball.png    shape-space map/loops + tip paths
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

# ============================ CONFIG (edit here) ============================
N            = 2                # number of balls; start at 2 for the clean 2-D map
ENV_DIR      = "."              # dir containing cilia_n_ball_env.py
HOWARD_DIR = "../phase4_Nball_exploration/results/howard/dtheta_pi20"             # dir with howard_stroke_{N}ball.npy + howard_summary_{N}ball.npz
OUT_DIR      = "./results/flux_optimal_strokes_3"              # where to write outputs

# Physics -- MUST match the env config used to BUILD the Howard vi_tables.
RAD          = 0.05             # fixed-radius small-N regime
EPSILON      = None             # None -> env uses 1.5*RAD (coupled, as in the env)
MU           = 1.0
DT           = 1.0
REWARD_RESCALE = 100.0          # env default; keeps units identical to Howard R / eta

DTHETA_OVERRIDE = None          # None -> auto-detect from the Howard grid

# Fourier loop / optimizer
N_HARMONICS  = 2                # K
N_SAMPLES    = 180              # M points around the loop
DE_MAXITER   = 250
DE_POPSIZE   = 18
DE_SEED      = 0
COEFF_BOUND_SCALE = 1.0         # harmonic-amplitude bound as fraction of half-range
POLISH       = True             # Nelder-Mead polish after differential evolution
# ===========================================================================

sys.path.insert(0, os.path.abspath(ENV_DIR))
from cilia_n_ball_env import CiliaNBallEnv  # noqa: E402

from scipy.optimize import differential_evolution, minimize  # noqa: E402

PLANE_VEC = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)


# ----------------------------------------------------------------------------
# Grid / angle conventions (identical to the env and the Howard script)
# ----------------------------------------------------------------------------
def angle_ranges(n):
    mins = np.array([-np.pi / 4] + [-np.pi / 2] * (n - 1), dtype=float)
    maxs = np.array([np.pi / 4] + [np.pi / 2] * (n - 1), dtype=float)
    return mins, maxs


def detect_dtheta(n):
    """Auto-detect dtheta from the Howard policy grid; fall back to override."""
    if DTHETA_OVERRIDE is not None:
        return float(DTHETA_OVERRIDE)
    pol_path = os.path.join(HOWARD_DIR, f"howard_policy_{n}ball.npy")
    if not os.path.exists(pol_path):
        raise FileNotFoundError(
            f"{pol_path} not found and DTHETA_OVERRIDE is None; "
            "set DTHETA_OVERRIDE to the dtheta used for this N."
        )
    n_bins = np.load(pol_path).shape
    mins, maxs = angle_ranges(n)
    cand = (maxs - mins) / (np.asarray(n_bins, float) - 1.0)
    if not np.allclose(cand, cand[0]):
        print(f"  WARNING: non-uniform dtheta across coords: {cand}")
    return float(cand[0])


# ----------------------------------------------------------------------------
# Physics primitives -- thin wrappers over the env
# ----------------------------------------------------------------------------
class FluxModel:
    def __init__(self, env):
        self.env = env
        self.mins, self.maxs = env.angle_mins, env.angle_maxs
        self.dangle = env.dangle
        self.N = env.Nballs
        self.rad = env.rad

    def ang2bin(self, phi):
        return (np.asarray(phi, float) - self.mins) / self.dangle

    def in_box(self, phi):
        return bool(np.all(phi >= self.mins - 1e-9) and np.all(phi <= self.maxs + 1e-9))

    def node_positions(self, phi):
        """Bead positions at a single shape phi (U=0)."""
        c = self.ang2bin(phi)
        X, _ = self.env.pos_vel_from_two_states(c, c)
        return X

    def flux_move(self, phi_a, phi_b):
        """
        Pumped flux (raw, pre-rescale) for the move phi_a -> phi_b, using the
        env's midpoint-shape construction and regularized image-system mobility.
        Returns (flux, feasible). Feasibility = wall + nonadjacent-contact at the
        midpoint shape (exactly the env's compute_flux guards).
        """
        c_a, c_b = self.ang2bin(phi_a), self.ang2bin(phi_b)
        X, U = self.env.pos_vel_from_two_states(c_a, c_b)

        if np.any(X[:, 2] <= self.rad):
            return 0.0, False
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if np.linalg.norm(X[i] - X[j]) < 2.2 * self.rad:
                    return 0.0, False

        M = self.env.form_stokes_image_system_3D_cm(
            X, X, self.env.epsilon, self.env.mu, PLANE_VEC
        )
        try:
            F = np.linalg.solve(M, U.flatten(order="F")).reshape((self.N, 3), order="F")
        except np.linalg.LinAlgError:
            return 0.0, False
        if not np.isfinite(F).all() or np.max(np.abs(F)) > 1e6:
            return 0.0, False

        flux = self.env.dt / (np.pi * self.env.mu) * float(np.dot(X[:, 2], F[:, 0]))
        if not np.isfinite(flux) or abs(flux) > 1e6:
            return 0.0, False
        return flux, True

    def connection(self, phi, delta=1e-3):
        """Flux 1-form A_j(phi) ~ flux for a unit-rate move along coord j."""
        A = np.zeros(self.N)
        for j in range(self.N):
            da = np.zeros(self.N)
            da[j] = delta
            f, ok = self.flux_move(phi - 0.5 * da, phi + 0.5 * da)
            A[j] = (f / delta) if ok else np.nan
        return A


# ----------------------------------------------------------------------------
# Fourier loop parameterization
# ----------------------------------------------------------------------------
def n_params(n, K):
    return n * (1 + 2 * K)


def unpack(params, n, K):
    p = np.asarray(params, float).reshape(n, 1 + 2 * K)
    a0 = p[:, 0]
    a = p[:, 1:1 + K]
    b = p[:, 1 + K:1 + 2 * K]
    return a0, a, b


def loop_angles(params, n, K, M):
    """Return phi at M loop nodes, shape (M, n). Closed: node M wraps to 0."""
    a0, a, b = unpack(params, n, K)
    s = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    ks = np.arange(1, K + 1)[None, :] * s[:, None]      # (M, K)
    cos, sin = np.cos(ks), np.sin(ks)
    phi = a0[None, :] + cos @ a.T + sin @ b.T           # (M, n)
    return phi


# ----------------------------------------------------------------------------
# Loop metrics -- THE OBJECTIVE lives here
# ----------------------------------------------------------------------------
def metrics_of_loop(params, model, n, K, M, dtheta):
    phi = loop_angles(params, n, K, M)
    phi_next = np.roll(phi, -1, axis=0)

    fluxes = np.empty(M)
    feas = np.ones(M, dtype=bool)
    for k in range(M):
        if not model.in_box(phi[k]):
            feas[k] = False
            fluxes[k] = 0.0
            continue
        f, ok = model.flux_move(phi[k], phi_next[k])
        fluxes[k] = f
        feas[k] = ok

    feasible_frac = float(np.mean(feas))
    Q_raw = float(np.sum(fluxes))                       # sum of raw per-step flux
    Q_cyc = REWARD_RESCALE * Q_raw                      # total reward around loop (Howard units)

    # L-infinity (Chebyshev) arc length -> number of monolithic steps
    dphi = phi_next - phi
    Linf = float(np.sum(np.max(np.abs(dphi), axis=1)))
    L_steps = Linf / dtheta if dtheta > 0 else np.inf

    gain = Q_cyc / L_steps if L_steps > 0 else 0.0      # per-step gain (compare to Howard eta)

    # tip-loop signed area in the x-z plane (tip = last ball)
    tip = np.array([model.node_positions(phi[k])[-1] for k in range(M)])
    x, z = tip[:, 0], tip[:, 2]
    A_tip = 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))

    rho = Q_cyc / abs(A_tip) if abs(A_tip) > 1e-12 else np.nan

    return {
        "gain": gain,
        "gain_per_dtheta": gain / dtheta if dtheta > 0 else np.nan,
        "Q_cyc": Q_cyc,
        "A_tip": A_tip,
        "rho": rho,
        "Linf": Linf,
        "L_steps": L_steps,
        "feasible_frac": feasible_frac,
        "phi": phi,
        "tip": tip,
    }


def objective(params, model, n, K, M, dtheta):
    """Minimized by the optimizer: -gain, with a feasibility penalty."""
    m = metrics_of_loop(params, model, n, K, M, dtheta)
    if m["feasible_frac"] < 1.0:
        return 1.0e3 * (1.0 + (1.0 - m["feasible_frac"]))
    return -m["gain"]


# ----------------------------------------------------------------------------
# Howard cycle comparison + validation
# ----------------------------------------------------------------------------
def load_howard_cycle(n):
    stroke_p = os.path.join(HOWARD_DIR, f"howard_stroke_{n}ball.npy")
    summ_p = os.path.join(HOWARD_DIR, f"howard_summary_{n}ball.npz")
    if not (os.path.exists(stroke_p) and os.path.exists(summ_p)):
        return None
    angles = np.load(stroke_p)                          # (T+1, n)
    with np.load(summ_p, allow_pickle=True) as z:
        cs = int(z["cycle_start"])
        cl = int(z["cycle_length"])
        rewards = np.asarray(z["rewards"], float)       # (T,) rescaled rewards
    if cs < 0 or cl <= 0:
        print("  WARNING: Howard rollout did not close a cycle; skipping comparison.")
        return None
    cyc_nodes = angles[cs:cs + cl + 1]                  # (cl+1, n), node cl == node 0
    cyc_rewards = rewards[cs:cs + cl]                   # (cl,)
    return {"nodes": cyc_nodes, "rewards": cyc_rewards, "cycle_length": cl}


def howard_metrics(model, hc, dtheta):
    nodes = hc["nodes"]
    cl = hc["cycle_length"]

    # Re-derive flux around Howard's lattice cycle with OUR pipeline (validation).
    flux_ours = np.empty(cl)
    for k in range(cl):
        f, ok = model.flux_move(nodes[k], nodes[k + 1])
        flux_ours[k] = f if ok else np.nan
    Q_ours = REWARD_RESCALE * float(np.nansum(flux_ours))
    gain_ours = Q_ours / cl

    # Authoritative values straight from the Howard summary.
    Q_summary = float(np.sum(hc["rewards"]))
    gain_summary = Q_summary / cl

    tip = np.array([model.node_positions(nodes[k])[-1] for k in range(cl)])
    x, z = tip[:, 0], tip[:, 2]
    A_tip = 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))
    rho_summary = Q_summary / abs(A_tip) if abs(A_tip) > 1e-12 else np.nan

    return {
        "cycle_length": cl,
        "gain_summary": gain_summary,
        "gain_per_dtheta_summary": gain_summary / dtheta,
        "gain_ours": gain_ours,
        "Q_cyc_summary": Q_summary,
        "Q_cyc_ours": Q_ours,
        "A_tip": A_tip,
        "rho_summary": rho_summary,
        "gain_validation_rel_err": abs(gain_ours - gain_summary) / (abs(gain_summary) + 1e-12),
        "nodes": nodes,
        "tip": tip,
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_figure(n, model, opt, how, dtheta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax0 = axes[0]
    if n == 2:
        mins, maxs = model.mins, model.maxs
        g0 = np.linspace(mins[0], maxs[0], 61)
        g1 = np.linspace(mins[1], maxs[1], 61)
        curl = np.full((g1.size, g0.size), np.nan)
        h = 1e-3
        for ii, p0 in enumerate(g0):
            for jj, p1 in enumerate(g1):
                A_pp = model.connection(np.array([p0, p1 + h]))[0]
                A_pm = model.connection(np.array([p0, p1 - h]))[0]
                A_qp = model.connection(np.array([p0 + h, p1]))[1]
                A_qm = model.connection(np.array([p0 - h, p1]))[1]
                if np.any(np.isnan([A_pp, A_pm, A_qp, A_qm])):
                    continue
                curl[jj, ii] = (A_qp - A_qm) / (2 * h) - (A_pp - A_pm) / (2 * h)
        vmax = np.nanmax(np.abs(curl)) if np.isfinite(np.nanmax(np.abs(curl))) else 1.0
        im = ax0.pcolormesh(g0, g1, curl, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax0, label=r"flux field strength  $\partial_0 A_1-\partial_1 A_0$")
        ax0.set_xlabel(r"$\phi_0$ (base orientation)")
        ax0.set_ylabel(r"$\phi_1$ (joint)")
    else:
        ax0.set_xlabel(r"$\phi_0$")
        ax0.set_ylabel(r"$\phi_1$")

    ax0.plot(opt["phi"][:, 0], opt["phi"][:, 1], "-", lw=2.4, color="k",
             label="continuum gain-optimal")
    ax0.plot(np.r_[opt["phi"][:, 0], opt["phi"][0, 0]],
             np.r_[opt["phi"][:, 1], opt["phi"][0, 1]], "-", lw=2.4, color="k")
    if how is not None:
        ax0.plot(how["nodes"][:, 0], how["nodes"][:, 1], "o-", ms=3, lw=1.4,
                 color="tab:green", label="Howard cycle")
    ax0.set_title(f"N={n} shape-space loops" + ("  (+ field strength)" if n == 2 else ""))
    ax0.legend(loc="best", fontsize=8)

    ax1 = axes[1]
    t = opt["tip"]
    ax1.plot(np.r_[t[:, 0], t[0, 0]], np.r_[t[:, 2], t[0, 2]], "-", lw=2.2, color="k",
             label="continuum optimal")
    if how is not None:
        th = how["tip"]
        ax1.plot(np.r_[th[:, 0], th[0, 0]], np.r_[th[:, 2], th[0, 2]], "o-", ms=3,
                 lw=1.4, color="tab:green", label="Howard")
    ax1.set_xlabel("tip x")
    ax1.set_ylabel("tip z")
    ax1.set_title("tip path (x-z plane)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    K, M = N_HARMONICS, N_SAMPLES

    dtheta = detect_dtheta(N)
    mins, maxs = angle_ranges(N)

    env = CiliaNBallEnv(
        Nballs=N, dtheta=dtheta, rad=RAD, epsilon=EPSILON, mu=MU, dt=DT,
        reward_rescale=REWARD_RESCALE, action_mode="monolithic",
        precompute=False, verbose=False, angle_mins=mins, angle_maxs=maxs,
    )
    model = FluxModel(env)

    print(f"N={N}  dtheta={dtheta:.6g}  n_bins={env.n_bins.tolist()}  "
          f"rad={env.rad}  epsilon={env.epsilon:.6g}  rescale={REWARD_RESCALE}")
    print(f"Fourier: K={K} harmonics, {M} samples, "
          f"{n_params(N, K)} parameters")

    # ---- bounds: a0 in box; harmonic amplitudes scaled by half-range ----
    half = 0.5 * (maxs - mins)
    bounds = []
    for j in range(N):
        bounds.append((mins[j], maxs[j]))                       # a0_j
        for _ in range(2 * K):
            bounds.append((-COEFF_BOUND_SCALE * half[j],
                           COEFF_BOUND_SCALE * half[j]))         # a_jk, b_jk

    print("Running differential evolution ...")
    res = differential_evolution(
        objective, bounds, args=(model, N, K, M, dtheta),
        maxiter=DE_MAXITER, popsize=DE_POPSIZE, seed=DE_SEED,
        tol=1e-10, mutation=(0.4, 1.2), recombination=0.8,
        polish=False, init="sobol", updating="deferred",
    )
    best = res.x

    if POLISH:
        print("Nelder-Mead polish ...")
        pol = minimize(objective, best, args=(model, N, K, M, dtheta),
                       method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
        if pol.fun < res.fun:
            best = pol.x

    opt = metrics_of_loop(best, model, N, K, M, dtheta)
    if opt["feasible_frac"] < 1.0:
        print(f"  WARNING: optimum has feasible_frac={opt['feasible_frac']:.3f} "
              "(some loop samples violate wall/contact). Tighten COEFF_BOUND_SCALE "
              "or raise N_SAMPLES.")

    # orient for positive (+x) pumping
    if opt["gain"] < 0:
        print("  (found a -x pumping loop; reporting its sign as-is)")

    print("\n=== CONTINUUM GAIN-OPTIMAL ===")
    print(f"  gain (per step)      = {opt['gain']:.6g}")
    print(f"  gain / dtheta        = {opt['gain_per_dtheta']:.6g}   <-- the ceiling")
    print(f"  Q_cyc (flux/beat)    = {opt['Q_cyc']:.6g}")
    print(f"  A_tip                = {opt['A_tip']:.6g}")
    print(f"  rho = Q_cyc/|A_tip|  = {opt['rho']:.6g}")

    how = load_howard_cycle(N)
    hm = None
    if how is not None:
        hm = howard_metrics(model, how, dtheta)
        print("\n=== HOWARD (from summary) ===")
        print(f"  cycle length         = {hm['cycle_length']}")
        print(f"  gain (per step)      = {hm['gain_summary']:.6g}")
        print(f"  gain / dtheta        = {hm['gain_per_dtheta_summary']:.6g}")
        print(f"  Q_cyc                = {hm['Q_cyc_summary']:.6g}")
        print(f"  rho                  = {hm['rho_summary']:.6g}")
        print("  --- pipeline validation (our flux on Howard's cycle) ---")
        print(f"  gain (ours)          = {hm['gain_ours']:.6g}")
        print(f"  rel. err vs summary  = {hm['gain_validation_rel_err']:.3e}")
        print("\n=== LADDER (expect Howard <= continuum) ===")
        print(f"  Howard    gain/dtheta = {hm['gain_per_dtheta_summary']:.6g}")
        print(f"  continuum gain/dtheta = {opt['gain_per_dtheta']:.6g}")
    else:
        print("\n(No Howard outputs found in HOWARD_DIR; skipping comparison.)")

    # ---- save ----
    stem = os.path.join(OUT_DIR, f"flux_optimal_{N}ball")
    np.save(stem + ".npy", opt["phi"])

    payload = {
        "config": {
            "N": N, "dtheta": dtheta, "n_bins": env.n_bins.tolist(),
            "rad": env.rad, "epsilon": env.epsilon, "mu": env.mu, "dt": env.dt,
            "reward_rescale": REWARD_RESCALE, "n_harmonics": K, "n_samples": M,
            "de_maxiter": DE_MAXITER, "de_popsize": DE_POPSIZE, "de_seed": DE_SEED,
            "coeff_bound_scale": COEFF_BOUND_SCALE, "polish": POLISH,
        },
        "continuum_optimum": {
            "gain": opt["gain"], "gain_per_dtheta": opt["gain_per_dtheta"],
            "Q_cyc": opt["Q_cyc"], "A_tip": opt["A_tip"], "rho": opt["rho"],
            "Linf_arclength": opt["Linf"], "L_steps": opt["L_steps"],
            "feasible_frac": opt["feasible_frac"],
            "fourier_params": np.asarray(best, float).reshape(N, 1 + 2 * K).tolist(),
        },
    }
    if hm is not None:
        payload["howard"] = {k: v for k, v in hm.items()
                             if k not in ("nodes", "tip")}
        payload["ladder"] = {
            "howard_gain_per_dtheta": hm["gain_per_dtheta_summary"],
            "continuum_gain_per_dtheta": opt["gain_per_dtheta"],
            "continuum_minus_howard": opt["gain_per_dtheta"] - hm["gain_per_dtheta_summary"],
        }
    with open(stem + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    make_figure(N, model, opt, hm, dtheta, stem + ".png")

    print(f"\nWrote:\n  {stem}.json\n  {stem}.npy\n  {stem}.png")


if __name__ == "__main__":
    main()