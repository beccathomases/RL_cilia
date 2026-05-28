import itertools

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Cilia3BallEnv(gym.Env):
    """
    3-hinge system with discrete relative angles and physics-based reward (Stokes flow).

    IMPORTANT ANGLE CONVENTION
    --------------------------
    The state stores relative joint angles phi = [phi1, phi2, phi3].

    Physical segment angles are cumulative:
        psi1 = phi1
        psi2 = phi1 + phi2
        psi3 = phi1 + phi2 + phi3

    This matches the MATLAB-style serial-link convention.

    Boundary modes
    --------------
      - "clip_penalty":
            clip each coordinate into range, compute physics reward using the
            effective clipped move, and add invalid_penalty if clipping occurred

      - "stay_penalty":
            if any coordinate would leave the box, stay in place and return
            invalid_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=500,
        precompute=True,
        boundary_mode="clip_penalty",
        invalid_penalty=-0.1,
        reward_rescale=100.0,
        n_bins=None,
        angle_mins=None,
        angle_maxs=None,
    ):
        super().__init__()

        # --------------------------------------------------
        # Discretization
        # --------------------------------------------------
        if n_bins is None:
            n_bins = [9, 17, 17]
        if angle_mins is None:
            angle_mins = [-np.pi / 4, -np.pi / 2, -np.pi / 2]
        if angle_maxs is None:
            angle_maxs = [ np.pi / 4,  np.pi / 2,  np.pi / 2]

        self.n_bins = np.array(n_bins, dtype=int)
        self.angle_mins = np.array(angle_mins, dtype=float)
        self.angle_maxs = np.array(angle_maxs, dtype=float)

        if self.n_bins.shape != (3,):
            raise ValueError("n_bins must have length 3")
        if self.angle_mins.shape != (3,) or self.angle_maxs.shape != (3,):
            raise ValueError("angle_mins and angle_maxs must each have length 3")
        if np.any(self.n_bins < 2):
            raise ValueError("Each entry of n_bins must be at least 2")
        if np.any(self.angle_maxs <= self.angle_mins):
            raise ValueError("Each angle_max must be greater than angle_min")

        self.dangle = (self.angle_maxs - self.angle_mins) / (self.n_bins - 1)

        # --------------------------------------------------
        # Physics parameters
        # --------------------------------------------------
        self.Nballs = 3
        self.len = 1.0 / 3.0
        self.rad = 0.05
        self.epsilon = 1.5 * self.rad
        self.X0 = np.array([0.0, 0.0, 0.0], dtype=float)

        self.dt = 1.0
        self.mu = 1.0

        # --------------------------------------------------
        # RL / MDP settings
        # --------------------------------------------------
        if boundary_mode not in {"clip_penalty", "stay_penalty"}:
            raise ValueError("boundary_mode must be 'clip_penalty' or 'stay_penalty'")

        self.boundary_mode = boundary_mode
        self.invalid_penalty = float(invalid_penalty)
        self.reward_rescale = float(reward_rescale)

        # --------------------------------------------------
        # Spaces
        # --------------------------------------------------
        self.observation_space = spaces.MultiDiscrete(self.n_bins.tolist())

        action_list = [a for a in itertools.product([-1, 0, 1], repeat=3) if a != (0, 0, 0)]
        self.action_map = np.array(action_list, dtype=int)
        self.action_space = spaces.Discrete(len(self.action_map))  # 26

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        self.state = np.array(
            [self.n_bins[0] // 2, self.n_bins[1] // 2, self.n_bins[2] // 2],
            dtype=int,
        )
        self.steps = 0
        self.max_steps = int(max_steps)

        # --------------------------------------------------
        # Precompute
        # --------------------------------------------------
        self.precompute = bool(precompute)
        self.flux_table = None
        self.next_state_table = None
        self.was_clipped_table = None

        if self.precompute:
            self._precompute_tables()

    # ============================================================
    # Angle conversion
    # ============================================================
    def bin_to_angle(self, i, hinge):
        return self.angle_mins[hinge] + i * self.dangle[hinge]

    def state_to_angles(self, state):
        state = np.asarray(state, dtype=float)
        phi = np.zeros(3, dtype=float)
        for k in range(3):
            phi[k] = self.bin_to_angle(state[k], k)
        return phi

    def state_to_segment_angles(self, state):
        phi = self.state_to_angles(state)
        psi = np.cumsum(phi)
        return psi

    # ============================================================
    # Physics
    # ============================================================
    def pos_vel_from_two_states(self, state, new_state):
        """
        Compute midpoint positions X and velocities U from a discrete transition.
        """
        state = np.asarray(state, dtype=float)
        new_state = np.asarray(new_state, dtype=float)

        mid_state = 0.5 * (new_state + state)
        state_change = new_state - state

        # relative joint angles / angular velocities
        phi = self.state_to_angles(mid_state)
        phivel = state_change * (self.dangle / self.dt)

        # cumulative segment angles / angular velocities
        psi = np.cumsum(phi)
        psivel = np.cumsum(phivel)

        X = np.zeros((self.Nballs, 3), dtype=float)
        U = np.zeros((self.Nballs, 3), dtype=float)

        # first segment / first ball
        X[0] = self.X0 + self.len * np.array(
            [np.sin(psi[0]), 0.0, np.cos(psi[0])], dtype=float
        )
        U[0] = self.len * psivel[0] * np.array(
            [np.cos(psi[0]), 0.0, -np.sin(psi[0])], dtype=float
        )

        # remaining segments / balls
        for k in range(1, self.Nballs):
            X[k] = X[k - 1] + self.len * np.array(
                [np.sin(psi[k]), 0.0, np.cos(psi[k])], dtype=float
            )
            U[k] = U[k - 1] + self.len * psivel[k] * np.array(
                [np.cos(psi[k]), 0.0, -np.sin(psi[k])], dtype=float
            )

        return X, U

    def compute_flux(self, X, U):
        """
        Reward quantity currently used in the code:
            flux ~ z-position dotted with x-force

        Includes simple physical/numerical safeguards:
          - reject states with ball centers at or below one radius above the wall
          - reject near-contact states
          - reject huge finite force or flux values
        """
        plane_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

        # --------------------------------------------------
        # Physical admissibility: keep ball centers above wall
        # --------------------------------------------------
        if np.any(X[:, 2] <= self.rad):
            return self.invalid_penalty

        # --------------------------------------------------
        # Numerical safety: reject near-contact states
        # --------------------------------------------------
        for i in range(self.Nballs):
            for j in range(i + 1, self.Nballs):
                if np.linalg.norm(X[i] - X[j]) < 2.2 * self.rad:
                    return self.invalid_penalty

        M = self.form_stokes_image_system_3D_cm(
            X, X, self.epsilon, self.mu, plane_vec
        )

        try:
            F = np.linalg.solve(M, U.flatten(order="F"))
        except np.linalg.LinAlgError:
            return self.invalid_penalty

        F = F.reshape((self.Nballs, 3), order="F")

        if not np.isfinite(F).all():
            return self.invalid_penalty

        if np.max(np.abs(F)) > 1e6:
            return self.invalid_penalty

        flux = self.dt / (np.pi * self.mu) * np.dot(X[:, 2], F[:, 0])

        if not np.isfinite(flux):
            return self.invalid_penalty

        if abs(flux) > 1e6:
            return self.invalid_penalty

        return float(flux)

    # ============================================================
    # Boundary handling helpers
    # ============================================================
    def check_state(self, state):
        state = np.asarray(state, dtype=int)
        return np.all(state >= 0) and np.all(state < self.n_bins)

    def clip_state(self, state):
        state = np.asarray(state, dtype=int)
        clipped = np.minimum(
            np.maximum(state, np.zeros(3, dtype=int)),
            self.n_bins - 1
        )
        return clipped

    def transition_info(self, state, action):
        state = np.asarray(state, dtype=int)
        delta = self.action_map[action]
        requested_state = state + delta
        is_valid_requested = self.check_state(requested_state)

        if self.boundary_mode == "clip_penalty":
            next_state = self.clip_state(requested_state)
            was_clipped = not np.array_equal(next_state, requested_state)

        elif self.boundary_mode == "stay_penalty":
            if is_valid_requested:
                next_state = requested_state.copy()
                was_clipped = False
            else:
                next_state = state.copy()
                was_clipped = True

        else:
            raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

        return {
            "requested_state": requested_state,
            "next_state": next_state,
            "was_clipped": was_clipped,
            "is_valid_requested": is_valid_requested,
        }

    def immediate_reward_from_transition(self, state, action, trans):
        state = np.asarray(state, dtype=int)
        next_state = np.asarray(trans["next_state"], dtype=int)
        was_clipped = bool(trans["was_clipped"])
        is_valid_requested = bool(trans["is_valid_requested"])

        if self.boundary_mode == "stay_penalty":
            if not is_valid_requested:
                reward = self.invalid_penalty
            else:
                X, U = self.pos_vel_from_two_states(state, next_state)
                reward = self.compute_flux(X, U)

        elif self.boundary_mode == "clip_penalty":
            X, U = self.pos_vel_from_two_states(state, next_state)
            reward = self.compute_flux(X, U)
            if was_clipped:
                reward += self.invalid_penalty

        else:
            raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

        reward *= self.reward_rescale
        return float(reward)

    # ============================================================
    # Precompute
    # ============================================================
    def _precompute_tables(self):
        n0, n1, n2 = map(int, self.n_bins)
        nA = self.action_space.n

        self.flux_table = np.zeros((n0, n1, n2, nA), dtype=float)
        self.next_state_table = np.zeros((n0, n1, n2, nA, 3), dtype=int)
        self.was_clipped_table = np.zeros((n0, n1, n2, nA), dtype=bool)

        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    state = np.array([i, j, k], dtype=int)

                    for a in range(nA):
                        trans = self.transition_info(state, a)
                        reward = self.immediate_reward_from_transition(state, a, trans)

                        self.flux_table[i, j, k, a] = reward
                        self.next_state_table[i, j, k, a, :] = trans["next_state"]
                        self.was_clipped_table[i, j, k, a] = trans["was_clipped"]

    # ============================================================
    # Gym API
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(
            [
                self.np_random.integers(0, self.n_bins[0]),
                self.np_random.integers(0, self.n_bins[1]),
                self.np_random.integers(0, self.n_bins[2]),
            ],
            dtype=int,
        )
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        assert self.action_space.contains(action)

        self.steps += 1

        if (
            self.precompute
            and self.flux_table is not None
            and self.next_state_table is not None
        ):
            i, j, k = self.state
            reward = self.flux_table[i, j, k, action]
            new_state = self.next_state_table[i, j, k, action, :].copy()
            was_clipped = bool(self.was_clipped_table[i, j, k, action])

        else:
            trans = self.transition_info(self.state, action)
            reward = self.immediate_reward_from_transition(self.state, action, trans)
            new_state = trans["next_state"].copy()
            was_clipped = trans["was_clipped"]

        self.state = new_state

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {"was_clipped": was_clipped}
        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self):
        phi = self.state_to_angles(self.state)
        psi = np.cumsum(phi)
        print(
            f"state={self.state}, "
            f"phi=({phi[0]:.6f}, {phi[1]:.6f}, {phi[2]:.6f}), "
            f"psi=({psi[0]:.6f}, {psi[1]:.6f}, {psi[2]:.6f}), "
            f"boundary_mode={self.boundary_mode}"
        )

    def close(self):
        pass

    # ============================================================
    # Stokes solver
    # ============================================================
    def form_stokes_image_system_3D_cm(self, X, X0, epsilon, mu, plane_vec):
        Nt = X.shape[0]
        Ns = X0.shape[0]

        n = plane_vec[:3]
        d = plane_vec[3] / np.linalg.norm(n)
        n = n / np.linalg.norm(n)

        M = np.zeros((3 * Nt, 3 * Ns), dtype=float)

        def block3(A11, A12, A13, A21, A22, A23, A31, A32, A33):
            top = np.hstack((A11, A12, A13))
            mid = np.hstack((A21, A22, A23))
            bot = np.hstack((A31, A32, A33))
            return np.vstack((top, mid, bot))

        Xm = np.repeat(X[:, 0][:, None], Ns, axis=1)
        Ym = np.repeat(X[:, 1][:, None], Ns, axis=1)
        Zm = np.repeat(X[:, 2][:, None], Ns, axis=1)

        X0m_expanded = np.repeat(X0[:, 0][None, :], Nt, axis=0)
        Y0m_expanded = np.repeat(X0[:, 1][None, :], Nt, axis=0)
        Z0m_expanded = np.repeat(X0[:, 2][None, :], Nt, axis=0)

        XX = (Xm - X0m_expanded) ** 2
        YY = (Ym - Y0m_expanded) ** 2
        ZZ = (Zm - Z0m_expanded) ** 2

        XY = (Xm - X0m_expanded) * (Ym - Y0m_expanded)
        XZ = (Xm - X0m_expanded) * (Zm - Z0m_expanded)
        YZ = (Ym - Y0m_expanded) * (Zm - Z0m_expanded)

        r = np.sqrt(XX + YY + ZZ)
        re = np.sqrt(r**2 + epsilon**2)

        H2 = 1.0 / re**3
        H1 = (r**2 + 2 * epsilon**2) * H2

        M_s = block3(
            H1 + H2 * XX, H2 * XY, H2 * XZ,
            H2 * XY, H1 + H2 * YY, H2 * YZ,
            H2 * XZ, H2 * YZ, H1 + H2 * ZZ,
        )
        M += M_s / (8 * np.pi * mu)

        X0_ref = X0 - 2 * ((X0 @ n - d)[:, None] * n[None, :])

        X0m_expanded = np.repeat(X0_ref[:, 0][None, :], Nt, axis=0)
        Y0m_expanded = np.repeat(X0_ref[:, 1][None, :], Nt, axis=0)
        Z0m_expanded = np.repeat(X0_ref[:, 2][None, :], Nt, axis=0)

        XX = (Xm - X0m_expanded) ** 2
        YY = (Ym - Y0m_expanded) ** 2
        ZZ = (Zm - Z0m_expanded) ** 2

        XY = (Xm - X0m_expanded) * (Ym - Y0m_expanded)
        XZ = (Xm - X0m_expanded) * (Zm - Z0m_expanded)
        YZ = (Ym - Y0m_expanded) * (Zm - Z0m_expanded)

        r = np.sqrt(XX + YY + ZZ)
        re = np.sqrt(r**2 + epsilon**2)

        H2 = 1.0 / re**3
        H1 = (r**2 + 2 * epsilon**2) * H2

        M_s_im = block3(
            H1 + H2 * XX, H2 * XY, H2 * XZ,
            H2 * XY, H1 + H2 * YY, H2 * YZ,
            H2 * XZ, H2 * YZ, H1 + H2 * ZZ,
        )
        M += -M_s_im / (8 * np.pi * mu)

        h_vals = np.abs(X0_ref @ n - d)

        P = np.kron(np.eye(3) - 2 * np.outer(n, n), np.eye(Ns))
        h = np.kron(np.eye(3), np.diag(h_vals))

        D2 = 1.0 / re**5
        D1 = (r**2 - 2 * epsilon**2) * D2
        D2 = -3 * D2

        M_pd = block3(
            D1 + D2 * XX, D2 * XY, D2 * XZ,
            D2 * XY, D1 + D2 * YY, D2 * YZ,
            D2 * XZ, D2 * YZ, D1 + D2 * ZZ,
        )
        M_pd = M_pd @ (h @ h) @ P / (4 * np.pi * mu)
        M += M_pd

        Xhat = Xm - X0m_expanded
        Yhat = Ym - Y0m_expanded
        Zhat = Zm - Z0m_expanded

        H3 = -(r**2 + 4 * epsilon**2) / re**5
        H4 = D2

        SD1 = block3(
            H2 * Xhat * n[0], H2 * Xhat * n[1], H2 * Xhat * n[2],
            H2 * Yhat * n[0], H2 * Yhat * n[1], H2 * Yhat * n[2],
            H2 * Zhat * n[0], H2 * Zhat * n[1], H2 * Zhat * n[2],
        )

        XdotN = Xhat * n[0] + Yhat * n[1] + Zhat * n[2]
        D_sd = H2 * XdotN

        zero_block_ns = np.zeros((Nt, Ns), dtype=float)
        SD2 = np.block(
            [
                [D_sd, zero_block_ns, zero_block_ns],
                [zero_block_ns, D_sd, zero_block_ns],
                [zero_block_ns, zero_block_ns, D_sd],
            ]
        )

        SD3 = block3(
            H3 * Xhat * n[0], H3 * Yhat * n[0], H3 * Zhat * n[0],
            H3 * Xhat * n[1], H3 * Yhat * n[1], H3 * Zhat * n[1],
            H3 * Xhat * n[2], H3 * Yhat * n[2], H3 * Zhat * n[2],
        )

        SD4 = block3(
            H4 * XdotN * XX, H4 * XdotN * XY, H4 * XdotN * XZ,
            H4 * XdotN * XY, H4 * XdotN * YY, H4 * XdotN * YZ,
            H4 * XdotN * XZ, H4 * XdotN * YZ, H4 * XdotN * ZZ,
        )

        M_sd = (SD1 + SD2 + SD3 + SD4) @ (-2 * h @ P) / (8 * np.pi * mu)
        M += M_sd

        H5 = H3 + H2

        R1 = block3(
            H5 * Xhat * n[0], H5 * Yhat * n[0], H5 * Zhat * n[0],
            H5 * Xhat * n[1], H5 * Yhat * n[1], H5 * Zhat * n[1],
            H5 * Xhat * n[2], H5 * Yhat * n[2], H5 * Zhat * n[2],
        )

        D_rot = -H5 * XdotN
        R2 = np.block(
            [
                [D_rot, zero_block_ns, zero_block_ns],
                [zero_block_ns, D_rot, zero_block_ns],
                [zero_block_ns, zero_block_ns, D_rot],
            ]
        )

        M_rot = (R1 + R2) @ (2 * h) / (8 * np.pi * mu)
        M += M_rot

        return M