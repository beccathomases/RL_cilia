#!/usr/bin/env python3
"""
check_scaled_radius_geometry.py
===============================

Run from phase6_factored:

    python check_scaled_radius_geometry.py
    python check_scaled_radius_geometry.py --nballs 8 9 10 12 --rad-scale 0.4
    python check_scaled_radius_geometry.py --nballs 10 --rad 0.04
"""

import argparse
import numpy as np
from cilia_n_ball_env import CiliaNBallEnv


def resolve_rad(N, rad=None, rad_scale=None):
    if rad is not None and rad_scale is not None:
        raise ValueError("Use either --rad or --rad-scale, not both.")
    if rad is not None:
        return float(rad)
    if rad_scale is not None:
        return float(rad_scale) / int(N)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nballs", nargs="+", type=int, default=[8, 9, 10, 11, 12])
    p.add_argument("--rad", type=float, default=None)
    p.add_argument("--rad-scale", type=float, default=None)
    p.add_argument("--dtheta", type=float, default=np.pi / 20)
    p.add_argument("--action-mode", type=str, default="factored", choices=["factored", "monolithic"])
    args = p.parse_args()

    print("N   seg        rad        2.2rad     seg/(2.2rad)   noop_reward")
    print("-" * 76)

    for N in args.nballs:
        rad_eff = resolve_rad(N, rad=args.rad, rad_scale=args.rad_scale)

        kwargs = dict(
            Nballs=N,
            dtheta=args.dtheta,
            precompute=False,
            reset_mode="midpoint",
            action_mode=args.action_mode,
            verbose=False,
        )
        if rad_eff is not None:
            kwargs["rad"] = rad_eff

        env = CiliaNBallEnv(**kwargs)
        env.reset()

        if args.action_mode == "factored":
            action = np.ones(N, dtype=int)  # no-op: 1 -> zero increment
        else:
            action = 0  # monolithic excludes global no-op; placeholder only

        reward = env.step(action)[1]
        rad_actual = float(getattr(env, "rad", np.nan))
        seg = 1.0 / N
        threshold = 2.2 * rad_actual
        margin = seg / threshold if threshold > 0 else np.nan

        print(
            f"{N:2d}  {seg:9.5f}  {rad_actual:9.5f}  {threshold:9.5f}  "
            f"{margin:13.5f}  {float(reward):11.5g}"
        )


if __name__ == "__main__":
    main()
