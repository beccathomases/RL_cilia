#!/usr/bin/env python3
"""
patch_phase6_scaled_radius.py
=============================

Run from phase6_factored:

    python patch_phase6_scaled_radius.py

It reads your current ppo_sweep_phase6.py and writes a new file:

    ppo_sweep_phase6_scaledrad.py

The original file is left untouched. The new file adds:

    --rad FLOAT          explicit bead radius
    --rad-scale FLOAT    use rad = FLOAT/Nballs, e.g. --rad-scale 0.4

It also adds pi40 and pi50 to DTHETA_CASES if they are not already present,
and records rad/rad_label in each summary.json.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path("ppo_sweep_phase6.py")
OUT = Path("ppo_sweep_phase6_scaledrad.py")

if not SRC.exists():
    raise FileNotFoundError(
        "Could not find ppo_sweep_phase6.py. Run this from phase6_factored."
    )

text = SRC.read_text()
orig = text

# ---------------------------------------------------------------------
# 1. Extend dtheta cases.
# ---------------------------------------------------------------------
if '"pi40"' not in text:
    text = text.replace(
        '    "pi30": np.pi / 30,\n}',
        '    "pi30": np.pi / 30,\n    "pi40": np.pi / 40,\n    "pi50": np.pi / 50,\n}',
    )

# ---------------------------------------------------------------------
# 2. Insert radius helpers after DTHETA_CASES block.
# ---------------------------------------------------------------------
helper = r'''

# ---------------------------------------------------------------------
# Radius / geometry scaling
# ---------------------------------------------------------------------

def resolve_rad(nballs: int, rad: float | None = None, rad_scale: float | None = None):
    """
    Decide the bead radius used in the environment.

    rad:
        Explicit radius, e.g. --rad 0.04.

    rad_scale:
        Constant relative thickness, e.g. --rad-scale 0.4 gives rad=0.4/N.

    If neither is supplied, return None and let CiliaNBallEnv use its default.
    """
    if rad is not None and rad_scale is not None:
        raise ValueError("Use either --rad or --rad-scale, not both.")

    if rad is not None:
        return float(rad)

    if rad_scale is not None:
        return float(rad_scale) / int(nballs)

    return None


def radius_label(rad_effective: float | None):
    if rad_effective is None:
        return "default"
    return f"{rad_effective:.8g}"
'''

if "def resolve_rad(" not in text:
    m = re.search(r"DTHETA_CASES\s*=\s*\{.*?\}\n", text, flags=re.S)
    if not m:
        raise RuntimeError("Could not locate DTHETA_CASES block.")
    text = text[: m.end()] + helper + text[m.end():]

# ---------------------------------------------------------------------
# 3. make_raw_env signature and conditional rad insertion.
# ---------------------------------------------------------------------
text = text.replace(
    'def make_raw_env(nballs: int, dtheta: float, seed: int | None = None,\n'
    '                 action_mode: str = "monolithic"):',
    'def make_raw_env(nballs: int, dtheta: float, seed: int | None = None,\n'
    '                 action_mode: str = "monolithic", rad: float | None = None):',
)

insert_after_attempts = '''    ]\n\n    last_err = None'''
replace_after_attempts = '''    ]\n\n    if rad is not None:\n        for kwargs in attempts:\n            kwargs["rad"] = float(rad)\n\n    last_err = None'''
if 'kwargs["rad"] = float(rad)' not in text:
    if insert_after_attempts not in text:
        raise RuntimeError("Could not find end of attempts list in make_raw_env().")
    text = text.replace(insert_after_attempts, replace_after_attempts, 1)

# ---------------------------------------------------------------------
# 4. make_train_env signature and call.
# ---------------------------------------------------------------------
text = text.replace(
    'def make_train_env(nballs: int, dtheta: float, seed: int, action_mode: str = "monolithic"):',
    'def make_train_env(nballs: int, dtheta: float, seed: int, action_mode: str = "monolithic",\n'
    '                   rad: float | None = None):',
)
text = text.replace(
    'env = make_raw_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode)',
    'env = make_raw_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode, rad=rad)',
)

# ---------------------------------------------------------------------
# 5. train_one signature and train/eval env calls.
# ---------------------------------------------------------------------
text = text.replace(
    '    action_mode: str = "monolithic",\n    force: bool = False,',
    '    action_mode: str = "monolithic",\n    rad: float | None = None,\n    force: bool = False,',
)
text = text.replace(
    'train_env = make_train_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode)',
    'train_env = make_train_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode, rad=rad)',
)
text = text.replace(
    'eval_env = make_raw_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode)',
    'eval_env = make_raw_env(nballs=nballs, dtheta=dtheta, seed=seed, action_mode=action_mode, rad=rad)',
)

# Include rad in the train print, if exact line is present.
text = text.replace(
    'print(f"[train] N={nballs}, action_mode={action_mode}, dtheta={dtheta_label}, seed={seed}, timesteps={timesteps}")',
    'print(f"[train] N={nballs}, action_mode={action_mode}, dtheta={dtheta_label}, seed={seed}, timesteps={timesteps}, rad={radius_label(rad)}")',
)

# Add rad to summary row.
text = text.replace(
    '        "elapsed_sec": float(elapsed),\n        "cycle_start": int(cycle_start),',
    '        "elapsed_sec": float(elapsed),\n        "rad": float(rad) if rad is not None else np.nan,\n        "rad_label": radius_label(rad),\n        "cycle_start": int(cycle_start),',
)

# ---------------------------------------------------------------------
# 6. CLI args.
# ---------------------------------------------------------------------
cli_insert = '''    p.add_argument("--force", action="store_true")\n\n    return p.parse_args()'''
cli_replace = '''    p.add_argument("--force", action="store_true")\n    p.add_argument("--rad", type=float, default=None,\n                   help="Explicit bead radius, e.g. --rad 0.04. Do not combine with --rad-scale.")\n    p.add_argument("--rad-scale", type=float, default=None,\n                   help="Use rad = RAD_SCALE/Nballs, e.g. --rad-scale 0.4.")\n\n    return p.parse_args()'''
if '"--rad-scale"' not in text:
    if cli_insert not in text:
        raise RuntimeError("Could not find parse_args() insertion point.")
    text = text.replace(cli_insert, cli_replace, 1)

# ---------------------------------------------------------------------
# 7. main(): compute rad_effective and pass it into train_one.
# ---------------------------------------------------------------------
main_insert = '''def main():\n    args = parse_args()\n\n    root = Path(args.outroot)'''
main_replace = '''def main():\n    args = parse_args()\n    rad_effective = resolve_rad(args.nballs, rad=args.rad, rad_scale=args.rad_scale)\n\n    print("=" * 72)\n    print("[geometry]")\n    print(f"N={args.nballs}")\n    print(f"action_mode={args.action_mode}")\n    print(f"rad={radius_label(rad_effective)}")\n    if rad_effective is not None:\n        seg = 1.0 / args.nballs\n        thresh = 2.2 * rad_effective\n        print(f"segment length = {seg:.8g}")\n        print(f"2.2*rad       = {thresh:.8g}")\n        print(f"seg/(2.2rad)  = {seg/thresh:.6g}")\n    else:\n        print("rad uses CiliaNBallEnv default")\n    print("=" * 72)\n\n    root = Path(args.outroot)'''
if 'rad_effective = resolve_rad' not in text:
    if main_insert not in text:
        raise RuntimeError("Could not find main() insertion point.")
    text = text.replace(main_insert, main_replace, 1)

text = text.replace(
    '                action_mode=args.action_mode,\n                force=args.force,',
    '                action_mode=args.action_mode,\n                rad=rad_effective,\n                force=args.force,',
)

# ---------------------------------------------------------------------
# 8. Clean labels.
# ---------------------------------------------------------------------
text = text.replace('print("N=5 PPO sweep summary")', 'print("Phase 6 PPO sweep summary")')
text = text.replace('print("\\n[done] Phase 5 N=5 PPO sweep complete.")', 'print("\\n[done] Phase 6 PPO sweep complete.")')
text = text.replace('default="results/ppo_sweeps_n5"', 'default="results/ppo_sweeps_phase6"')

if text == orig:
    raise RuntimeError("No changes were made; patch script did not match your file.")

OUT.write_text(text)
print(f"[wrote] {OUT}")
print("Next: python -m py_compile ppo_sweep_phase6_scaledrad.py")
