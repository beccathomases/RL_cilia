#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

def seed_from_path(path):
    m = re.search(r"seed_(\d+)", str(path))
    return int(m.group(1)) if m else -1

def load_cycle_tip(path):
    data = np.load(path, allow_pickle=True)
    tip_all = np.asarray(data["tip"], dtype=float)
    i0 = int(np.asarray(data["cycle_start"]).reshape(-1)[0])
    L = int(np.asarray(data["cycle_length"]).reshape(-1)[0])
    return tip_all[i0:i0+L]

def resample_periodic(points, M=240):
    pts = np.asarray(points, dtype=float)
    L = len(pts)
    pts2 = np.vstack([pts, pts[0]])
    t_old = np.arange(L+1)
    t_new = np.linspace(0, L, M, endpoint=False)
    x = np.interp(t_new, t_old, pts2[:,0])
    z = np.interp(t_new, t_old, pts2[:,1])
    return np.column_stack([x, z])

def normalize(points, center=True, scale=True):
    q = np.asarray(points, dtype=float).copy()
    if center:
        q -= q.mean(axis=0, keepdims=True)
    if scale:
        s = np.sqrt(np.mean(np.sum(q*q, axis=1)))
        if s > 0:
            q /= s
    return q

def best_shift_score(a, b):
    # a, b same shape, already normalized
    best = np.inf
    best_shift = 0
    for k in range(len(a)):
        bb = np.roll(b, k, axis=0)
        score = np.sqrt(np.mean(np.sum((a - bb)**2, axis=1)))
        if score < best:
            best = score
            best_shift = k
    return best, best_shift

def signed_area(points):
    pts = np.asarray(points, dtype=float)
    x, z = pts[:,0], pts[:,1]
    x2 = np.r_[x, x[0]]
    z2 = np.r_[z, z[0]]
    return 0.5*np.sum(x2[:-1]*z2[1:] - x2[1:]*z2[:-1])

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True, help="case dir containing seed_*/result.npz")
ap.add_argument("--ref-seed", type=int, default=None)
ap.add_argument("--M", type=int, default=240)
args = ap.parse_args()

case = Path(args.case)
paths = sorted(case.glob("seed_*/result.npz"), key=seed_from_path)
if not paths:
    raise SystemExit(f"No result.npz files found under {case}")

ref_path = None
if args.ref_seed is not None:
    for p in paths:
        if seed_from_path(p) == args.ref_seed:
            ref_path = p
            break
if ref_path is None:
    ref_path = paths[0]

ref_seed = seed_from_path(ref_path)
ref = normalize(resample_periodic(load_cycle_tip(ref_path), args.M))

rows = []
for p in paths:
    seed = seed_from_path(p)
    tip = load_cycle_tip(p)
    q = normalize(resample_periodic(tip, args.M))

    q_mirror = q.copy()
    q_mirror[:,0] *= -1

    q_rev = q[::-1].copy()
    q_mirror_rev = q_mirror[::-1].copy()

    same, _ = best_shift_score(ref, q)
    mirror, _ = best_shift_score(ref, q_mirror)
    rev, _ = best_shift_score(ref, q_rev)
    mirror_rev, _ = best_shift_score(ref, q_mirror_rev)

    scores = {
        "same": same,
        "mirror_x": mirror,
        "time_reverse": rev,
        "mirror_x_time_reverse": mirror_rev,
    }
    best_family = min(scores, key=scores.get)

    rows.append({
        "seed": seed,
        "L": len(tip),
        "tip_signed_area": signed_area(tip),
        "area_sign": int(np.sign(signed_area(tip))),
        "mean_tip_x": float(np.mean(tip[:,0])),
        "same_score": same,
        "mirror_score": mirror,
        "time_reverse_score": rev,
        "mirror_time_reverse_score": mirror_rev,
        "best_family_vs_ref": best_family,
    })

df = pd.DataFrame(rows).sort_values("seed")
print(f"reference seed = {ref_seed}")
print(df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

print("\nfamily counts:")
print(df["best_family_vs_ref"].value_counts().to_string())
