import numpy as np
from scipy.io import loadmat


# ============================================================
# USER SETTINGS
# ============================================================

# MATLAB cycle file
mat_file = "cycle4_run_1_g0.99_eps00.75_alp00.99_nEpisode1000.mat"

# Python file: could be VI, tabular-Q, PPO-exported cycle, etc.
py_file = "value_iteration_cilia_2_ball_clip_penalty_bins11x21_g0.990.npy"

# If py_file contains many cycles, pick which one
py_cycle_id = 0

# If py_file is a tabular-Q all-seeds file, pick the seed here
tabq_seed = 1

# Set this depending on what the Python file is
py_file_type = "vi"   # options: "vi", "tabq", "generic_cycles"


# ============================================================
# HELPERS
# ============================================================

def canonical_cycle(cycle):
    """
    Canonical representation invariant under cyclic rotation.
    """
    cyc = [tuple(map(int, s)) for s in cycle]
    n = len(cyc)
    if n == 0:
        return tuple()
    rots = [tuple(cyc[k:] + cyc[:k]) for k in range(n)]
    return min(rots)


def cycles_match_up_to_shift(cycle_a, cycle_b):
    return canonical_cycle(cycle_a) == canonical_cycle(cycle_b)


def print_cycle_summary(name, cycle):
    print(f"\n{name}")
    print(f"  length = {len(cycle)}")
    print(f"  cycle  = {cycle}")


def matlab_cycle_from_matfile(mat_file):
    """
    Assumes MATLAB file contains cycle_states as produced in your earlier scripts.
    """
    M = loadmat(mat_file)

    if "cycle_states" not in M:
        raise ValueError(f"{mat_file} does not contain variable 'cycle_states'.")

    raw = np.array(M["cycle_states"]).squeeze()

    # convert to python list of tuples
    cycle = [int(x) for x in raw.tolist()]

    # MATLAB files often save repeated first state at the end.
    if len(cycle) >= 2 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    return cycle


def python_cycle_from_file(py_file, py_file_type="vi", py_cycle_id=0, tabq_seed=1):
    data = np.load(py_file, allow_pickle=True).item()

    if py_file_type == "vi":
        cyc = data["cycles"][py_cycle_id]
        cycle = cyc["cycle"]
        return [tuple(map(int, s)) for s in cycle]

    elif py_file_type == "generic_cycles":
        cyc = data["cycles"][py_cycle_id]
        cycle = cyc["cycle"]
        return [tuple(map(int, s)) for s in cycle]

    elif py_file_type == "tabq":
        results = list(data["results"])
        match = None
        for r in results:
            if int(r["seed"]) == tabq_seed:
                match = r
                break
        if match is None:
            raise ValueError(f"Seed {tabq_seed} not found in {py_file}")

        cyc = list(match["cycles"])[py_cycle_id]
        cycle = cyc["cycle"]
        return [tuple(map(int, s)) for s in cycle]

    else:
        raise ValueError(f"Unknown py_file_type: {py_file_type}")


# ============================================================
# LOAD MATLAB CYCLE
# ============================================================

mat_cycle_linear = matlab_cycle_from_matfile(mat_file)

# MATLAB state numbers are linear indices; Python uses (i,j) pairs.
# If your MATLAB states are already stored as pairs, adjust this block.
#
# For your recent Python env, n_bins = [11, 21], so states are 11 x 21.
n0 = 11
n1 = 21

def matlab_linear_to_pair(s, n0, n1):
    """
    Convert MATLAB-style 1-based linear state index to Python (i,j) pair,
    assuming MATLAB column-major indexing on an n0 x n1 array.
    """
    s0 = int(s) - 1   # convert to 0-based
    i = s0 % n0
    j = s0 // n0
    return (i, j)

mat_cycle = [matlab_linear_to_pair(s, n0, n1) for s in mat_cycle_linear]


# ============================================================
# LOAD PYTHON CYCLE
# ============================================================

py_cycle = python_cycle_from_file(
    py_file,
    py_file_type=py_file_type,
    py_cycle_id=py_cycle_id,
    tabq_seed=tabq_seed,
)


# ============================================================
# COMPARE
# ============================================================

print_cycle_summary("MATLAB cycle", mat_cycle)
print_cycle_summary("Python cycle", py_cycle)

same = cycles_match_up_to_shift(mat_cycle, py_cycle)

print("\nComparison:")
print("  same up to cyclic shift?", same)

if not same:
    print("\nCanonical MATLAB cycle:")
    print(canonical_cycle(mat_cycle))
    print("\nCanonical Python cycle:")
    print(canonical_cycle(py_cycle))

    # Optional quick mismatch report
    cmat = canonical_cycle(mat_cycle)
    cpy = canonical_cycle(py_cycle)
    if len(cmat) == len(cpy):
        print("\nFirst mismatching entries:")
        for k, (a, b) in enumerate(zip(cmat, cpy)):
            if a != b:
                print(f"  k={k}: MATLAB {a}, Python {b}")
    else:
        print("\nCycle lengths differ, so they cannot match exactly.")