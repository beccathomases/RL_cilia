import numpy as np

R = np.load("gamma_sweep_value_iteration_results.npy", allow_pickle=True).item()
print(type(R))
print("gamma keys:", R.keys())

g = sorted(R.keys())[0]
print("\nexample gamma:", g)
print("entry type:", type(R[g]))
print("entry keys:", R[g].keys())

print("\ncycles type:", type(R[g]["cycles"]))
print("cycles value:", R[g]["cycles"])

if len(R[g]["cycles"]) > 0:
    c0 = R[g]["cycles"][0]
    print("\nfirst cycle type:", type(c0))
    print("first cycle:", c0)
    if isinstance(c0, dict):
        print("first cycle keys:", c0.keys())