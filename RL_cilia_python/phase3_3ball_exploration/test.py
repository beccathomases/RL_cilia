import numpy as np

from cilia_3_ball_env import Cilia3BallEnv

env = Cilia3BallEnv(precompute=False)
s, _ = env.reset()
print("reset:", s)
for _ in range(3):
    s, r, term, trunc, info = env.step(env.action_space.sample())
    print("step:", s, r, info)

env = Cilia3BallEnv(precompute=True)
print(env.flux_table.shape)
print(env.next_state_table.shape)

print("reward min:", np.min(env.flux_table))
print("reward max:", np.max(env.flux_table))
print("reward mean abs:", np.mean(np.abs(env.flux_table)))

idx = np.unravel_index(np.argmax(np.abs(env.flux_table)), env.flux_table.shape)
print("argmax abs reward index:", idx)
print("max abs reward:", env.flux_table[idx])