import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from cilia_2_ball_env import Cilia2BallEnv

# ===== 1. Create and validate env =====
env=Cilia2BallEnv(max_steps=500,precompute=True)
check_env(env)

env = Monitor(env)


# ===== 2. PPO model =====
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=4096,          # larger rollout = more stable
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,         # ↑ important to avoid early collapse into cycles
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)


# ===== 3. Evaluation callback =====
eval_env = Monitor(Cilia2BallEnv(max_steps=500,precompute=True))

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)


# ===== 4. Train =====
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)


# ===== 5. Save =====
model.save("ppo_model")