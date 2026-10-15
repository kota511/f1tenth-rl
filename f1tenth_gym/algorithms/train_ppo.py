# train_ppo.py
import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from f110_gym.envs.base_classes import Integrator
from purepursuit import PurePursuitPlanner
import gym
import torch

from wrappers import F110RLWrapperPPO
from reward_callback import RewardTrackingCallback

# constants for saving models and logs
MODEL_DIR = "models"
LOG_DIR = "logs"
TRACK = "example"
BEST_MODEL_DIR = f"./PPO_train_{TRACK}/"
WHEELBASE = 0.17145 + 0.15875

# reward/environment constants
MAX_SPEED = 8.0
CRASH_PENALTY = -10.0
ANGLE_PENALTY = -1.0
ALIVE_BONUS = 1.0
LAP_BONUS = 2.0
DISTANCE_REWARD_COEF = 500
STEERING_CONSISTENCY_COEF = -3.0

# Make directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

def make_env():
    """
    Creates and returns the wrapped F1TENTH environment with PurePursuit planner.
    """
    # Load YAML config
    with open(f"../f1tenth_racetracks/{TRACK}/config_{TRACK}_map.yaml") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)
    conf.map_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_map"
    conf.wpt_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_waypoints.csv"

    # Create base F1TENTH gym environment
    base_env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4
    )

    # Instantiate the PurePursuit planner
    planner = PurePursuitPlanner(conf, WHEELBASE)

    # wrap the environment to interface with PPO using our wrapper with reward parameters
    wrapped_env = F110RLWrapperPPO(
        base_env, 
        planner, 
        conf,
        max_speed=MAX_SPEED,
        crash_penalty=CRASH_PENALTY,
        angle_penalty=ANGLE_PENALTY,
        alive_bonus=ALIVE_BONUS,
        lap_bonus=LAP_BONUS,
        distance_reward_coef=DISTANCE_REWARD_COEF,
        steering_consistency_coef=STEERING_CONSISTENCY_COEF
    )
    return wrapped_env


if __name__ == "__main__":
    # Create vectorised environments for PPO training and evaluation
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Define the PPO model
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],  # Smaller actor network to prevent overreacting
            vf=[256, 256]  # Larger critic network for better value estimation
        )
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0002,  # Slightly lower learning rate for stability
        batch_size=256,        # Larger batch size
        n_steps=1024,          # Slightly shorter horizon
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Callback to log and save the best model during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Create the reward tracking callback
    reward_callback = RewardTrackingCallback(
        log_freq=100, 
        window_size=1000,
        log_dir=LOG_DIR,
        algorithm_name="PPO",
        track_name=TRACK,
    )
    
    # Combine callbacks
    callbacks = [eval_callback, reward_callback]

    # Start training
    model.learn(
        total_timesteps=1_500_000,
        callback=callbacks,
        progress_bar=True
    )

    # Save final model
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model.save(f"{MODEL_DIR}/ppo_model_MS{MAX_SPEED}_CP{int(CRASH_PENALTY)}_AP{int(ANGLE_PENALTY)}_AB{ALIVE_BONUS}_LB{int(LAP_BONUS)}_DR{int(DISTANCE_REWARD_COEF)}_SC{int(STEERING_CONSISTENCY_COEF)}_TS{timestamp}.zip")