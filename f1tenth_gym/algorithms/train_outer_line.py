# train_outer_line.py
"""
Train RL models (PPO or SAC) on the outer racing line
"""
import os
import time
import yaml
import numpy as np
import argparse
from argparse import Namespace
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from f110_gym.envs.base_classes import Integrator
from purepursuit import PurePursuitPlanner
import gym
import torch

from wrappers import F110RLWrapperPPO, F110RLWrapperSAC
from reward_callback import RewardTrackingCallback

# Constants for saving models and logs
MODEL_DIR = "models"
LOG_DIR = "logs"
TRACK = "example"
WHEELBASE = 0.17145 + 0.15875

# Reward and environment constants
DEFAULT_MAX_SPEED = 7.0
DEFAULT_CRASH_PENALTY = -10.0
DEFAULT_ANGLE_PENALTY = -1.0
DEFAULT_ALIVE_BONUS = 1.0
DEFAULT_LAP_BONUS = 2.0
DEFAULT_DISTANCE_REWARD_COEF = 500
DEFAULT_STEERING_CONSISTENCY_COEF = -2.0

def make_env(track, algorithm):
    """
    Creates and returns the wrapped F1TENTH environment with specified waypoints.
    """
    # Load YAML config
    config_file = f"../f1tenth_racetracks/{track}/center_config_{track}_map.yaml"
    print(f"Loading config from: {config_file}")

    with open(config_file) as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)
    conf.map_path = f"../f1tenth_racetracks/{track}/{track}_map"

    conf.wpt_path = f"../f1tenth_racetracks/{track}/{track}_centerline_inner.csv"
    conf.wpt_path2 = f"../f1tenth_racetracks/{track}/{track}_centerline_outer.csv"
    
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
    
    # Set up starting pose - use the outer line starting position (sx2, sy2)
    conf.sx = conf.sx2
    conf.sy = conf.sy2
    conf.stheta = conf.stheta2
    
    # Wrap the environment based on the algorithm
    if algorithm.upper() == 'PPO':
        wrapped_env = F110RLWrapperPPO(
            base_env, 
            planner, 
            conf,
            max_speed=DEFAULT_MAX_SPEED,
            crash_penalty=DEFAULT_CRASH_PENALTY,
            angle_penalty=DEFAULT_ANGLE_PENALTY,
            alive_bonus=DEFAULT_ALIVE_BONUS,
            lap_bonus=DEFAULT_LAP_BONUS,
            distance_reward_coef=DEFAULT_DISTANCE_REWARD_COEF,
            steering_consistency_coef=DEFAULT_STEERING_CONSISTENCY_COEF
        )
    elif algorithm.upper() == 'SAC':
        wrapped_env = F110RLWrapperSAC(
            base_env, 
            planner, 
            conf,
            max_speed=DEFAULT_MAX_SPEED,
            crash_penalty=DEFAULT_CRASH_PENALTY,
            angle_penalty=DEFAULT_ANGLE_PENALTY,
            alive_bonus=DEFAULT_ALIVE_BONUS,
            lap_bonus=DEFAULT_LAP_BONUS,
            distance_reward_coef=DEFAULT_DISTANCE_REWARD_COEF,
            steering_consistency_coef=DEFAULT_STEERING_CONSISTENCY_COEF
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'PPO' or 'SAC'")
        
    return wrapped_env

def train_ppo(track, config_path, steps):
    """Train a PPO model with the given configuration"""
    # Create directories if they don't exist
    best_model_dir = f"./PPO_outer_{track}/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Create vectorised environments
    train_env = DummyVecEnv([lambda: make_env(track, 'PPO')])
    eval_env = DummyVecEnv([lambda: make_env(track, 'PPO')])
    
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
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    reward_callback = RewardTrackingCallback(
        log_freq=100, 
        window_size=1000,
        log_dir=LOG_DIR,
        algorithm_name="PPO_outer",
        track_name=track,
    )
    
    # Combine callbacks
    callbacks = [eval_callback, reward_callback]
    
    # Start training
    print(f"\nTraining PPO model on '{track}' track with outer waypoints...")
    print(f"Max speed: {DEFAULT_MAX_SPEED} m/s, Steering consistency: {DEFAULT_STEERING_CONSISTENCY_COEF}")
    
    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"{MODEL_DIR}/ppo_outer_{track}_MS{DEFAULT_MAX_SPEED}_CP{int(DEFAULT_CRASH_PENALTY)}_AP{int(DEFAULT_ANGLE_PENALTY)}_AB{DEFAULT_ALIVE_BONUS}_LB{int(DEFAULT_LAP_BONUS)}_DR{int(DEFAULT_DISTANCE_REWARD_COEF)}_SC{int(DEFAULT_STEERING_CONSISTENCY_COEF)}_TS{timestamp}.zip"
    model.save(model_path)
    print(f"Training complete! Final model saved to: {model_path}")
    
    return model_path

def train_sac(track, config_path, steps):
    """Train a SAC model with the given configuration"""
    # Create directories if they don't exist
    best_model_dir = f"./SAC_outer_{track}/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Create vectorised environments
    train_env = DummyVecEnv([lambda: make_env(track, 'SAC')])
    eval_env = DummyVecEnv([lambda: make_env(track, 'SAC')])
    
    # Define the SAC model
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],  # Smaller actor network to prevent overreacting
            qf=[256, 256]  # Larger critic network for better value estimation
        )
    )
    
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0002,
        buffer_size=500000,
        learning_starts=100,
        batch_size=256,
        ent_coef="auto",  # Automatic entropy tuning
        policy_kwargs=policy_kwargs, 
        tensorboard_log=LOG_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    reward_callback = RewardTrackingCallback(
        log_freq=100, 
        window_size=1000,
        log_dir=LOG_DIR,
        algorithm_name="SAC_outer",
        track_name=track,
    )
    
    # Combine callbacks
    callbacks = [eval_callback, reward_callback]
    
    # Start training
    print(f"\nTraining SAC model on '{track}' track with outer waypoints...")
    print(f"Max speed: {DEFAULT_ALIVE_BONUS} m/s, Steering consistency: {DEFAULT_STEERING_CONSISTENCY_COEF}")
    
    model.learn(
        total_timesteps=steps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"{MODEL_DIR}/sac_outer_{track}_MS{DEFAULT_MAX_SPEED}_CP{int(DEFAULT_CRASH_PENALTY)}_AP{int(DEFAULT_ANGLE_PENALTY)}_AB{DEFAULT_ALIVE_BONUS}_LB{int(DEFAULT_LAP_BONUS)}_DR{int(DEFAULT_DISTANCE_REWARD_COEF)}_SC{int(DEFAULT_STEERING_CONSISTENCY_COEF)}_TS{timestamp}.zip"
    model.save(model_path)
    print(f"Training complete! Final model saved to: {model_path}")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on outer racing line")
    parser.add_argument("algorithm", choices=["PPO", "SAC"], help="RL algorithm to train (PPO or SAC)")
    parser.add_argument("--steps", type=int, default=2_000_000, help="Training steps (default: 3,000,000)")
    args = parser.parse_args()

    config_path = f"../f1tenth_racetracks/{TRACK}/center_config_{TRACK}_map.yaml"

    if args.algorithm == "PPO":
        train_ppo(TRACK, config_path, args.steps)
    elif args.algorithm == "SAC":
        train_sac(TRACK, config_path, args.steps)

    print("\n Training finished.")


if __name__ == "__main__":
    main()