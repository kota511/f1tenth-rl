# eval_sac.py

import gym
import yaml
import numpy as np
import time
import csv
import os
from argparse import Namespace, ArgumentParser
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from purepursuit import PurePursuitPlanner
from f110_gym.envs.base_classes import Integrator

from wrappers import F110RLWrapperSAC

# constants should match what was used in training files
WHEELBASE = 0.17145 + 0.15875

MAX_SPEED = 8.0
CRASH_PENALTY = 0
ANGLE_PENALTY = -1.0
ALIVE_BONUS = 1.0
LAP_BONUS = 2.0
DISTANCE_REWARD_COEF = 500
STEERING_CONSISTENCY_COEF = -3.0

def make_env(track_name):
    """Create and return the wrapped environment for the given track"""
    with open(f"../f1tenth_racetracks/{track_name}/config_{track_name}_map.yaml") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)
    conf.map_path = f"../f1tenth_racetracks/{track_name}/{track_name}_map"
    conf.wpt_path = f"../f1tenth_racetracks/{track_name}/{track_name}_waypoints.csv"
    
    base_env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    
    planner = PurePursuitPlanner(conf, WHEELBASE)
    
    # Use the same wrapper and reward parameters as in training
    wrapped_env = F110RLWrapperSAC(
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

def evaluate(model_path, track, n_episodes=10, render=False, save_csv=True):
    env = DummyVecEnv([lambda: make_env(track)])
    model = SAC.load(model_path)

    results = []
    print(f"\nStarting SAC Evaluation on track '{track}' for {n_episodes} episodes...\n")
    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        max_speed = 0.0
        lap_time = 0.0
        crashed = False
        steps = 0
        prev_lap_count = 0
        start_time = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            raw_obs = env.envs[0].obs
            current_speed = float(raw_obs['linear_vels_x'][0])
            max_speed = max(max_speed, current_speed)
            current_lap_count = int(raw_obs['lap_counts'][0])
            if current_lap_count > prev_lap_count:
                lap_time = float(raw_obs['lap_times'][0])
                prev_lap_count = current_lap_count
            if raw_obs['collisions'][0] > 0:
                crashed = True

            if render:
                env.envs[0].env.render()

        elapsed = time.time() - start_time
        track_coverage = env.envs[0].track_coverage
        results.append({
            "episode": ep,
            "lap_time": lap_time,
            "total_reward": float(total_reward),
            "max_speed": max_speed,
            "crashed": crashed,
            "steps": steps,
            "track_coverage": track_coverage,
            "wall_time": round(elapsed, 2),
        })

        print(f"Episode {ep:2d}: Lap-time: {lap_time:.2f}s | Reward: {total_reward:.1f} | Crash: {crashed} | Real-time: {elapsed:.2f}s")

    lap_times = [r["lap_time"] for r in results]
    rewards = [r["total_reward"] for r in results]
    crash_rate = sum(r["crashed"] for r in results) / n_episodes

    print("\nSAC Evaluation Summary:")
    print(f"Avg Lap Time   : {np.mean(lap_times):.2f} ± {np.std(lap_times):.2f} sec")
    print(f"Avg Reward     : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Crash Rate     : {crash_rate * 100:.1f}%")
    print(f"Max Speed (any): {max(r['max_speed'] for r in results):.2f} m/s")

    if save_csv:
        os.makedirs("eval_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"eval_results/eval_sac_{track}_MS{MAX_SPEED}_CP{int(CRASH_PENALTY)}_AP{int(ANGLE_PENALTY)}_AB{ALIVE_BONUS}_LB{int(LAP_BONUS)}_DR{int(DISTANCE_REWARD_COEF)}_SC{int(STEERING_CONSISTENCY_COEF)}.csv"
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--track", type=str, default="example", help="Track name")
    parser.add_argument("--model", type=str, default=None, help="Path to SAC model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render simulation")
    args = parser.parse_args()

    if args.model is None:
        # args.model = f"./SAC_train2_{args.track}/best_model"
        args.model = f"models/sac_model_BASELINE.zip"
        # args.model = f"models/sac_model_MS8.0_CP-10_AP-1_AB1.0_LB2_DR500_SC0_TS2025-04-12_13-40-54.zip"

    evaluate(args.model, track=args.track, n_episodes=args.episodes, render=args.render)