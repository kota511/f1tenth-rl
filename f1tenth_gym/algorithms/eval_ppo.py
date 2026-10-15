# eval_ppo.py
from datetime import datetime
import gym
import yaml
import numpy as np
import time
import csv
import os
from argparse import Namespace, ArgumentParser
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from purepursuit import PurePursuitPlanner
from f110_gym.envs.base_classes import Integrator

from wrappers import F110RLWrapperPPO

# constants should be same as trained files
TRACK = "example"
WHEELBASE = 0.17145 + 0.15875

MAX_SPEED = 8.0
CRASH_PENALTY = -10.0
ANGLE_PENALTY = -1.0
ALIVE_BONUS = 1.0
LAP_BONUS = 2.0
DISTANCE_REWARD_COEF = 500
STEERING_CONSISTENCY_COEF = -3.0

def make_env(track_name, enable_render=False):
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


# Main Evaluation Loop
def evaluate(model_path, track, n_episodes=10, render=False, save_csv=True):
    env = DummyVecEnv([lambda: make_env(track)])
    model = PPO.load(model_path)
    results = []
    print(f"\nStarting PPO Evaluation on track '{track}' for {n_episodes} episodes...\n")
    for ep in range(1, n_episodes + 1):
        obs, done = env.reset(), False
        total_reward, max_speed, lap_time, steps = 0.0, 0.0, 0.0, 0
        crashed = False
        start_time = time.time()
        prev_lap_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            raw_obs = env.envs[0].obs
            current_speed = float(raw_obs['linear_vels_x'][0])
            max_speed = max(max_speed, current_speed)
            current_lap_count = int(raw_obs['lap_counts'][0])
            
            # Only update lap_time if a lap is actually completed
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
            "total_reward": total_reward,
            "max_speed": max_speed,
            "crashed": crashed,
            "steps": steps,
            "track_coverage": track_coverage,
            "wall_time": round(elapsed, 2),
        })
        print(f"Episode {ep:2d}: Lap-time: {lap_time:.2f}s | Reward: {total_reward:.1f} | Crash: {crashed} | Real-time: {elapsed:.2f}s")

    # Summary
    lap_times = [r["lap_time"] for r in results]
    rewards = [r["total_reward"] for r in results]
    crash_rate = sum(r["crashed"] for r in results) / n_episodes
    print("\nPPO Evaluation Summary:")
    print(f"Avg Lap Time   : {np.mean(lap_times):.2f} ± {np.std(lap_times):.2f} sec")
    print(f"Avg Reward     : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Crash Rate     : {crash_rate * 100:.1f}%")
    print(f"Max Speed (any): {max(r['max_speed'] for r in results):.2f} m/s")

    # save CSV
    if save_csv:
        os.makedirs("eval_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"eval_results/eval_ppo_{track}_MS{MAX_SPEED}_CP{int(CRASH_PENALTY)}_AP{int(ANGLE_PENALTY)}_AB{ALIVE_BONUS}_LB{int(LAP_BONUS)}_DR{int(DISTANCE_REWARD_COEF)}_SC{int(STEERING_CONSISTENCY_COEF)}_TS{timestamp}.csv"
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📁 Results saved to: {csv_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--track", type=str, default="example", help="Track name (e.g. 'example')")
    parser.add_argument("--model", type=str, default=None, help="Path to PPO model zip")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render simulation")
    args = parser.parse_args()
    if args.model is None:
        # args.model = f"PPO_train3_example/best_model"
        args.model = f"./models/ppo_model_BASELINE.zip"
        # args.model = f"./models/ppo_model_MS8.0_CP-10_AP-1_AB1.0_LB2_DR0_SC-3_TS2025-04-15_05-08-53.zip"
    evaluate(args.model, track=args.track, n_episodes=args.episodes, render=args.render)