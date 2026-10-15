# run_multi_agent.py
import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv
from argparse import ArgumentParser, Namespace
import gym
from stable_baselines3 import PPO, SAC
from f110_gym.envs.base_classes import Integrator
from purepursuit import PurePursuitPlanner

from wrappers.multi_agent_wrapper import MultiAgentF110WrapperPPO, MultiAgentF110WrapperSAC


TRACK = "example"
WHEELBASE = 0.17145 + 0.15875

INNER_MAX_SPEED = 6.0
OUTER_MAX_SPEED = 7.0
CRASH_PENALTY = -10.0
ANGLE_PENALTY = -1.0
ALIVE_BONUS = 1.0
LAP_BONUS = 2.0
DISTANCE_REWARD_COEF = 500  # Coefficient for distance reward
STEERING_CONSISTENCY_COEF = -2.0  # Coefficient for steering consistency reward
def create_multi_agent_env(
        track_name,
        rl_model_path,
        model_type,
        debug_mode=False
    ):
    """
    Create a multi-agent environment with an RL agent and PurePursuit
    """
    # Load YAML config
    config_file = f"../f1tenth_racetracks/{TRACK}/center_config_{TRACK}_map.yaml"
    print(f"Loading config from: {config_file}")

    with open(config_file) as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)
    conf.map_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_map"
    conf.wpt_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_centerline_inner.csv"
    conf.wpt_path2 = f"../f1tenth_racetracks/{TRACK}/{TRACK}_centerline_outer.csv"
    # Print debug info
    if debug_mode:
        print(f"Creating environment with:")
        print(f"  Track: {track_name}")
        print(f"  Map path: {conf.map_path}")
        print(f"  Map extension: {conf.map_ext}")
        print(f"  Inner waypoints: {conf.wpt_path}")
        print(f"  Outer waypoints: {conf.wpt_path2}")
    
    # Create base F1TENTH gym environment with 2 agents
    
    base_env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=2,  # Two agents
        timestep=0.01,
        integrator=Integrator.RK4
    )
    
    # Set up inner planner (for PurePursuit)
    inner_planner = PurePursuitPlanner(conf, WHEELBASE)
    if conf.wpt_path:
        inner_planner.load_waypoints(conf.wpt_path)
            
    # Set up outer planner (for observation processing for RL)
    outer_planner = PurePursuitPlanner(conf, WHEELBASE)
    if conf.wpt_path2:
        outer_planner.load_waypoints(conf.wpt_path2)
    
    # Add render callback to show waypoints
    def render_callback(env_renderer):
        inner_planner.render_waypoints(env_renderer)
        outer_planner.render_waypoints(env_renderer)
    
    base_env.add_render_callback(render_callback)
    
    # Load the RL model
    if model_type.upper() == "PPO":
        rl_model = PPO.load(rl_model_path)
        # Create the PPO-specific multi-agent wrapper
        wrapped_env = MultiAgentF110WrapperPPO(
            env=base_env,
            rl_model=rl_model,
            conf=conf,
            inner_planner=inner_planner,
            outer_planner=outer_planner,
            inner_max_speed=INNER_MAX_SPEED,
            outer_max_speed=OUTER_MAX_SPEED
        )
        if debug_mode:
            print("Using PPO-specific wrapper")
            print(f"Observation space: {wrapped_env.observation_space}")
            print(f"Action space: {wrapped_env.action_space}")
    
    elif model_type.upper() == "SAC":
        rl_model = SAC.load(rl_model_path)
        # Create the SAC-specific multi-agent wrapper
        wrapped_env = MultiAgentF110WrapperSAC(
            env=base_env,
            rl_model=rl_model,
            conf=conf,
            inner_planner=inner_planner,
            outer_planner=outer_planner,
            inner_max_speed=INNER_MAX_SPEED,
            outer_max_speed=OUTER_MAX_SPEED
        )
        if debug_mode:
            print("Using SAC-specific wrapper")
            print(f"Observation space: {wrapped_env.observation_space}")
            print(f"Action space: {wrapped_env.action_space}")
            print(f"Observation normalization - Means: {wrapped_env.obs_means}")
            print(f"Observation normalization - Stds: {wrapped_env.obs_stds}")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'PPO' or 'SAC'")
    
    return wrapped_env

def run_multi_agent_comparison(
    track_name,
    rl_model_path,
    model_type,
    n_episodes=1,
    render=True,
    save_csv=True,
    debug_mode=False
):
    """
    Run a multi-agent comparison between RL agent and PurePursuit.
    """
    # Create the multi-agent environment
    env = create_multi_agent_env(
        track_name=track_name,
        rl_model_path=rl_model_path,
        model_type=model_type,
        debug_mode=debug_mode
    )
    
    results = []
    print(f"\nStarting Multi-Agent Comparison on track '{track_name}' for {n_episodes} episodes...")
    
    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        
        # Initialize per-episode stats
        rl_total_reward = 0.0
        rl_max_speed = 0.0
        pp_max_speed = 0.0
        rl_lap_time = 0.0
        pp_lap_time = 0.0
        rl_crashed = False
        pp_crashed = False
        steps = 0
        
        # Debug initial observation
        if debug_mode and ep == 1:
            print("\nInitial Observation:")
            print(f"Shape: {obs.shape}")
            print(f"Values: {obs}")
        
        start_time = time.time()
        
        # Collect trajectory data for debugging if needed
        trajectory = []
        
        while not done:
            # Get action from the RL model
            action, _ = env.rl_model.predict(obs, deterministic=True)
            
            # Debug action
            if debug_mode and ep == 1 and steps < 10:
                print(f"\nStep {steps}:")
                print(f"RL Action: {action}")
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            rl_total_reward += float(reward)
            steps += 1
            
            # Debug observation, reward and state
            if debug_mode and ep == 1 and steps < 10:
                print(f"Observation after step: {obs}")
                print(f"Reward: {reward}")
                print(f"RL Position: ({env.obs['poses_x'][0]:.2f}, {env.obs['poses_y'][0]:.2f})")
                print(f"RL Angle Error: {env.angle_error:.4f}")
                print(f"RL Speed: {env.obs['linear_vels_x'][0]:.2f}")
                print(f"RL Steering: {env.prev_steering_rl:.4f}")
            
            # Collect trajectory data
            if debug_mode:
                trajectory.append({
                    'step': steps,
                    'x': env.obs['poses_x'][0],
                    'y': env.obs['poses_y'][0],
                    'theta': env.obs['poses_theta'][0],
                    'speed': env.obs['linear_vels_x'][0],
                    'steering': env.prev_steering_rl,
                    'angle_error': env.angle_error,
                    'reward': reward
                })
            
            # Update stats
            rl_speed = float(env.obs['linear_vels_x'][0])
            pp_speed = float(env.obs['linear_vels_x'][1])
            rl_max_speed = max(rl_max_speed, rl_speed)
            pp_max_speed = max(pp_max_speed, pp_speed)
            
            # Check for collisions
            if env.obs['collisions'][0] > 0 and not rl_crashed:
                rl_crashed = True
                if debug_mode:
                    print(f"\n!!! RL agent crashed at step {steps} !!!")
                    print(f"Position: ({env.obs['poses_x'][0]:.2f}, {env.obs['poses_y'][0]:.2f})")
                    print(f"Orientation: {env.obs['poses_theta'][0]:.4f}")
                    print(f"Speed: {env.obs['linear_vels_x'][0]:.2f}")
                    print(f"Steering: {env.prev_steering_rl:.4f}")
                    print(f"Angle Error: {env.angle_error:.4f}")
                    print(f"Last 3 actions:")
                    for i in range(max(0, len(trajectory)-3), len(trajectory)):
                        print(f"  Step {trajectory[i]['step']}: steering={trajectory[i]['steering']:.4f}, speed={trajectory[i]['speed']:.2f}")
            
            if env.obs['collisions'][1] > 0 and not pp_crashed:
                pp_crashed = True
                if debug_mode:
                    print(f"\n!!! PurePursuit crashed at step {steps} !!!")
            
            # Define finish line from start pose (only once per episode)
            if steps == 1:
                rl_start = (env.conf.sx, env.conf.sy)
                pp_start = (env.conf.sx2, env.conf.sy2)
                rl_finished = False
                pp_finished = False
                rl_start_time = time.time()
                pp_start_time = rl_start_time
                rl_lap_time = None
                pp_lap_time = None
                finish_threshold = 0.3  # meters

            # Check if agents finished a lap (distance from start, after a delay)
            if not rl_finished:
                dx = env.obs['poses_x'][0] - rl_start[0]
                dy = env.obs['poses_y'][0] - rl_start[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < finish_threshold and steps > 100:  # Wait a few steps to avoid early loop closure
                    rl_lap_time = time.time() - rl_start_time
                    rl_finished = True

            if not pp_finished:
                dx = env.obs['poses_x'][1] - pp_start[0]
                dy = env.obs['poses_y'][1] - pp_start[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < finish_threshold and steps > 100:
                    pp_lap_time = time.time() - pp_start_time
                    pp_finished = True
            
            # Render if requested
            if render:
                env.render()
        
        elapsed = time.time() - start_time
        
        # Save trajectory data for debugging
        if debug_mode and ep == 1 and save_csv:
            os.makedirs("debug_data", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            traj_path = f"debug_data/trajectory_{track_name}_{model_type}_{timestamp}.csv"
            with open(traj_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=trajectory[0].keys())
                writer.writeheader()
                writer.writerows(trajectory)
            print(f"\nTrajectory data saved to: {traj_path}")
        
        # Record results for this episode
        episode_result = {
            "episode": ep,
            "rl_lap_time": rl_lap_time if rl_lap_time is not None else -1,
            "pp_lap_time": pp_lap_time if pp_lap_time is not None else -1,
            "rl_total_reward": rl_total_reward,
            "rl_max_speed": rl_max_speed,
            "pp_max_speed": pp_max_speed,
            "rl_crashed": rl_crashed,
            "pp_crashed": pp_crashed,
            "steps": steps,
            "wall_time": round(elapsed, 2),
        }
        results.append(episode_result)
        rl_time_str = f"{rl_lap_time:.2f}s" if rl_lap_time is not None else "DNF"
        pp_time_str = f"{pp_lap_time:.2f}s" if pp_lap_time is not None else "DNF"

        # Print episode summary
        print(f"Episode {ep}:")
        print(f"  RL ({model_type}): Lap time: {rl_time_str} | Reward: {rl_total_reward:.1f} | Max speed: {rl_max_speed:.2f} m/s | Crashed: {rl_crashed}")
        print(f"  PurePursuit: Lap time: {pp_time_str} | Max speed: {pp_max_speed:.2f} m/s | Crashed: {pp_crashed}")
        print(f"  Steps: {steps} | Real-time: {elapsed:.2f}s")
    
    # Calculate summary statistics
    rl_lap_times = [r["rl_lap_time"] for r in results if r["rl_lap_time"] > 0]
    pp_lap_times = [r["pp_lap_time"] for r in results if r["pp_lap_time"] > 0]
    rl_rewards = [r["rl_total_reward"] for r in results]
    rl_crash_rate = sum(r["rl_crashed"] for r in results) / n_episodes
    pp_crash_rate = sum(r["pp_crashed"] for r in results) / n_episodes
    
    # Print overall summary
    print("\nMulti-Agent Comparison Summary:")
    print(f"RL Agent ({model_type}):")
    if rl_lap_times:
        print(f"  Avg Lap Time: {np.mean(rl_lap_times):.2f} ± {np.std(rl_lap_times):.2f} sec")
    else:
        print("  No completed laps")
    print(f"  Avg Reward: {np.mean(rl_rewards):.2f} ± {np.std(rl_rewards):.2f}")
    print(f"  Crash Rate: {rl_crash_rate * 100:.1f}%")
    print(f"  Max Speed: {max(r['rl_max_speed'] for r in results):.2f} m/s")
    
    print("\nPurePursuit:")
    if pp_lap_times:
        print(f"  Avg Lap Time: {np.mean(pp_lap_times):.2f} ± {np.std(pp_lap_times):.2f} sec")
    else:
        print("  No completed laps")
    print(f"  Crash Rate: {pp_crash_rate * 100:.1f}%")
    print(f"  Max Speed: {max(r['pp_max_speed'] for r in results):.2f} m/s")
    
    # Winner determination
    if rl_lap_times and pp_lap_times:
        rl_best = min(rl_lap_times)
        pp_best = min(pp_lap_times)
        winner = f"{model_type}" if rl_best < pp_best else "PurePursuit"
        time_diff = abs(rl_best - pp_best)
        print(f"\nBest lap times - {model_type}: {rl_best:.2f}s, PurePursuit: {pp_best:.2f}s")
        print(f"Winner: {winner} by {time_diff:.2f} seconds")
    
    # Save results to CSV if requested
    if save_csv:
        os.makedirs("multi_agent_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"multi_agent_results/comparison_{track_name}_{model_type}_{timestamp}.csv"
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Run a multi-agent comparison between RL and PurePursuit")
    parser.add_argument("--track", type=str, default="example", help="Track name")
    parser.add_argument("--model", type=str, help="Path to RL model")
    # parser.add_argument("--model", type=str, default=f"./models/ppo_outer_example_2025-04-16_08-17-46.zip", help="Path to RL model"))
    parser.add_argument("--model_type", type=str, choices=["PPO", "SAC"], required=True, help="Type of RL model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    parser.add_argument("--no_save", action="store_true", help="Don't save results to CSV")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.model_type.upper() == "PPO":
        args.model = f"./models/ppo_outer_example_MS7.0_CP-10_AP-1_AB1.0_LB2_DR500_SC-2_TS2025-04-16_20-23-16.zip"
    elif args.model_type.upper() == "SAC":
        args.model = f"./models/sac_outer_example_MS7.0_CP-10_AP-1_AB1.0_LB2_DR700_SC-2.zip"
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}. Must be 'PPO' or 'SAC'")
    
    run_multi_agent_comparison(
        track_name=args.track,
        rl_model_path=args.model,
        model_type=args.model_type,
        n_episodes=args.episodes,
        render=not args.no_render,
        save_csv=not args.no_save,
        debug_mode=args.debug
    )