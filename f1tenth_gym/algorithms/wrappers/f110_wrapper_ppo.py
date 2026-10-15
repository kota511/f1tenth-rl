# wrappers/f110_wrapper_ppo.py
import numpy as np
import gym
from purepursuit import PurePursuitPlanner, nearest_point_on_trajectory

class F110RLWrapperPPO(gym.Wrapper):
    """
    A wrapper to bridge the F1TENTH gym environment with the PPO algorithm.
    PPO controls steering, while speed is controlled by PurePursuit.
    """
    def __init__(self, 
                env, 
                planner, 
                conf, 
                max_steps=5000, 
                num_segments=20,
                max_speed=8.0,
                crash_penalty=-10.0,
                angle_penalty=-1.0,
                alive_bonus=1.0,
                lap_bonus=2.0,
                distance_reward_coef=500,
                steering_consistency_coef=-.0,
                waypoint_path=None):
        super().__init__(env)
        self.env = env
        self.planner = planner
        self.conf = conf
        self.max_steps = max_steps
        self.step_count = 0
        self.obs = None
        self.angle_error = 0.0  # cache angle error for reward
        self.prev_lap_count = 0
        self.prev_progress_distance = 0.0  # initialise progress along track
        self.prev_steering = 0.0  # For steering consistency reward
        
        # Reward parameters
        self.max_speed = max_speed
        self.crash_penalty = crash_penalty
        self.angle_penalty = angle_penalty
        self.alive_bonus = alive_bonus
        self.lap_bonus = lap_bonus
        self.distance_reward_coef = distance_reward_coef
        self.steering_consistency_coef = steering_consistency_coef
        self.track_coverage = 0.0
        
        # If an alternative waypoint path is provided, update the planner
        if waypoint_path is not None:
            self.planner.load_waypoints(waypoint_path)
        
        # Track segmentation for exploration rewards
        self.num_segments = num_segments
        thered_segments = set()
        self.segment_size = self.conf.track_length / num_segments

        # Updated observation space: [x, y, theta, speed, angle_error]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.pi]),
            shape=(5,),
            dtype=np.float32
        )

        # PPO only controls steering (normalised)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            shape=(1,),
            dtype=np.float32
        )

    def reset(self):
        self.step_count = 0
        # Reset steering history
        self.prev_steering = 0.0
        # Reset visited segments for the new episode
        self.visited_segments = set()
        
        initial_pose = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        raw_obs, *_ = self.env.reset(poses=initial_pose)
        self.obs = raw_obs
        processed_obs = self._preprocess_obs(raw_obs)
        
        # Initialise progress tracking based on current position
        wpts = np.vstack((self.planner.waypoints[:, self.conf.wpt_xind],
                          self.planner.waypoints[:, self.conf.wpt_yind])).T
        current_position = np.array([raw_obs['poses_x'][0], raw_obs['poses_y'][0]])
        _, _, t, seg_idx = nearest_point_on_trajectory(current_position, wpts)
        self.prev_progress_distance = seg_idx + t
        return processed_obs

    def step(self, action):
        sim_action = self._preprocess_action(action)
        
        # Handle both Gym and Gymnasium API
        step_result = self.env.step(sim_action)
        if len(step_result) == 5:
            self.obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            self.obs, _, done, info = step_result
            
        self.step_count += 1

        reward = self._compute_reward()

        self.prev_lap_count = self.obs['lap_counts'][0]

        done = done or self.step_count >= self.max_steps
        self.track_coverage = len(self.visited_segments) / self.num_segments
        return self._preprocess_obs(self.obs), reward, done, info

    def _preprocess_obs(self, obs):
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0]
        linear_vel = obs['linear_vels_x'][0]

        # Compute angle error to lookahead point
        position = np.array([pose_x, pose_y])
        lookahead_point = self.planner._get_current_waypoint(
            self.planner.waypoints, self.conf.tlad, position, pose_theta
        )
        if lookahead_point is not None:
            dx = lookahead_point[0] - pose_x
            dy = lookahead_point[1] - pose_y
            desired_heading = np.arctan2(dy, dx)
            self.angle_error = np.arctan2(
                np.sin(desired_heading - pose_theta),
                np.cos(desired_heading - pose_theta)
            )
        else:
            self.angle_error = 0.0  # fallback

        return np.array([pose_x, pose_y, pose_theta, linear_vel, self.angle_error], dtype=np.float32)

    def _preprocess_action(self, action):
        MAX_STEER = 1.0
        # Apply action smoothing to reduce oscillations
        raw_steer = float(action[0]) * MAX_STEER
        smoothed_steer = 0.8 * self.prev_steering + 0.2 * raw_steer  # 80/20 blend
        self.prev_steering = smoothed_steer  # Update for next time

        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]

        speed, _ = self.planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        speed = np.clip(speed, 0.0, self.max_speed)

        return np.array([[smoothed_steer, speed]], dtype=np.float32)

    def _compute_reward(self):
        # Extract current steering for consistency calculation
        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]
        
        # Get the steering value that was applied
        _, current_steering = self.planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)

        # Basic reward components
        speed_reward = 0.1 * self.obs['linear_vels_x'][0]
        crash_penalty = self.crash_penalty if self.obs['collisions'][0] > 0 else 0.0

        # Use squared angle error for smoother penalty
        angle_penalty = self.angle_penalty * (self.angle_error ** 2)
        
        alive_bonus = self.alive_bonus

        # Steering consistency reward
        steering_consistency = self.steering_consistency_coef * abs(self.prev_steering - current_steering)
        self.prev_steering = current_steering

        # Lap Bonus / Penalty Logic
        MIN_VALID_LAP_RATIO = 0.8  # fraction of track to qualify as a valid lap
        lap_bonus = 0.0

        # Distance-based reward
        wpts = np.vstack((self.planner.waypoints[:, self.conf.wpt_xind],
                        self.planner.waypoints[:, self.conf.wpt_yind])).T
        current_position = np.array([self.obs['poses_x'][0], self.obs['poses_y'][0]])
        _, _, t, seg_idx = nearest_point_on_trajectory(current_position, wpts)
        current_progress = seg_idx + t
        lap_increase = self.obs['lap_counts'][0] - self.prev_lap_count
        if lap_increase > 0:
            current_progress += lap_increase * self.conf.track_length

        progress_diff = current_progress - self.prev_progress_distance
        self.prev_progress_distance = current_progress  # Update the previous progress distance

        if lap_increase > 0:
            if progress_diff < MIN_VALID_LAP_RATIO * self.conf.track_length:
                lap_bonus = -self.lap_bonus
            else:
                lap_bonus = self.lap_bonus

        # Calculate current track segment for exploration reward
        current_segment = int((current_progress % self.conf.track_length) / self.segment_size)
        
        # New segment exploration reward
        exploration_reward = 0.0
        if current_segment not in self.visited_segments:
            # Reward for discovering new segment
            exploration_reward = 0.5 + (0.5 * current_segment / self.num_segments)
            self.visited_segments.add(current_segment)

        distance_reward = self.distance_reward_coef * (progress_diff / self.conf.track_length)

        centerline_deviation = abs(self.angle_error)
        center_driving_reward = -1.0 * centerline_deviation**2  # penalty for deviation

        # combine all rewards
        total_reward = (
            speed_reward + 
            alive_bonus + 
            angle_penalty + 
            crash_penalty + 
            lap_bonus + 
            distance_reward +
            steering_consistency +
            exploration_reward +
            center_driving_reward
        )

        # store reward components for logging
        self.last_reward_components = {
            'speed_reward': speed_reward,
            'alive_bonus': alive_bonus,
            'angle_penalty': angle_penalty,
            'crash_penalty': crash_penalty,
            'lap_bonus': lap_bonus,
            'distance_reward': distance_reward,
            'steering_consistency': steering_consistency,
            'exploration_reward': exploration_reward,
            'center_driving_reward': center_driving_reward,
            'total_reward': total_reward
        }

        return total_reward