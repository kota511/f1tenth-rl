# wrappers/multi_agent_wrapper.py
import numpy as np
import gym
from purepursuit import PurePursuitPlanner, nearest_point_on_trajectory

class MultiAgentF110WrapperPPO(gym.Wrapper):
    """
    A wrapper for multi-agent comparison between PPO agents and PurePursuit.
    This wrapper exactly matches the F110RLWrapperPPO observation space, action space,
    and reward computation to ensure compatibility.
    """
    def __init__(self, 
                env, 
                rl_model,
                conf,
                inner_planner,
                outer_planner,
                max_steps=5000,
                num_segments=20,
                inner_max_speed=6.0,
                outer_max_speed=7.0,
                crash_penalty=-10.0,
                angle_penalty=-1.0,
                alive_bonus=1.0,
                lap_bonus=2.0,
                distance_reward_coef=500,
                steering_consistency_coef=-2.0):
        super().__init__(env)
        self.env = env
        self.rl_model = rl_model
        self.conf = conf
        self.max_steps = max_steps
        self.step_count = 0
        self.obs = None
        
        # Speed parameters
        self.inner_max_speed = inner_max_speed
        self.outer_max_speed = outer_max_speed
        
        # Set up the planners
        self.inner_planner = inner_planner  # For the PurePursuit agent
        self.outer_planner = outer_planner  # For the RL agent
        
        # Tracking lap counts
        self.rl_lap_count = 0
        self.pp_lap_count = 0
        
        self.angle_error = 0.0  # cache angle error for reward
        self.prev_lap_count = 0
        self.prev_progress_distance = 0.0  # initialise progress along track
        self.prev_steering = 0.0  # For steering consistency reward
        
        # Reward parameters
        self.max_speed = outer_max_speed
        self.crash_penalty = crash_penalty
        self.angle_penalty = angle_penalty
        self.alive_bonus = alive_bonus
        self.lap_bonus = lap_bonus
        self.distance_reward_coef = distance_reward_coef
        self.steering_consistency_coef = steering_consistency_coef
        
        # Track segmentation for exploration rewards
        self.num_segments = num_segments
        self.visited_segments = set()
        self.segment_size = self.conf.track_length / num_segments
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.pi]),
            shape=(5,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            shape=(1,),
            dtype=np.float32
        )

    def reset(self):
        self.step_count = 0

        self.prev_steering = 0.0

        self.visited_segments = set()
        self.rl_lap_count = 0
        self.pp_lap_count = 0
        self.prev_lap_count = 0
        
        # Reset environment with two agents
        # Agent 0: PPO agent (outer line)
        # Agent 1: PurePursuit (inner line)

        initial_poses = np.array([
            [self.conf.sx2, self.conf.sy2, self.conf.stheta2],  # PPO agent - outer line
            [self.conf.sx, self.conf.sy, self.conf.stheta]      # PurePursuit - inner line
        ])
        
        raw_obs, *_ = self.env.reset(poses=initial_poses)
        self.obs = raw_obs
        
        # Initialise progress tracking based on current position
        wpts = np.vstack((self.outer_planner.waypoints[:, self.conf.wpt_xind],
                          self.outer_planner.waypoints[:, self.conf.wpt_yind])).T
        current_position = np.array([raw_obs['poses_x'][0], raw_obs['poses_y'][0]])
        _, _, t, seg_idx = nearest_point_on_trajectory(current_position, wpts)
        self.prev_progress_distance = seg_idx + t
        
        # Process observation for RL agent
        processed_obs = self._preprocess_obs(raw_obs)
        return processed_obs

    def step(self, rl_action):
        # Convert RL action to simulation action
        rl_sim_action = self._preprocess_action(rl_action)
        
        # Generate PurePursuit action
        pp_sim_action = self._generate_purepursuit_action()
        
        # Combine actions for both agents
        combined_action = np.vstack((rl_sim_action, pp_sim_action))
        
        # Step the environment
        step_result = self.env.step(combined_action)
        if len(step_result) == 5:
            self.obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            self.obs, _, done, info = step_result
        
        self.step_count += 1
        
        # Track lap counts
        if self.obs['lap_counts'][0] > self.rl_lap_count:
            self.rl_lap_count = self.obs['lap_counts'][0]
            print(f"RL agent completed lap {self.rl_lap_count} in {self.obs['lap_times'][0]:.2f}s")
            
        if self.obs['lap_counts'][1] > self.pp_lap_count:
            self.pp_lap_count = self.obs['lap_counts'][1]
            print(f"PurePursuit completed lap {self.pp_lap_count} in {self.obs['lap_times'][1]:.2f}s")
        
        # Process observation for RL
        processed_obs = self._preprocess_obs(self.obs)
        
        # Compute reward
        reward = self._compute_reward()
        self.prev_lap_count = self.obs['lap_counts'][0]
        
        done = done or self.step_count >= self.max_steps
        
        info['rl_lap_count'] = self.rl_lap_count
        info['pp_lap_count'] = self.pp_lap_count
        if self.rl_lap_count > 0:
            info['rl_lap_time'] = self.obs['lap_times'][0]
        if self.pp_lap_count > 0:
            info['pp_lap_time'] = self.obs['lap_times'][1]
        
        return processed_obs, reward, done, info

    def _preprocess_obs(self, obs):
        """
        Process observations
        """
        # Get PPO agent pose and velocity
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0]
        linear_vel = obs['linear_vels_x'][0]

        # compute angle error to lookahead point
        position = np.array([pose_x, pose_y])
        lookahead_point = self.outer_planner._get_current_waypoint(
            self.outer_planner.waypoints, self.conf.tlad, position, pose_theta
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
        """
        Convert RL agent's action to simulation action
        """
        MAX_STEER = 1.0
        # Apply action smoothing to reduce oscillations
        raw_steer = float(action[0]) * MAX_STEER
        smoothed_steer = 0.8 * self.prev_steering + 0.2 * raw_steer  # 80/20 blend
        self.prev_steering = smoothed_steer  # update
        
        # Get current position and orientation
        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]
        
        # use the outer planner to determine speed
        speed, _ = self.outer_planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        speed = np.clip(speed, 0.0, self.max_speed)
        
        return np.array([[smoothed_steer, speed]], dtype=np.float32)

    def _generate_purepursuit_action(self):
        """Generate PurePursuit action"""
        # get current position and orientation for purepursuit agent
        pose_x = self.obs['poses_x'][1]
        pose_y = self.obs['poses_y'][1]
        pose_theta = self.obs['poses_theta'][1]
        
        # Generate PurePursuit action from inner planner
        speed, steering = self.inner_planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        speed = np.clip(speed, 0.0, self.inner_max_speed)
        
        return np.array([[steering, speed]])
    
    def _compute_reward(self):
        """
        Compute reward using the same structure as in F110RLWrapperPPO
        """
        # Extract current steering for consistency calculation
        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]
        
        # Get the steering value that was applied
        _, current_steering = self.outer_planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)

        # --- Basic reward components ---
        speed_reward = 0.1 * self.obs['linear_vels_x'][0]
        crash_penalty = self.crash_penalty if self.obs['collisions'][0] > 0 else 0.0

        # Use squared angle error for smoother penalty
        angle_penalty = self.angle_penalty * (self.angle_error ** 2)
        
        alive_bonus = self.alive_bonus

        # --- Steering consistency reward ---
        steering_consistency = self.steering_consistency_coef * abs(self.prev_steering - current_steering)
        self.prev_steering = current_steering

        # --- Lap Bonus / Penalty Logic ---
        MIN_VALID_LAP_RATIO = 0.8  # fraction of track to qualify as a valid lap
        lap_bonus = 0.0

        # --- Distance-based reward ---
        wpts = np.vstack((self.outer_planner.waypoints[:, self.conf.wpt_xind],
                        self.outer_planner.waypoints[:, self.conf.wpt_yind])).T
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
            # Reward for discovering a new segment (higher reward for segments farther along the track)
            exploration_reward = 0.5 + (0.5 * current_segment / self.num_segments)
            self.visited_segments.add(current_segment)

        distance_reward = self.distance_reward_coef * (progress_diff / self.conf.track_length)

        centerline_deviation = abs(self.angle_error)
        center_driving_reward = -1.0 * centerline_deviation**2  # Quadratic penalty for deviation

        # Combine all rewards
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

        # Store reward components for logging
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

    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode)


class MultiAgentF110WrapperSAC(gym.Wrapper):
    """
    A wrapper for multi-agent comparison between SAC agents and PurePursuit.
    This wrapper exactly matches the F110RLWrapperSAC observation space, action space,
    and reward computation to ensure compatibility.
    """
    def __init__(self, 
                env, 
                rl_model,
                conf,
                inner_planner,
                outer_planner,
                max_steps=5000,
                num_segments=20,
                inner_max_speed=6.0,
                outer_max_speed=7.0,
                crash_penalty=-10.0,
                angle_penalty=-1.0,
                alive_bonus=1.0,
                lap_bonus=2.0,
                distance_reward_coef=500,
                steering_consistency_coef=-3.0):
        super().__init__(env)
        self.env = env
        self.rl_model = rl_model
        self.conf = conf
        self.max_steps = max_steps
        self.step_count = 0
        self.obs = None
        
        # Speed parameters
        self.inner_max_speed = inner_max_speed
        self.max_speed = outer_max_speed
        
        # Set up the planners
        self.inner_planner = inner_planner
        self.planner = outer_planner
        
        # SAC-specific observation normalisation parameters
        self.obs_means = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
        self.obs_stds = np.array([5.0, 5.0, np.pi, 2.0, 0.5])
        
        # Tracking lap counts
        self.rl_lap_count = 0
        self.pp_lap_count = 0
        
        self.angle_error = 0.0
        self.prev_lap_count = 0
        self.prev_progress_distance = 0.0
        self.prev_steering = 0.0
        
        # Reward parameters
        self.crash_penalty = crash_penalty
        self.angle_penalty = angle_penalty
        self.alive_bonus = alive_bonus
        self.lap_bonus = lap_bonus
        self.distance_reward_coef = distance_reward_coef
        self.steering_consistency_coef = steering_consistency_coef
        
        # Track segmentation for exploration rewards
        self.num_segments = num_segments
        self.visited_segments = set()
        self.segment_size = self.conf.track_length / num_segments
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-10.0, -10.0, -10.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            shape=(5,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            shape=(1,),
            dtype=np.float32
        )

    def reset(self):
        self.step_count = 0
        self.prev_steering = 0.0
        self.rl_lap_count = 0
        self.pp_lap_count = 0
        self.prev_lap_count = 0
        self.visited_segments = set()
        
        # Reset environment with two agents
        # Agent 0: SAC agent (outer line)
        # Agent 1: PurePursuit (inner line)
        initial_poses = np.array([
            [self.conf.sx2, self.conf.sy2, self.conf.stheta2],  # SAC agent starting pose (outer)
            [self.conf.sx, self.conf.sy, self.conf.stheta]   # PurePursuit starting pose (inner)
        ])
        
        raw_obs, *_ = self.env.reset(poses=initial_poses)
        self.obs = raw_obs
        
        # Initialise progress tracking based on current position
        wpts = np.vstack((self.planner.waypoints[:, self.conf.wpt_xind],
                          self.planner.waypoints[:, self.conf.wpt_yind])).T
        current_position = np.array([raw_obs['poses_x'][0], raw_obs['poses_y'][0]])
        _, _, t, seg_idx = nearest_point_on_trajectory(current_position, wpts)
        self.prev_progress_distance = seg_idx + t
        
        # Process observation for RL agent
        processed_obs = self._preprocess_obs(raw_obs)
        return processed_obs

    def step(self, rl_action):
        # Convert RL action to simulation action
        rl_sim_action = self._preprocess_action(rl_action)
        
        # Generate PurePursuit action
        pp_sim_action = self._generate_purepursuit_action()
        
        # Combine actions for both agents
        combined_action = np.vstack((rl_sim_action, pp_sim_action))
        
        # Step the environment
        step_result = self.env.step(combined_action)
        if len(step_result) == 5:
            self.obs, rewards, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            self.obs, rewards, done, info = step_result
        
        self.step_count += 1
        
        # Track lap counts
        if self.obs['lap_counts'][0] > self.rl_lap_count:
            self.rl_lap_count = self.obs['lap_counts'][0]
            print(f"RL agent completed lap {self.rl_lap_count} in {self.obs['lap_times'][0]:.2f}s")
            
        if self.obs['lap_counts'][1] > self.pp_lap_count:
            self.pp_lap_count = self.obs['lap_counts'][1]
            print(f"PurePursuit completed lap {self.pp_lap_count} in {self.obs['lap_times'][1]:.2f}s")
        
        # Process observation for RL
        processed_obs = self._preprocess_obs(self.obs)
        
        # Compute reward using the same structure
        reward = self._compute_reward()
        self.prev_lap_count = self.obs['lap_counts'][0]
        
        # Check if done
        done = done or self.step_count >= self.max_steps
        
        info['rl_lap_count'] = self.rl_lap_count
        info['pp_lap_count'] = self.pp_lap_count
        if self.rl_lap_count > 0:
            info['rl_lap_time'] = self.obs['lap_times'][0]
        if self.pp_lap_count > 0:
            info['pp_lap_time'] = self.obs['lap_times'][1]
        
        return processed_obs, reward, done, info

    def _preprocess_obs(self, obs):
        """Process observations for the RL agent"""
        # extract pose and velocity
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
        
        # Create raw observation vector
        raw_obs = np.array([pose_x, pose_y, pose_theta, linear_vel, self.angle_error], dtype=np.float32)
        
        # Apply normalisation
        normalized_obs = (raw_obs - self.obs_means) / self.obs_stds
        
        return normalized_obs

    def _preprocess_action(self, action):
        """Convert SAC agent's action to simulation action"""
        MAX_STEER = 1.0
        # Apply action smoothing to reduce oscillations
        raw_steer = float(action[0]) * MAX_STEER
        smoothed_steer = 0.8 * self.prev_steering + 0.2 * raw_steer  # 80/20 blend
        self.prev_steering = smoothed_steer  # Update for next time
        
        # Get current position and orientation for SAC agent
        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]
        
        # Use the planner to determine baseline speed
        speed, _ = self.planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        speed = np.clip(speed, 0.0, self.max_speed)
        
        return np.array([[smoothed_steer, speed]], dtype=np.float32)

    def _generate_purepursuit_action(self):
        """Generate PurePursuit action"""
        # Get current position and orientation for purepursuit agent
        pose_x = self.obs['poses_x'][1]
        pose_y = self.obs['poses_y'][1]
        pose_theta = self.obs['poses_theta'][1]
        
        # Generate PurePursuit action from inner planner
        speed, steering = self.inner_planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        speed = np.clip(speed, 0.0, self.inner_max_speed)
        
        return np.array([[steering, speed]])
    
    def _compute_reward(self):
        """Compute reward using the same structure"""
        # extract current state
        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]
        
        # get current steering for consistency calculation
        _, current_steering = self.planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        
        # Basic reward components
        speed_reward = 0.1 * self.obs['linear_vels_x'][0]
        crash_penalty = self.crash_penalty if self.obs['collisions'][0] > 0 else 0.0
        angle_penalty = self.angle_penalty * (self.angle_error ** 2)
        alive_bonus = self.alive_bonus
        
        # Steering consistency
        steering_consistency = self.steering_consistency_coef * abs(self.prev_steering - current_steering)
        self.prev_steering = current_steering
        
        # Distance-based reward
        wpts = np.vstack((self.planner.waypoints[:, self.conf.wpt_xind],
                         self.planner.waypoints[:, self.conf.wpt_yind])).T
        current_position = np.array([pose_x, pose_y])
        _, _, t, seg_idx = nearest_point_on_trajectory(current_position, wpts)
        current_progress = seg_idx + t
        lap_increase = self.obs['lap_counts'][0] - self.prev_lap_count
        if lap_increase > 0:
            current_progress += lap_increase * self.conf.track_length
        
        progress_diff = current_progress - self.prev_progress_distance
        self.prev_progress_distance = current_progress
        
        # Lap bonus
        lap_bonus = 0.0
        MIN_VALID_LAP_RATIO = 0.8
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
            # Reward for discovering a new segment (higher reward for segments farther along the track)
            exploration_reward = 0.5 + (0.5 * current_segment / self.num_segments)
            self.visited_segments.add(current_segment)
        
        # Distance reward proportional to progress made
        distance_reward = self.distance_reward_coef * (progress_diff / self.conf.track_length)
        
        # Center driving reward (quadratic penalty for deviation)
        center_driving_reward = -1.0 * self.angle_error**2
        
        # Combine all rewards
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
        
        # Store reward components for logging
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

    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode)