# reward_callback.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging
from stable_baselines3.common.callbacks import BaseCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Just log to console - file handler will be added by the callback
    ]
)
logger = logging.getLogger("reward_debug")

class RewardTrackingCallback(BaseCallback):
    """
    Callback for tracking and logging reward components during training.
    Can be used with both SAC and PPO algorithms.
    """
    def __init__(self, verbose=0, log_freq=100, window_size=1000, log_dir="logs", 
                 algorithm_name=None, track_name=None):
        super().__init__(verbose)
        self.log_freq = log_freq  # How often to log
        self.window_size = window_size  # Size of the sliding window for stats
        self.log_dir = log_dir
        
        # Set algorithm name (PPO or SAC) for better log organization
        self.algorithm_name = algorithm_name or "UNKNOWN"
        self.track_name = track_name or "UNKNOWN"
        
        # Create a unique subfolder for this training run with algorithm name
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.run_log_dir = os.path.join(self.log_dir, f"{self.algorithm_name}_{self.track_name}_run_{timestamp}")
        os.makedirs(self.run_log_dir, exist_ok=True)
        
        # Create a specific directory for lap completion logs
        self.laps_dir = os.path.join(self.run_log_dir, "lap_completions")
        os.makedirs(self.laps_dir, exist_ok=True)
        
        # Configure logger to use the new directory
        file_handler = logging.FileHandler(f"{self.run_log_dir}/{self.algorithm_name}_reward_debug.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Create separate logger for lap completions
        self.lap_logger = logging.getLogger(f"lap_completions_{self.algorithm_name}")
        lap_file_handler = logging.FileHandler(f"{self.laps_dir}/{self.algorithm_name}_lap_completions.log")
        lap_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.lap_logger.addHandler(lap_file_handler)
        self.lap_logger.setLevel(logging.INFO)
        
        # Create deques for each reward component
        self.reward_components = {
            'speed_reward': deque(maxlen=window_size),
            'alive_bonus': deque(maxlen=window_size),
            'angle_penalty': deque(maxlen=window_size),
            'crash_penalty': deque(maxlen=window_size),
            'lap_bonus': deque(maxlen=window_size),
            'distance_reward': deque(maxlen=window_size),
            'steering_consistency': deque(maxlen=window_size),
            'exploration_reward': deque(maxlen=window_size),
            'center_driving_reward': deque(maxlen=window_size),
            'total_reward': deque(maxlen=window_size)
        }
        
        # Create deques for tracking recent rewards for lap-specific analysis
        self.recent_rewards = {
            'speed_reward': deque(maxlen=500),  # Store more recent values for lap snapshots
            'alive_bonus': deque(maxlen=500),
            'angle_penalty': deque(maxlen=500),
            'crash_penalty': deque(maxlen=500),
            'lap_bonus': deque(maxlen=500),
            'distance_reward': deque(maxlen=500),
            'steering_consistency': deque(maxlen=500),
            'exploration_reward': deque(maxlen=500),
            'center_driving_reward': deque(maxlen=500),
            'total_reward': deque(maxlen=500)
        }
        
        self.step_count = 0
        self.episode_count = 0
        self.crashes = 0
        self.successful_laps = 0
        self.lap_detection_cooldown = 0  # To prevent multiple detections for the same lap
        
        # Save training parameters for later reference
        self._save_training_info()
        
    def _save_training_info(self):
        """Save basic information about this training run for reference"""
        try:
            info_file = os.path.join(self.run_log_dir, "training_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"Algorithm: {self.algorithm_name}\n")
                f.write(f"Track: {self.track_name}\n")
                f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log frequency: {self.log_freq}\n")
                f.write(f"Window size for stats: {self.window_size}\n")
                
                # Try to extract env parameters if available
                try:
                    if hasattr(self.training_env.envs[0].unwrapped, 'max_steps'):
                        f.write(f"Max steps per episode: {self.training_env.envs[0].unwrapped.max_steps}\n")
                    
                    # Add other interesting parameters
                    constants = [
                        'MAX_SPEED', 'CRASH_PENALTY', 'ANGLE_PENALTY', 'ALIVE_BONUS',
                        'LAP_BONUS', 'DISTANCE_REWARD_COEF', 'STEERING_CONSISTENCY_COEF'
                    ]
                    
                    for const in constants:
                        if hasattr(self.training_env.envs[0].unwrapped, const):
                            value = getattr(self.training_env.envs[0].unwrapped, const)
                            f.write(f"{const}: {value}\n")
                except:
                    f.write("Could not extract environment parameters\n")
                    
        except Exception as e:
            logger.error(f"Failed to save training info file: {str(e)}")
        
    def _on_step(self):
        # Get reward components from the environment
        if hasattr(self.training_env.envs[0].unwrapped, 'last_reward_components'):
            components = self.training_env.envs[0].unwrapped.last_reward_components
            
            # Update our tracking deques
            for key, value in components.items():
                if key in self.reward_components:
                    self.reward_components[key].append(value)
                    self.recent_rewards[key].append(value)
            
            # Track crashes and laps - more aggressive crash detection
            if self.training_env.envs[0].unwrapped.obs['collisions'][0] > 0:
                self.crashes += 1
                
            # Check for lap completion
            if self.lap_detection_cooldown <= 0 and components.get('lap_bonus', 0) > 0:
                # Lap completed!
                self.successful_laps += 1
                self._log_lap_completion(components)
                self.lap_detection_cooldown = 20  # Set cooldown to prevent multiple detections
            
            # Decrement cooldown counter if active
            if self.lap_detection_cooldown > 0:
                self.lap_detection_cooldown -= 1
            
            self.step_count += 1
            
            # Track real episodes more accurately
            if self.training_env.get_attr('step_count')[0] == 0:  # Reset was called
                self.episode_count += 1
            
            # Log periodically
            if self.step_count % self.log_freq == 0:
                self._log_stats()
                
                # Log to TensorBoard
                if self.logger is not None:
                    for key, values in self.reward_components.items():
                        if values:  # Check if deque is not empty
                            # Log mean of each reward component to TensorBoard
                            self.logger.record(f"{self.algorithm_name}/reward_components/{key}", np.mean(values))
                    
                    # Log crash rate and lap completion rate
                    self.logger.record(f"{self.algorithm_name}/metrics/crash_rate", self.crashes / max(1, self.step_count))
                    self.logger.record(f"{self.algorithm_name}/metrics/successful_laps", self.successful_laps)
                    
                    # Dump TensorBoard data
                    self.logger.dump(self.step_count)
            
        return True
    
    def _log_lap_completion(self, components):
        """Special logging when a lap is completed"""
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        lap_number = self.successful_laps
        
        # Get the environment state
        try:
            episode_step = self.training_env.get_attr('step_count')[0]
            pose_x = self.training_env.envs[0].unwrapped.obs['poses_x'][0]
            pose_y = self.training_env.envs[0].unwrapped.obs['poses_y'][0]
            pose_theta = self.training_env.envs[0].unwrapped.obs['poses_theta'][0] 
            velocity = self.training_env.envs[0].unwrapped.obs['linear_vels_x'][0]
            
            # Detailed lap completion log
            log_message = (
                f"[{self.algorithm_name}] LAP COMPLETED! #{lap_number} at step {self.step_count} (episode step: {episode_step})\n"
                f"Position: x={pose_x:.2f}, y={pose_y:.2f}, theta={pose_theta:.2f}, velocity={velocity:.2f}\n"
                f"Current reward components: {components}\n"
                f"Total laps: {self.successful_laps}, Total crashes: {self.crashes}"
            )
            
            self.lap_logger.info(log_message)
            logger.info(f"[{self.algorithm_name}] LAP COMPLETED! #{lap_number} at step {self.step_count} - See lap logs for details")
            
            # Save a special visualization of the reward components leading up to this lap
            self._plot_lap_rewards(lap_number, timestamp)
            
            # Try to extract any additional environment data that might be useful
            if hasattr(self.training_env.envs[0].unwrapped, 'prev_progress_distance'):
                progress = self.training_env.envs[0].unwrapped.prev_progress_distance
                self.lap_logger.info(f"Track progress at completion: {progress:.2f}")
            
        except Exception as e:
            self.lap_logger.error(f"Error logging lap completion: {str(e)}")
    
    def _plot_lap_rewards(self, lap_number, timestamp):
        """Create a special plot of rewards leading up to lap completion"""
        # Create a lap-specific plot 
        filename = f"{self.laps_dir}/{self.algorithm_name}_lap_{lap_number}_rewards_{timestamp}.png"
        
        plt.figure(figsize=(12, 10))
        
        # Add title with algorithm name
        plt.suptitle(f"{self.algorithm_name} - Rewards Leading to Lap {lap_number} Completion", 
                     fontsize=16, y=0.98)
        
        # Plot the most important reward components for lap analysis
        key_components = ['total_reward', 'speed_reward', 'angle_penalty', 
                          'distance_reward', 'steering_consistency', 'center_driving_reward']
        
        for i, component in enumerate(key_components):
            values = list(self.recent_rewards[component])
            if values:
                plt.subplot(len(key_components), 1, i+1)
                plt.plot(values)
                plt.title(f"{component} (leading to lap {lap_number})")
                plt.grid(True)
                
                # Add some statistics
                if len(values) > 0:
                    mean_val = np.mean(values)
                    plt.axhline(y=mean_val, color='r', linestyle='--', 
                                label=f'Mean: {mean_val:.4f}')
                    plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        plt.savefig(filename)
        plt.close()
        
        # Also save a comprehensive plot with all components
        self._plot_rewards(lap_suffix=f"_lap_{lap_number}")
    
    def _on_rollout_end(self):
        # Log stats every 10 episodes
        if self.episode_count % 10 == 0:
            # Basic stats logging without plots
            self._log_stats(detailed=False)
        
        # Only create plots every 100 episodes (or another suitable value)
        if self.episode_count % 100 == 0:
            # Detailed logging with plots
            self._log_stats(detailed=True)
    
    def _log_stats(self, detailed=False):
        """Log the current reward statistics"""
        # Calculate statistics for each component
        stats = {}
        for key, values in self.reward_components.items():
            if values:  # Check if deque is not empty
                stats[key] = {
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values)
                }
        
        # Basic log message with means
        mean_values = {k: round(v['mean'], 4) for k, v in stats.items()}
        logger.info(f"[{self.algorithm_name}] Step {self.step_count}, Episode {self.episode_count} - " 
                   f"Reward means: {mean_values}, "
                   f"Crashes: {self.crashes}, Successful laps: {self.successful_laps}")
        
        # Detailed logging
        if detailed and stats:
            for component, metrics in stats.items():
                logger.info(f"[{self.algorithm_name}]  {component}: mean={metrics['mean']:.4f}, "
                           f"min={metrics['min']:.4f}, max={metrics['max']:.4f}, "
                           f"std={metrics['std']:.4f}")
            
            # Plot the reward components
            self._plot_rewards()
    
    def _plot_rewards(self, lap_suffix=""):
        """Create and save a plot of reward components"""
        components = list(self.reward_components.keys())
        data = [list(self.reward_components[c]) for c in components]
        
        # Only plot if we have data
        if not all(data):
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Add algorithm name to the title
        plt.suptitle(f"{self.algorithm_name} - Reward Components{' for ' + lap_suffix if lap_suffix else ''}", 
                     fontsize=16, y=0.98)
        
        for i, (component, values) in enumerate(self.reward_components.items()):
            if values:  # Check if we have data
                plt.subplot(len(components), 1, i+1)
                plt.plot(list(values))
                plt.title(component)
                plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Use different filename if this is a lap-specific plot
        if lap_suffix:
            plt.savefig(f"{self.laps_dir}/{self.algorithm_name}_reward_components{lap_suffix}.png")
        else:
            plt.savefig(f"{self.run_log_dir}/{self.algorithm_name}_reward_components_step_{self.step_count}.png")
            
        plt.close()