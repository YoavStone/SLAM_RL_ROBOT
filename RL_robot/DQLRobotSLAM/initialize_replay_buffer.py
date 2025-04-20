import time
import numpy as np
from std_msgs.msg import Empty
import rclpy


def initialize_replay_buffer(self):
    last_reset_request_time = 0
    min_reset_interval = 10.0  # Minimum seconds between reset requests
    """Fills the replay buffer with initial random experiences."""
    if self.current_obs is None:
        self.current_obs, _ = self.env.reset()
        if self.current_obs is None:
            self.get_logger().error("Failed to get initial observation from environment!")
            return False  # Indicate failure

    init_steps = 0
    buffer_initialization_start = time.time()
    max_initialization_time = self.min_replay_size * 2.5  # in seconds timeout for buffer init

    # Add a counter to detect and break out of reset loops
    consecutive_reset_detections = 0
    max_consecutive_detections = 5  # Allow up to 5 consecutive reset detections before forcing continuation
    last_reset_check_time = time.time()
    force_continue = False

    self.get_logger().info(f"Starting replay buffer initialization, target size: {self.min_replay_size}")

    while len(self.replay_buffer) < self.min_replay_size:
        # Check for timeout
        if time.time() - buffer_initialization_start > max_initialization_time:
            self.get_logger().error(f"Buffer initialization timed out after {max_initialization_time} seconds")
            return False

        # Check if reset is in progress
        current_time = time.time()
        reset_in_progress = (self.reset_handler.is_reset_in_progress() or
                             self.env.reset_handler.is_reset_in_progress() or
                             self.env.observation_space is None)

        # Detect reset loop
        if reset_in_progress:
            if current_time - last_reset_check_time < 2.0:  # Less than 2 seconds since last check
                consecutive_reset_detections += 1
                if consecutive_reset_detections > max_consecutive_detections:
                    self.get_logger().warn(
                        f"Detected potential reset loop (max_consecutive_detections: {max_consecutive_detections}). Forcing continuation.")
                    force_continue = True
                    # Force reset observation space refresh
                    rclpy.spin_once(self.env.gazebo_env, timeout_sec=0.1)
                    self.env.update_observation_space()  # Try to update the observation space
                    time.sleep(1.0)  # Short delay
                    reset_in_progress = False  # Pretend reset is not in progress
            else:
                consecutive_reset_detections = 1  # Reset counter if enough time has passed

            last_reset_check_time = current_time
        else:
            consecutive_reset_detections = 0  # Reset counter if no reset detected

        # Handle reset state
        if reset_in_progress and not force_continue:
            self.get_logger().info("Reset detected during buffer initialization, waiting...", throttle_duration_sec=5.0)
            time.sleep(1.0)  # Wait for reset to complete
            continue
        else:
            if force_continue:
                self.get_logger().info("Forcing continuation after reset loop")
                force_continue = False  # Reset flag

            # If we're forcing continuation or if reset is genuinely complete, check current observation
            if self.current_obs is None:
                self.get_logger().warn("Current observation is None, attempting reset...")
                self.current_obs, _ = self.env.reset()
                if self.current_obs is None:
                    self.get_logger().error("Failed to get observation after forced continuation")
                    time.sleep(2.0)
                    continue

        # Take random action
        try:
            action = self.env.action_space.sample()
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            is_done = terminated or truncated
        except Exception as e:
            self.get_logger().error(f"Error during environment step: {e}")
            time.sleep(1.0)
            continue

        # Handle None observations or reset indication
        current_time = time.time()
        if new_obs is None or (info and info.get("reset_in_progress", False)):
            if current_time - last_reset_request_time > min_reset_interval:
                self.get_logger().warn("Received None observation, requesting reset...")
                last_reset_request_time = current_time
                self.episode_end_pub.publish(Empty())
                time.sleep(2.0)
            else:
                self.get_logger().info("Skipping reset request (cooldown period)")
                time.sleep(1.0)
            # Try to get a new observation
            self.current_obs, _ = self.env.reset()
            continue

        # Ensure observations are numpy arrays before adding
        current_obs_np = np.array(self.current_obs, dtype=np.float32)
        new_obs_np = np.array(new_obs, dtype=np.float32)

        # Add to replay buffer
        self.replay_buffer.append((current_obs_np, action, reward, is_done, new_obs_np))
        init_steps += 1

        # Handle episode end
        if is_done:
            self.get_logger().info(
                f"Episode ended during buffer initialization. Buffer size: {len(self.replay_buffer)}/{self.min_replay_size}")
            # Publish episode end to trigger external reset
            self.episode_end_pub.publish(Empty())
            # Wait for some time but not too long
            time.sleep(2.0)
            # Get new observation
            self.current_obs, _ = self.env.reset()
        else:
            self.current_obs = new_obs

        # Progress log
        if init_steps % 20 == 0:
            self.get_logger().info(f"Replay buffer filling: {len(self.replay_buffer)}/{self.min_replay_size}")

    self.get_logger().info(
        f"✅ Replay buffer initialized with {len(self.replay_buffer)} experiences after {init_steps} steps.")
    self.training_initialized = True
    return True