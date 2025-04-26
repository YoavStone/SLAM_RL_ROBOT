import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time

from sim_control.SimulationResetHandler import SimulationResetHandler
from .GazeboEnv import GazeboEnv


class DQLEnv:
    """Adapter class that bridges between GazeboEnv and the DQL agent"""

    def __init__(self, rad_of_robot=0.34):
        # Initialize ROS node for environment
        self.gazebo_env = GazeboEnv(rad_of_robot=rad_of_robot)

        self.reset_handler = SimulationResetHandler(self.gazebo_env)

        # Run a few spin cycles to get initial data
        for _ in range(10):
            rclpy.spin_once(self.gazebo_env, timeout_sec=0.1)

        # Properties needed by DQL agent
        self.observation_space = None
        self.action_space = self.gazebo_env.action_space
        self.actions = self.gazebo_env.actions
        self.rad_of_robot = self.gazebo_env.rad_of_robot

        # Track episode state
        self.current_episode_reward = 0.0
        self.step_count = 0

    def get_state_size(self):
        return self.gazebo_env.get_state_size()

    def get_action_size(self):
        return self.gazebo_env.get_action_size()

    def get_state(self):
        return self.gazebo_env.get_state()

    def update_observation_space(self):
        if self.gazebo_env.observation_space is not None:
            self.observation_space = self.gazebo_env.observation_space
            return True
        return False

    def step(self, action):
        """
        Execute action and get new state, reward, etc. (gym-like interface)
        Also handles automatic environment resets when needed.
        """
        # Skip actions if reset is in progress
        if self.reset_handler.is_reset_in_progress():
            self.gazebo_env.get_logger().debug("Skipping action during reset")
            # Return the last state with zero reward and done=False
            return self.get_state(), 0.0, False, False, {}

        # Execute the action
        self.gazebo_env.execute_action(action)

        # Wait for a period (this gives the simulation time to update)
        start_time = time.time()
        while time.time() - start_time < 0.1:  # Wait for 0.1 seconds
            rclpy.spin_once(self.gazebo_env, timeout_sec=0.01)

        # Get the new state
        new_state = self.gazebo_env.get_state()

        # Calculate reward for this step
        reward, terminated = self.gazebo_env.reward_calculator.calc_reward(
            0.1,  # Fixed time step of 0.1 seconds
            self.gazebo_env.measured_distance_to_walls,
            self.gazebo_env.map_processed,
            self.gazebo_env.grid_position,
            self.gazebo_env.pos
        )

        # Track episode reward
        self.current_episode_reward += reward
        self.step_count += 1
        self.gazebo_env.step_counter = self.step_count

        # Let the reset handler evaluate if we need a reset
        truncated = False

        # Check if this is a self-termination (e.g., time limit or map completion)
        if not terminated:
            truncated = self.gazebo_env.reward_calculator.check_time_and_map_completion(self.gazebo_env.map_processed)

        # Only call the reset handler if something ended the episode
        is_done = terminated or truncated
        if is_done:
            self.reset_handler.update(terminated, truncated)

        return new_state, reward, terminated, truncated, {}

    def reset(self):
        """
        External reset interface for the agent.
        Returns initial state and empty info dict.
        """
        # Check if we've received initial odometry data
        if self.gazebo_env.current_odom is None:
            self.gazebo_env.get_logger().info("Waiting for initial odometry data before reset...")

            # Wait for a brief period for initial data, then proceed anyway
            start_time = time.time()
            timeout = 5.0  # 5 second timeout

            while self.gazebo_env.current_odom is None:
                rclpy.spin_once(self.gazebo_env, timeout_sec=0.1)

                if time.time() - start_time > timeout:
                    self.gazebo_env.get_logger().warn("Timeout waiting for odometry data")
                    break

                time.sleep(0.1)

        # Check if we're already in a reset state
        if self.reset_handler.is_reset_in_progress():
            self.gazebo_env.get_logger().info("Reset already in progress, waiting...")

            # Wait until reset is complete
            while self.reset_handler.is_reset_in_progress():
                rclpy.spin_once(self.gazebo_env, timeout_sec=0.1)

            self.gazebo_env.reset()

            return self.get_state(), {}

        # If no reset is in progress, initiate one
        self.gazebo_env.get_logger().info("Manual reset requested")

        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.step_count = 0

        # Let the reset handler know we're starting a reset
        self.reset_handler.is_resetting = True

        # Start the reset sequence (odom correction -> teleport -> SLAM reset)
        self.reset_handler.reset_environment()

        # Wait until reset is complete
        while self.reset_handler.is_reset_in_progress():
            rclpy.spin_once(self.gazebo_env, timeout_sec=0.1)

        self.gazebo_env.reset()

        # Now that reset is done, get the new state
        return self.get_state(), {}

    def close(self):
        """Clean up resources"""
        # Send stop command to robot
        stop_cmd = Twist()
        self.gazebo_env.cmd_vel_pub.publish(stop_cmd)