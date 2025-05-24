import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time
import numpy as np

from gymnasium import spaces

from sim_control.SimulationResetHandler import SimulationResetHandler

from .RewardCalculator import RewardCalculator
from .SensorsProcessor import SensorsProcessor
from .CommandPublisher import CommandPublisher

# Constants for env's map robot and vel
from constants.constants import (
    LINEAR_SPEED,
    ANGULAR_SPEED,
    RAD_OF_ROBOT,
    ROBOT_RAD_SAFE_FACTOR,
    MAP_SIZE
)


class DQLEnv:
    """
    Adapter class that bridges between GazeboEnv and the DQL agent
    And provides AI GYM like interface for the agent
    """

    def __init__(self, is_sim=True, rad_of_robot=RAD_OF_ROBOT):

        # Robot properties
        self.rad_of_robot = rad_of_robot * ROBOT_RAD_SAFE_FACTOR  # radius from lidar to tip with safety margin

        self.is_sim = is_sim

        # Initialize ROS node for environment
        self.cmd_publisher = CommandPublisher()
        self.sensors_processor = SensorsProcessor()

        if self.is_sim:
            self.reset_handler = SimulationResetHandler(self.sensors_processor, self.cmd_publisher)
        else:
            self.reset_handler = None

        # Run a few spin cycles to get initial data
        for _ in range(10):
            rclpy.spin_once(self.sensors_processor, timeout_sec=0.1)

        # Properties needed by DQL agent. Gym-like interface variables
        self.actions = self.sensors_processor.actions
        self.action_space = spaces.Discrete(len(self.actions))
        print("action space: ", self.action_space)
        self.observation_space = None
        self.set_obs_space()

        self.reward_calculator = RewardCalculator(LINEAR_SPEED, self.rad_of_robot)

        # Track episode state
        self.current_episode_reward = 0.0
        self.step_count = 0

    def set_obs_space(self):
        if self.observation_space is None:
            self.observation_space = spaces.Box(
                low=np.array([-1, -1, -100, -100] + [-3, -3] + [0] * 16 + [-1] * MAP_SIZE),
                high=np.array([1, 1, 100, 100] + [3, 3] + [13] * 16 + [1] * MAP_SIZE),
                dtype=np.float32
            )
            print(f"Observation space initialized with size {self.observation_space}")

    def get_state_size(self):
        return self.sensors_processor.get_state_size()

    def get_action_size(self):
        return self.sensors_processor.get_action_size()

    def get_state(self):
        return self.sensors_processor.get_state()

    def update_observation_space(self):
        if self.observation_space is not None:
            return True
        return False

    def step(self, action):
        """
        Execute action and get new state, reward, etc. (gym-like interface)
        Also handles automatic environment resets when needed.
        """
        # Skip actions if reset is in progress
        if self.is_sim:
            if self.reset_handler.is_reset_in_progress():
                self.sensors_processor.get_logger().debug("Skipping action during reset")
                # Return the last state with zero reward and done=False
                return self.get_state(), 0.0, False, False, {}

        # Execute the action
        self.cmd_publisher.execute_action(action)

        # Wait for a period (this gives the simulation time to update)
        start_time = time.time()
        while time.time() - start_time < 0.1:  # Wait for 0.1 seconds
            rclpy.spin_once(self.sensors_processor, timeout_sec=0.01)

        # Get the new state
        new_state = self.sensors_processor.get_state()

        # Calculate reward for this step
        reward, terminated = self.reward_calculator.calc_reward(
            0.1,  # Fixed time step of 0.1 seconds
            self.sensors_processor.measured_distance_to_walls,
            self.sensors_processor.map_processed,
            self.sensors_processor.grid_position,
            self.sensors_processor.pos,
            action
        )

        # Track episode reward
        self.current_episode_reward += reward
        self.step_count += 1
        self.sensors_processor.step_counter = self.step_count

        # Let the reset handler evaluate if we need a reset
        truncated = False

        # Check if this is a self-termination (e.g., step limit or map completion)
        if not terminated:
            truncated = self.reward_calculator.check_steps_and_map_completion()

        # Only call the reset handler if something ended the episode
        is_done = terminated or truncated
        if is_done:
            if self.is_sim:
                self.reset_handler.update(terminated, truncated)

        return new_state, reward, terminated, truncated, {}

    def reset(self):
        """
        External reset interface for the agent.
        Returns initial state and empty info dict.
        """
        # Check if we've received initial odometry data
        if self.sensors_processor.current_odom is None:
            self.sensors_processor.get_logger().info("Waiting for initial odometry data before reset...")

            # Wait for a brief period for initial data, then proceed anyway
            start_time = time.time()
            timeout = 5.0  # 5 second timeout

            while self.sensors_processor.current_odom is None:
                rclpy.spin_once(self.sensors_processor, timeout_sec=0.1)

                if time.time() - start_time > timeout:
                    self.sensors_processor.get_logger().warn("Timeout waiting for odometry data")
                    break

                time.sleep(0.1)

        # Check if we're already in a reset state
        if self.is_sim:
            if self.reset_handler.is_reset_in_progress():
                self.sensors_processor.get_logger().info("Reset already in progress, waiting...")

                # Wait until reset is complete
                while self.reset_handler.is_reset_in_progress():
                    rclpy.spin_once(self.sensors_processor, timeout_sec=0.1)

                time.sleep(0.5)  # so everything has time to reset / if not sim than so it wont try to move to much after hitting wall

                self.reward_calculator.reward_reset()

                self.sensors_processor.reset()

                # Send stop command
                self.cmd_publisher.execute_action(0)

                self.reward_calculator.set_total_cells(MAP_SIZE)

                # Now that reset is done, get the new state
                return self.get_state(), {}

        # If no reset is in progress, initiate one
        self.sensors_processor.get_logger().info("Manual reset requested")

        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.step_count = 0

        # Let the reset handler know we're starting a reset
        if self.is_sim:
            self.reset_handler.is_resetting = True

        # Start the reset sequence (odom correction -> teleport -> SLAM reset)
        if self.is_sim:
            self.reset_handler.reset_environment()

        # Wait until reset is complete
        if self.is_sim:
            while self.reset_handler.is_reset_in_progress():
                rclpy.spin_once(self.sensors_processor, timeout_sec=0.1)

        time.sleep(0.5)  # so everything has time to reset / if not sim than so it wont try to move to much after hitting wall

        self.reward_calculator.reward_reset()

        self.sensors_processor.reset()

        # Send stop command
        self.cmd_publisher.execute_action(0)

        self.reward_calculator.set_total_cells(MAP_SIZE)

        # Now that reset is done, get the new state
        return self.get_state(), {}

    def close(self):
        """Clean up resources"""
        # Send stop command
        self.cmd_publisher.execute_action(0)