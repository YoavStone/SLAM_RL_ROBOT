import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
import numpy as np
import time
import math
import torch
from gymnasium import spaces


# Constants
CONTINUES_PUNISHMENT = -2  # amount of punishment for every sec wasted
HIT_WALL_PUNISHMENT = -200
LINEAR_SPEED = 0.3  # m/s
ANGULAR_SPEED = 0.3  # rad/s


class GazeboEnv(Node):
    """ROS2 Node that interfaces with Gazebo and provides a gym-like environment interface"""

    def __init__(self, rad_of_robot=0.34):
        super().__init__('gazebo_env_node')

        # Robot properties
        self.rad_of_robot = rad_of_robot * 1.6  # radius from lidar to tip with safety margin

        # Environment state
        self.map_processed = []  # Processed map data for DQL input
        self.pos = [0.0, 0.0, 0.0]  # [orientation, x, y]
        self.measured_distance_to_walls = [10.0] * 8  # distances in eighths of circle
        self.last_update_time = time.time()

        # Action space: stop, forward, back, right, left
        self.actions = [0, 1, 2, 3, 4]

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)

        # Gym-like interface variables
        self.observation_space = None  # Will be initialized after first data is received
        self.action_space = spaces.Discrete(len(self.actions))

        # Data ready flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False

        # Timer for environment update (0.1 second interval)
        self.timer = self.create_timer(0.1, self.update_timer_callback)

        self.get_logger().info('Gazebo Environment Node initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Divide the scan into 8 sectors and get min distance for each sector
        ranges = np.array(msg.ranges)
        valid_ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

        # Split the scan into 8 equal sectors
        num_sectors = 8
        sector_size = len(valid_ranges) // num_sectors

        self.measured_distance_to_walls = []
        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size if i < num_sectors - 1 else len(valid_ranges)
            sector_ranges = valid_ranges[start_idx:end_idx]
            min_distance = np.min(sector_ranges)
            self.measured_distance_to_walls.append(float(min_distance))

        self.scan_ready = True

    def odom_callback(self, msg):
        """Process odometry data"""
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract orientation (yaw) from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Convert quaternion to Euler angles
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.pos = [yaw, x, y]
        self.odom_ready = True

    def map_callback(self, msg):
        """Process SLAM map data"""
        # Process the occupancy grid to extract relevant features
        width = msg.info.width
        height = msg.info.height

        # Downsample the map to a manageable size (e.g., 10x10 grid)
        downsampled_size = 10
        step_x = max(1, width // downsampled_size)
        step_y = max(1, height // downsampled_size)

        downsampled_map = []
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                idx = y * width + x
                if idx < len(msg.data):
                    # Normalize to [0, 1], where -1 (unknown) becomes 0.5
                    if msg.data[idx] == -1:  # Unknown
                        downsampled_map.append(0.5)
                    else:  # 0-100 scale to 0-1
                        downsampled_map.append(float(msg.data[idx]) / 100.0)

        # Pad or truncate to ensure consistent size
        target_size = downsampled_size * downsampled_size
        if len(downsampled_map) < target_size:
            downsampled_map.extend([0.5] * (target_size - len(downsampled_map)))
        elif len(downsampled_map) > target_size:
            downsampled_map = downsampled_map[:target_size]

        self.map_processed = downsampled_map
        self.map_ready = True

        # Initialize observation space if not already done
        if self.observation_space is None:
            obs_size = len(self.get_state())
            self.observation_space = spaces.Box(
                low=np.array([-math.pi, -100, -100] + [0] * 8 + [0] * len(self.map_processed)),
                high=np.array([math.pi, 100, 100] + [10] * 8 + [1] * len(self.map_processed)),
                dtype=np.float32
            )
            self.get_logger().info(f"Observation space initialized with size {obs_size}")

    def update_timer_callback(self):
        """Timer callback to update environment state at 10Hz (0.1 seconds)"""
        if self.scan_ready and self.odom_ready and self.map_ready:
            dt = time.time() - self.last_update_time
            self.last_update_time = time.time()

            # This would typically be called from the DQL agent's update loop
            new_state, reward, is_terminated = self.update_env(
                self.map_processed, self.pos, self.measured_distance_to_walls, dt
            )

            # In a real implementation, you would send this state back to your agent
            # For demonstration, we'll just log it
            if is_terminated:
                self.get_logger().info("Environment terminated (hit wall)")

    def action_to_cmd(self, action):
        """Convert action index to Twist command"""
        cmd = Twist()

        if action == 0:  # Stop
            pass  # All values are initialized to 0
        elif action == 1:  # Forward
            cmd.linear.x = LINEAR_SPEED
        elif action == 2:  # Back
            cmd.linear.x = -LINEAR_SPEED
        elif action == 3:  # Right
            cmd.angular.z = -ANGULAR_SPEED
        elif action == 4:  # Left
            cmd.angular.z = ANGULAR_SPEED

        return cmd

    def execute_action(self, action):
        """Execute action by publishing to cmd_vel"""
        cmd = self.action_to_cmd(action)
        self.cmd_vel_pub.publish(cmd)

    def get_state(self):
        """Get the current state representation"""
        return self.pos + self.measured_distance_to_walls + self.map_processed

    def get_state_size(self):
        """Get the size of the state vector"""
        return len(self.get_state())

    def get_action_size(self):
        """Get the number of possible actions"""
        return len(self.actions)

    def calc_reward(self, time_from_last_env_update, new_dis, new_map):
        """Calculate reward based on time spent and proximity to walls"""
        reward = CONTINUES_PUNISHMENT * time_from_last_env_update
        closest = min(new_dis)
        is_terminated = False

        if closest < self.rad_of_robot:  # Robot is too close to a wall
            reward += HIT_WALL_PUNISHMENT
            is_terminated = True

            # Stop the robot when it hits a wall
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

        return reward, is_terminated

    def update_env(self, new_map, new_pos, new_dis, dt=0.1):
        """Update environment state and calculate reward"""
        reward, is_terminated = self.calc_reward(dt, new_dis, new_map)
        self.map_processed = new_map
        self.pos = new_pos
        self.measured_distance_to_walls = new_dis
        new_state = self.get_state()
        return new_state, reward, is_terminated

    def reset(self):
        """Reset the environment - in Gazebo this would typically involve resetting the simulation"""
        # This is a placeholder - in a real implementation you would:
        # 1. Call a service to reset the Gazebo simulation
        # 2. Wait for new data from sensors
        # 3. Return the initial state

        # For now, we'll just wait for fresh data
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False

        # Send stop command
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Wait for new data
        timeout = 5.0  # seconds
        start_time = time.time()

        while not (self.scan_ready and self.odom_ready and self.map_ready):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().warn("Timeout waiting for sensor data during reset")
                break

        self.last_update_time = time.time()
        return self.get_state(), {}  # Return state and empty info dict (gym-like interface)


class DQLEnv:
    """Adapter class that bridges between GazeboEnv and the DQL agent"""

    def __init__(self, rad_of_robot=0.34):
        # Initialize ROS node for environment
        # Note: rclpy.init() should be called before this from the main node
        self.gazebo_env = GazeboEnv(rad_of_robot=rad_of_robot)

        # Run a few spin cycles to get initial data
        for _ in range(10):
            rclpy.spin_once(self.gazebo_env, timeout_sec=0.1)

        # Set up gym-compatible interface
        self.observation_space = self.gazebo_env.observation_space
        self.action_space = self.gazebo_env.action_space

        # Properties needed by DQL agent
        self.actions = self.gazebo_env.actions
        self.rad_of_robot = self.gazebo_env.rad_of_robot

    def get_state_size(self):
        return self.gazebo_env.get_state_size()

    def get_action_size(self):
        return self.gazebo_env.get_action_size()

    def get_state(self):
        return self.gazebo_env.get_state()

    def step(self, action):
        """Execute action and get new state, reward, etc. (gym-like interface)"""
        # Execute the action
        self.gazebo_env.execute_action(action)

        # Wait for a period (this gives the simulation time to update)
        start_time = time.time()
        while time.time() - start_time < 0.1:  # Wait for 0.1 seconds
            rclpy.spin_once(self.gazebo_env, timeout_sec=0.01)

        # Get the new state and reward
        new_state = self.gazebo_env.get_state()
        reward, terminated = self.gazebo_env.calc_reward(
            0.1,
            self.gazebo_env.measured_distance_to_walls,
            self.gazebo_env.map_processed
        )

        return new_state, reward, terminated, False, {}  # state, reward, terminated, truncated, info

    def reset(self):
        """Reset the environment (gym-like interface)"""
        return self.gazebo_env.reset()

    def close(self):
        """Clean up resources"""
        # Send stop command to robot
        stop_cmd = Twist()
        self.gazebo_env.cmd_vel_pub.publish(stop_cmd)