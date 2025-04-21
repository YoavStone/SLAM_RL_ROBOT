import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
import numpy as np
import time
import math
import torch
from gymnasium import spaces

from .cropped_map_visualizer import MapVisualizationNode

# Constants
CONTINUES_PUNISHMENT = -5  # amount of punishment for every sec wasted
HIT_WALL_PUNISHMENT = -200
CLOSE_TO_WALL_PUNISHMENT = 0.2  # calc dis to wall pun = calced punishment by dis to wall*CLOSE_TO_WALL_PUNISHMENT
EXPLORATION_REWARD = 5.0  # reward for every newly discovered cell
MOVEMENT_REWARD = 1.0  # reward for moving beyond a threshold (so it wont stay in place)
REVISIT_PENALTY = -0.1  # punishment for revisiting a cell in the map

LINEAR_SPEED = 0.3  # irl: 0.3  # m/s
ANGULAR_SPEED = 0.3 * 2  # irl: 0.3  # rad/s


class GazeboEnv(Node):
    """ROS2 Node that interfaces with Gazebo and provides a gym-like environment interface"""

    def __init__(self, rad_of_robot=0.34):
        super().__init__('gazebo_env_node')

        # cropped map visualizer
        print("Creating visualization node...")
        self.vis_node = MapVisualizationNode()
        # Create timer to periodically publish the map
        self.pub_crop_timer = self.create_timer(1.0, self.publish_cropped_map)
        print("Visualization node created")

        # Robot properties
        self.rad_of_robot = rad_of_robot * 1.3  # radius from lidar to tip with safety margin

        # Environment state
        self.map_processed = []  # Processed map data for DQL input
        self.pos = [0.0, 0.0, 0.0]  # [orientation, x, y]
        self.slam_pose = None  # Store the latest SLAM pose
        self.measured_distance_to_walls = [10.0] * 8  # distances in eighths of circle
        self.last_update_time = time.time()

        self.previous_map = None
        self.last_position = None
        self.map_raw = None
        self.visit_count_map = None

        # Action space: stop, forward, back, right, left
        self.actions = [0, 1, 2, 3, 4]

        # Termination indications
        self.episode_start_time = time.time()
        self.total_cells = None
        self.explored_threshold = 0.93  # 93%
        self.max_episode_duration = 120  # seconds

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.slam_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.slam_pose_callback, 10)

        # Gym-like interface variables
        self.observation_space = None  # Will be initialized after first data is received
        self.action_space = spaces.Discrete(len(self.actions))

        # Data ready flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.slam_pose_ready = False

        # Timer for environment update (0.1 second interval)
        self.timer = self.create_timer(0.1, self.update_timer_callback)

        print('Gazebo Environment Node initialized')

    def publish_cropped_map(self):
        """Trigger map visualization publication if map data is available"""
        if hasattr(self, 'map_processed') and self.map_processed:
            # If we have valid map data, call the visualization node to publish it
            self.vis_node.publish_map()

    def slam_pose_callback(self, msg):
        """Process SLAM pose data"""
        try:
            # Extract position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y

            # Extract orientation (yaw from quaternion)
            orientation = msg.pose.pose.orientation
            # Get yaw from quaternion
            yaw = 2 * math.atan2(orientation.z, orientation.w)

            self.slam_pose = [yaw, x, y]
            self.slam_pose_ready = True

        except Exception as e:
            self.get_logger().error(f"Error processing SLAM pose: {e}")

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

        # Extract orientation
        orientation = msg.pose.pose.orientation
        yaw = 2 * math.atan2(orientation.z, orientation.w)

        self.pos = [yaw, x, y]
        self.odom_ready = True

    def map_callback(self, msg):
        """Process SLAM map data by cropping a 6m x 6m area centered on the robot's starting position"""
        # Store raw map data
        self.map_raw = msg

        # Extract map metadata
        resolution = msg.info.resolution  # Typically 0.05m or similar. for me now its 0.15m
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Calculate size of 6m x 6m area in grid cells (maintaining resolution)
        crop_size_meters = 6.0  # 6m x 6m area
        crop_size_cells = int(crop_size_meters / resolution)

        # Ensure crop_size_cells is even for better centering
        if crop_size_cells % 2 != 0:
            crop_size_cells += 1

        # Store the center position (robot starting position) if not already stored
        if not hasattr(self, 'center_cell_x') or not hasattr(self, 'center_cell_y'):
            # Use SLAM pose if available for better initial position
            if self.slam_pose is not None:
                # Convert SLAM position to grid cell coordinates
                self.center_cell_x = int((self.slam_pose[1] - origin_x) / resolution)
                self.center_cell_y = int((self.slam_pose[2] - origin_y) / resolution)
                print(f"Fixed map center using SLAM pose: ({self.center_cell_x}, {self.center_cell_y})")
            elif self.odom_ready:
                # Fall back to odometry if SLAM not available
                self.center_cell_x = int((self.pos[1] - origin_x) / resolution)
                self.center_cell_y = int((self.pos[2] - origin_y) / resolution)
                print(f"Fixed map center using odometry: ({self.center_cell_x}, {self.center_cell_y})")
            else:
                # If no position data, use the center of the map
                self.center_cell_x = width // 2
                self.center_cell_y = height // 2
                print(f"Fixed map center using map center: ({self.center_cell_x}, {self.center_cell_y})")

        # Calculate boundaries for cropping
        half_size = crop_size_cells // 2
        min_x = max(0, self.center_cell_x - half_size)
        min_y = max(0, self.center_cell_y - half_size)

        # CRITICAL FIX: Ensure we don't go out of bounds but maintain fixed crop size
        # Instead of truncating at the edge, we shift the window to fully fit within bounds
        if min_x + crop_size_cells > width:
            min_x = max(0, width - crop_size_cells)
        if min_y + crop_size_cells > height:
            min_y = max(0, height - crop_size_cells)

        # Calculate max coordinates based on the fixed crop size
        max_x = min_x + crop_size_cells
        max_y = min_y + crop_size_cells

        # Debug output
        print(f"Cropping map: [{min_x}:{max_x}, {min_y}:{max_y}] from original {width}x{height}")
        print(f"Crop dimensions: {max_x - min_x}x{max_y - min_y}")

        # Create empty cropped map with the correct size
        cropped_map = []

        # Extract the map data cells, ensuring we stay in bounds
        for y in range(min_y, max_y):
            row_data = []  # Store row for debugging
            for x in range(min_x, max_x):
                if 0 <= y < height and 0 <= x < width:
                    idx = y * width + x
                    if idx < len(msg.data):
                        if msg.data[idx] == -1:  # Unknown
                            cell_value = -1.0
                        else:  # 0-100 scale to 0-1
                            cell_value = float(msg.data[idx]) / 100.0
                    else:
                        # This should not happen if our bounds checking is correct
                        cell_value = -1.0
                        print(f"Warning: Index {idx} out of bounds for msg.data (len={len(msg.data)})")
                else:
                    # Out of bounds of the original map
                    cell_value = -1.0
                    print(f"Warning: Coordinates ({x},{y}) out of bounds for original map {width}x{height}")

                cropped_map.append(cell_value)
                row_data.append(cell_value)

            # Debug: print first and last row
            # if y == min_y or y == max_y - 1:
            #     print(f"Row {y - min_y} data sample: {row_data[:5]}...")

        # Verify the size of the cropped map
        expected_size = crop_size_cells * crop_size_cells
        actual_size = len(cropped_map)
        if actual_size != expected_size:
            print(f"WARNING: Unexpected cropped map size. Expected {expected_size}, got {actual_size}")
            # Ensure correct size by padding/truncating if needed
            if actual_size < expected_size:
                cropped_map.extend([-1.0] * (expected_size - actual_size))
            else:
                cropped_map = cropped_map[:expected_size]

        self.map_processed = cropped_map
        self.map_ready = True

        # After processing the map, update visualization
        self.vis_node.set_map(cropped_map, resolution)

        if self.total_cells is None:
            self.total_cells = len(cropped_map)

        # Log info about the cropped map
        # print(f"Fixed cropped map: {crop_size_cells}x{crop_size_cells} cells " +
        #       f"centered near ({self.center_cell_x}, {self.center_cell_y}) from original {width}x{height}")
        # print(f"First 100 cells: {cropped_map[:100]}")
        # print(f"Last 100 cells: {cropped_map[-100:]}")

        print("updated map")
        if self.observation_space is None:
            obs_size = len(self.get_state())
            self.observation_space = spaces.Box(
                low=np.array([-4, -100, -100] + [0] * 8 + [-1] * len(self.map_processed)),
                high=np.array([4, 100, 100] + [10] * 8 + [1] * len(self.map_processed)),
                dtype=np.float32
            )
            print(f"Observation space initialized with size {obs_size}")

    def update_timer_callback(self):
        """Timer callback to update environment state at 10Hz (0.1 seconds)"""
        if self.scan_ready and self.map_ready and (self.odom_ready or self.slam_pose_ready):
            dt = time.time() - self.last_update_time
            self.last_update_time = time.time()

            # This would typically be called from the DQL agent's update loop
            new_state, reward, is_terminated = self.update_env(
                self.map_processed,
                self.slam_pose if self.slam_pose is not None else self.pos,
                self.measured_distance_to_walls,
                dt
            )

            if is_terminated:
                print("Environment terminated")

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
        # Use SLAM pose if available, otherwise fall back to odometry
        position = self.slam_pose if self.slam_pose is not None else self.pos
        return position + self.measured_distance_to_walls + self.map_processed

    def get_state_size(self):
        """Get the size of the state vector"""
        return len(self.get_state())

    def get_action_size(self):
        """Get the number of possible actions"""
        return len(self.actions)

    # def revisit_to_penalty(self):
    #     """Calculate penalty for revisiting already explored areas"""
    #     penalty = 0
    #
    #     # Initialize visit count map
    #     if self.map_raw is not None:
    #         self.visit_count_map = np.zeros(len(self.map_raw.data))
    #     else:
    #         self.visit_count_map = None
    #
    #     if self.visit_count_map is not None:
    #         # Get current position in grid coordinates
    #         if hasattr(self, 'map_raw') and hasattr(self, 'center_cell_x') and hasattr(self, 'center_cell_y'):
    #             resolution = self.map_raw.info.resolution
    #             origin_x = self.map_raw.info.origin.position.x
    #             origin_y = self.map_raw.info.origin.position.y
    #
    #             curr_pos = self.slam_pose if self.slam_pose is not None else self.pos
    #             grid_x = int((curr_pos[1] - origin_x) / resolution)
    #             grid_y = int((curr_pos[2] - origin_y) / resolution)
    #
    #             # Update visit count for current cell
    #             width = self.map_raw.info.width
    #             cell_idx = grid_y * width + grid_x
    #
    #             if 0 <= cell_idx < len(self.visit_count_map):
    #                 # Increment visit count
    #                 self.visit_count_map[cell_idx] += 1
    #
    #                 # Apply penalty for revisits (scaled by number of visits)
    #                 visit_count = self.visit_count_map[cell_idx]
    #                 if visit_count > 1:  # Only penalize cells visited more than once
    #                     revisit_penalty = REVISIT_PENALTY * (visit_count - 1)
    #                     penalty += revisit_penalty
    #                     print(f"Revisit penalty: {revisit_penalty:.2f} for {visit_count} visits")
    #
    #     return penalty

    def movement_to_reward(self, dt):
        """Calculate reward based on distance traveled since last update"""
        reward = 0

        if self.last_position is not None:
            # Calculate distance moved
            curr_pos = self.pos  # use odom pos since it's more accurate for shorter dis (update's more frq)
            distance_moved = math.sqrt(
                (curr_pos[1] - self.last_position[1]) ** 2 +
                (curr_pos[2] - self.last_position[2]) ** 2
            )

            # Only reward significant movement (prevents micro-movements)
            if distance_moved > 0.05:  # 5cm threshold
                movement_reward = MOVEMENT_REWARD * distance_moved * dt
                reward += movement_reward
                print(f"Movement reward: {movement_reward:.2f} for {distance_moved:.2f}m")

        # Store current position for next comparison
        self.last_position = self.pos.copy()

        return reward

    def percent_explored(self):
        if self.total_cells is None or not hasattr(self, 'map_processed') or len(self.map_processed) == 0:
            return 0.0
        known_cells = sum(1 for val in self.map_processed if val != -1.0)
        return known_cells / self.total_cells

    def dis_to_wall_to_punishment(self, dt, new_dis):
        punishment = 0
        closest = min(new_dis)
        is_terminated = False

        # Define the danger zone
        danger_zone_start = self.rad_of_robot * 2.0  # Start increasing punishment from 2x radius
        danger_zone_end = self.rad_of_robot  # Max punishment at actual collision

        # Check for immediate collision
        if closest < self.rad_of_robot:
            return HIT_WALL_PUNISHMENT, True

        # If outside danger zone, no punishment
        if closest >= danger_zone_start:
            return 0, False

        # In danger zone - normalize to 0-1 range
        danger_fraction = (danger_zone_start - closest) / (danger_zone_start - danger_zone_end)

        # Modified sigmoid that reaches full punishment at the end
        # We want the steep middle part to reach -200
        sigmoid_steepness = 12  # Higher value = steeper transition
        shift = 0.7  # Shift the sigmoid curve to make it steeper at the end

        # This modified sigmoid will approach -200 more quickly as danger_fraction increases
        sigmoid_value = 1 / (1 + math.exp(-sigmoid_steepness * (danger_fraction - shift)))

        # Scale to punishment range - we multiply by a factor > 1 to ensure it reaches full punishment
        punishment = -sigmoid_value * abs(HIT_WALL_PUNISHMENT) * CLOSE_TO_WALL_PUNISHMENT * dt # slight discount

        # Clip to maximum punishment (the smaller punishment)
        punishment = max(punishment, HIT_WALL_PUNISHMENT)

        return punishment, is_terminated

    def change_in_map_to_reward(self, new_map):
        """Calculate reward based on newly discovered map cells"""
        # Skip if we don't have a previous map to compare
        if not hasattr(self, 'previous_map') or self.previous_map is None:
            self.previous_map = new_map.copy()
            return 0

        # Count newly discovered cells (changed from -1 to any other value)
        reward = 0
        for i in range(len(new_map)):
            if i < len(self.previous_map):
                # Cell was unknown (-1) and is now known
                if self.previous_map[i] == -1 and new_map[i] != -1:
                    reward += EXPLORATION_REWARD  # Reward for each newly discovered cell

        # Store current map for next comparison
        self.previous_map = new_map.copy()

        return reward

    def check_time_and_map_completion(self):
        # Check map exploration condition
        explored_percent = self.percent_explored()
        if explored_percent >= self.explored_threshold:
            print(
                f"Terminating: {explored_percent * 100:.2f}% of map explored (target: {self.explored_threshold * 100}%)")
            return True

        # Check time-based termination
        elapsed = time.time() - self.episode_start_time
        if elapsed > self.max_episode_duration:
            print(f"Terminating: episode ran for {elapsed:.1f}s (max: {self.max_episode_duration}s)")
            return True

        return False

    def calc_reward(self, time_from_last_env_update, new_dis, new_map):
        """Calculate reward based on time spent, proximity to walls, and exploration"""
        reward = CONTINUES_PUNISHMENT * time_from_last_env_update

        # punishment for being close to walls
        pun, is_terminated = self.dis_to_wall_to_punishment(time_from_last_env_update, new_dis)
        reward += pun

        # reward for exploring new areas
        exploration_reward = self.change_in_map_to_reward(new_map)
        reward += exploration_reward

        # reward for moving to not stay in place
        movement_reward = self.movement_to_reward(time_from_last_env_update)
        reward += movement_reward

        if is_terminated:  # Robot is too close to a wall
            # Stop the robot when it hits a wall
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            if reward != CONTINUES_PUNISHMENT * time_from_last_env_update:  # log if reward is not static
                print("reward_t: ", reward, "is_terminated: ", is_terminated)
            return reward, is_terminated

        is_terminated = self.check_time_and_map_completion()

        if reward != CONTINUES_PUNISHMENT * time_from_last_env_update:  # log if reward is not static
            print("reward_t: ", reward, "is_terminated: ", is_terminated)

        return reward, is_terminated

    def update_env(self, new_map, new_pos, new_dis, dt=0.1):
        """Update environment state and calculate reward"""
        reward, is_terminated = self.calc_reward(dt, new_dis, new_map)
        self.map_processed = new_map
        # Only update self.pos if we're not using SLAM pose
        if self.slam_pose is None or new_pos is self.pos:
            self.pos = new_pos
        self.measured_distance_to_walls = new_dis
        new_state = self.get_state()
        return new_state, reward, is_terminated

    def reset(self):
        """Reset the environment - in Gazebo this would typically involve resetting the simulation"""
        # Reset flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.slam_pose_ready = False

        # Send stop command
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Wait for new data
        timeout = 5.0  # seconds
        start_time = time.time()

        while not (self.scan_ready and self.map_ready and (self.odom_ready or self.slam_pose_ready)):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                print("Timeout waiting for sensor data during reset")
                break

        self.last_update_time = time.time()
        self.episode_start_time = time.time()
        return self.get_state(), {}  # Return state and empty info dict (gym-like interface)


class DQLEnv:
    """Adapter class that bridges between GazeboEnv and the DQL agent"""

    def __init__(self, rad_of_robot=0.34):
        # Initialize ROS node for environment
        self.gazebo_env = GazeboEnv(rad_of_robot=rad_of_robot)

        # Run a few spin cycles to get initial data
        for _ in range(10):
            rclpy.spin_once(self.gazebo_env, timeout_sec=0.1)

        # Don't try to access observation_space directly yet
        self.observation_space = None
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

    def update_observation_space(self):
        if self.gazebo_env.observation_space is not None:
            self.observation_space = self.gazebo_env.observation_space
            return True
        return False

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