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
from .reward_visualizer import RewardVisualizer
from sim_control.sim_reset_handler import SimulationResetHandler


# Constants
CONTINUES_PUNISHMENT = -1.0  # amount of punishment for every sec
HIT_WALL_PUNISHMENT = -500.0
CLOSE_TO_WALL_PUNISHMENT = 0.35  # calc dis to wall pun = calced punishment by dis to wall*CLOSE_TO_WALL_PUNISHMENT
WALL_POWER = 7.0
EXPLORATION_REWARD = 3.5  # reward for every newly discovered cell
MOVEMENT_REWARD = 0.9  # reward for moving beyond a threshold (so it wont stay in place)
REVISIT_PENALTY = -0.2  # punishment for revisiting a cell in the map
REMEMBER_VISIT_TIME = 1.5  # how long to keep the visit time of a spot so it counts as visited in seconds

LINEAR_SPEED = 0.3  # irl: 0.3  # m/s
ANGULAR_SPEED = 0.3  # irl: 0.3  # rad/s


class GazeboEnv(Node):
    """ROS2 Node that interfaces with Gazebo and provides a gym-like environment interface"""

    def __init__(self, rad_of_robot=0.34):
        super().__init__('gazebo_env_node')

        self.declare_parameter('spawn_location', '')  # Default: empty string means random

        # Get the spawn_location parameter
        self.spawn_location_str = self.get_parameter('spawn_location').get_parameter_value().string_value
        self.get_logger().info(f"Received parameters: spawn_location='{self.spawn_location_str}'")

        # Initialize the reward visualizer
        print("Creating reward visualizer node...")
        self.reward_vis = RewardVisualizer(print_interval=100)  # print_interval= every x steps print slightly detailed loggs
        print("Reward visualizer node created")
        # Store recent reward components for visualization
        self.last_cont_punishment = 0
        self.last_wall_punishment = 0
        self.last_exploration_reward = 0
        self.last_movement_reward = 0
        self.last_revisit_penalty = 0
        self.last_total_reward = 0

        self.step_counter = 0

        # cropped map visualizer
        print("Creating visualization node...")
        self.vis_node = MapVisualizationNode(publish=False)
        # Create timer to periodically publish the map
        self.pub_crop_timer = self.create_timer(1.0, self.publish_cropped_map)
        print("Visualization node created")

        # Robot properties
        self.rad_of_robot = rad_of_robot * 1.3  # radius from lidar to tip with safety margin

        # Environment state
        self.map_processed = []  # Processed map data for DQL input
        self.pos = None  # [orientation, x, y]
        self.velocities = None  # [vx, va]
        self.slam_pose = None  # Store the latest SLAM pose [orientation, x, y]
        self.grid_position = None  # stores position on grid [sin(x), cos(x), x, y]
        self.measured_distance_to_walls = [10.0] * 16  # distances in sixteenths of circle
        self.last_update_time = time.time()

        self.previous_map = None
        self.last_position = None
        self.map_raw = None
        self.visit_count_map = None
        self.center_cell_x = None
        self.center_cell_y = None
        self.current_odom = None
        self.should_update_center = True

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

        print('Gazebo Environment Node initialized')

    def publish_cropped_map(self):
        """Trigger map visualization publication if map data is available"""
        if self.map_processed:
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
        # Divide the scan into 16 sectors and get min distance for each sector
        ranges = np.array(msg.ranges)
        valid_ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

        # Split the scan into 16 equal sectors
        num_sectors = 16
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
        """Process odometry data with enhanced tracking for reset handler"""
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract orientation
        orientation = msg.pose.pose.orientation
        yaw = 2 * math.atan2(orientation.z, orientation.w)

        # Extract velocities
        vx = msg.twist.twist.linear.x
        va = msg.twist.twist.angular.z

        # Update standard position and velocity
        self.pos = [yaw, x, y]
        self.velocities = [vx, va]

        # Store the full odometry message for reset handler
        self.current_odom = msg

        # Log first odometry reception
        if not self.odom_ready:
            self.get_logger().info(f"First odometry data received: position=[{x:.2f}, {y:.2f}]")

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
        if self.should_update_center:
            # Use SLAM pose if available for better initial position
            if self.slam_pose is not None:
                # Convert SLAM position to grid cell coordinates
                self.center_cell_x = int((self.slam_pose[1] - origin_x) / resolution)
                self.center_cell_y = int((self.slam_pose[2] - origin_y) / resolution)
                print(f"Updated map center using SLAM pose: ({self.center_cell_x}, {self.center_cell_y})")
            elif self.odom_ready:
                # Fall back to odometry if SLAM not available
                self.center_cell_x = int((self.pos[1] - origin_x) / resolution)
                self.center_cell_y = int((self.pos[2] - origin_y) / resolution)
                print(f"Updated map center using odometry: ({self.center_cell_x}, {self.center_cell_y})")
            else:
                # If no position data, use the center of the map
                self.center_cell_x = width // 2
                self.center_cell_y = height // 2
                print(f"Updated map center using map center: ({self.center_cell_x}, {self.center_cell_y})")

            # Reset the flag so we don't update center again until next reset
            self.should_update_center = False

        # Calculate boundaries for cropping
        half_size = crop_size_cells // 2
        min_x = max(0, self.center_cell_x - half_size)
        min_y = max(0, self.center_cell_y - half_size)

        # Instead of truncating at the edge, we shift the window to fully fit within bounds
        if min_x + crop_size_cells > width:
            min_x = max(0, width - crop_size_cells)
        if min_y + crop_size_cells > height:
            min_y = max(0, height - crop_size_cells)

        # Calculate max coordinates based on the fixed crop size
        max_x = min_x + crop_size_cells
        max_y = min_y + crop_size_cells

        # Debug output
        # print(f"Cropping map: [{min_x}:{max_x}, {min_y}:{max_y}] from original {width}x{height}")
        # print(f"Crop dimensions: {max_x - min_x}x{max_y - min_y}")

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
                        print(f"Warning: Index {idx} out of bounds for msg.data (len={len(msg.data)}) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! shouldnt happen prob bug in crop map, has never happened before")
                else:
                    # Out of bounds of the original map
                    cell_value = -1.0
                    # print(f"Warning: Coordinates ({x},{y}) out of bounds for original map {width}x{height}")

                cropped_map.append(cell_value)
                row_data.append(cell_value)

            # Debug: print first and last row
            # if y == min_y or y == max_y - 1:
            #     print(f"Row {y - min_y} data sample: {row_data[:5]}...")

        # Verify the size of the cropped map
        expected_size = crop_size_cells * crop_size_cells
        actual_size = len(cropped_map)
        if actual_size != expected_size:
            # print(f"WARNING: Unexpected cropped map size. Expected {expected_size}, got {actual_size}")
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

        # print("updated map")
        if self.observation_space is None:
            obs_size = len(self.get_state())
            self.observation_space = spaces.Box(
                low=np.array([-1, -1, -100, -100] + [-3, -3] + [0] * 16 + [-1] * len(self.map_processed)),
                high=np.array([1, 1, 100, 100] + [3, 3] + [13] * 16 + [1] * len(self.map_processed)),
                dtype=np.float32
            )
            # print(f"Observation space initialized with size {obs_size}")

    def pos_to_map_pos(self, position):
        """
        Convert world position coordinates to grid cell coordinates
        Args:
            position: [yaw, x, y] in world coordinates
        Returns:
            [yaw, grid_x, grid_y] with grid coordinates relative to the map
        """
        yaw, x, y = position

        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        # Check if map info is available
        if self.map_raw is None:
            print(
                "________________________ NO MAP RAW USING NORMAL POS IF HAPPENS HORRIBLE BUG BUT ONCE IS PROBABLY FINE ________________________")
            return [sin_yaw, cos_yaw, x, y]  # Return position with normalized yaw if no map info

        # Get map metadata
        resolution = self.map_raw.info.resolution
        origin_x = self.map_raw.info.origin.position.x
        origin_y = self.map_raw.info.origin.position.y

        # Convert to grid cell coordinates relative to the map origin
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        # Only use cropped map info if map_processed is already initialized
        if self.map_processed and self.center_cell_x is not None and self.center_cell_y is not None:
            # Calculate crop boundaries
            crop_size_cells = int(np.sqrt(len(self.map_processed)))
            half_size = crop_size_cells // 2
            min_x = max(0, self.center_cell_x - half_size)
            min_y = max(0, self.center_cell_y - half_size)

            # Adjust for map boundaries (same logic as in map_callback)
            width = self.map_raw.info.width
            height = self.map_raw.info.height
            if min_x + crop_size_cells > width:
                min_x = max(0, width - crop_size_cells)
            if min_y + crop_size_cells > height:
                min_y = max(0, height - crop_size_cells)

            # Adjust to coordinates within the cropped map
            grid_x = grid_x - min_x
            grid_y = grid_y - min_y

            # Ensure coordinates are within bounds of the cropped map
            grid_x = max(0, min(crop_size_cells - 1, grid_x))
            grid_y = max(0, min(crop_size_cells - 1, grid_y))

        # Return with grid position and normalized yaw
        return [sin_yaw, cos_yaw, float(grid_x), float(grid_y)]

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
        """Get the current state representation with position converted to grid cell coordinates"""
        # Use SLAM pose if available, otherwise fall back to odometry
        position = self.slam_pose if self.slam_pose is not None else self.pos

        # Convert position to map grid coordinates
        self.grid_position = self.pos_to_map_pos(position)

        if self.velocities is None:
            self.velocities = [0.0, 0.0]

        # Return state with grid position
        return self.grid_position + self.velocities + self.measured_distance_to_walls + self.map_processed

    def get_state_size(self):
        """Get the size of the state vector"""
        return len(self.get_state())

    def get_action_size(self):
        """Get the number of possible actions"""
        return len(self.actions)

    def update_visit_count(self):
        """
        Update:
            the visit count for the current cell
            if the cell was visited more than a short time ago (a few seconds) it resets its count
        Args:
            param dt
        """
        # Initialize visit count map if it doesn't exist yet
        if self.visit_count_map is None:
            # If we have a processed map, create a matching visit count map
            if self.map_processed:
                # Determine dimensions of the map
                crop_size = int(np.sqrt(len(self.map_processed)))  # if map is square
                # Create 3D array to track visit counts and times
                # [0] = visit count, [1] = last visit time
                self.visit_count_map = np.zeros((crop_size, crop_size, 2), dtype=float)
                print(f"Initialized visit count map with size {crop_size}x{crop_size}")
            else:
                print("Cannot initialize visit count map - no processed map available")
                return

        # Extract grid coordinates
        _, _, grid_x, grid_y = self.grid_position

        # Convert to integers for array indexing
        grid_x_int = int(grid_x)
        grid_y_int = int(grid_y)

        crop_size = self.visit_count_map.shape[0]
        if 0 <= grid_x_int < crop_size and 0 <= grid_y_int < crop_size:
            # check time since last
            current_time = time.time()
            last_visit_time = self.visit_count_map[grid_x_int, grid_y_int, 1]
            time_since_last_visit = current_time - last_visit_time
            if time_since_last_visit > REMEMBER_VISIT_TIME:
                self.visit_count_map[grid_x_int, grid_y_int, 0] = 0
            self.visit_count_map[grid_x_int, grid_y_int, 1] = current_time

            # Increment visit count for this cell with a 20 visits cap
            visits = self.visit_count_map[grid_x_int, grid_y_int, 0]
            if visits < 25:
                visits += 1
                self.visit_count_map[grid_x_int, grid_y_int, 0] = visits

    def calculate_revisit_penalty(self, dt):
        """
        Calculate penalty for revisiting cells that were visited a short time ago (a few seconds)
        Returns:
            Penalty value (negative reward)
        """
        self.update_visit_count()

        # Check if visit map exists
        if self.visit_count_map is None:
            return 0.0

        # Extract grid coordinates
        _, _, grid_x, grid_y = self.grid_position

        # Convert to integers for array indexing
        grid_x_int = int(grid_x)
        grid_y_int = int(grid_y)

        # Default penalty
        penalty = 0.0

        # Ensure coordinates are within bounds
        crop_size = self.visit_count_map.shape[0]
        if 0 <= grid_x_int < crop_size and 0 <= grid_y_int < crop_size:
            # Get visit count for this cell
            visits = self.visit_count_map[grid_x_int, grid_y_int, 0]

            # Only penalize cells visited more than once
            if visits > 1:
                # Apply increasing penalty for repeated visits
                # Penalty grows with each revisit, but with diminishing returns
                penalty = REVISIT_PENALTY * visits * dt
                self.last_revisit_penalty = penalty

        return penalty

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
            if distance_moved > (LINEAR_SPEED-0.05) * dt:  # threshold
                movement_reward = MOVEMENT_REWARD * (distance_moved * 100) * dt
                reward += movement_reward
                # Store for visualization
                self.last_movement_reward = movement_reward

        # Store current position for next comparison
        self.last_position = self.pos.copy()

        return reward

    def percent_explored(self):
        if self.total_cells is None or not hasattr(self, 'map_processed') or len(self.map_processed) == 0:
            return 0.0
        known_cells = sum(1 for val in self.map_processed if val != -1.0)
        return known_cells / self.total_cells

    def scale_distance_by_scan_angle(self, scan_distance, scan_idx, num_sectors=16):
        """
        Adjust the measured distance based on scan angle to make the safety distance
        more uniform around the robot despite the lidar being at the back.
        Args:
            scan_distance: Raw distance from lidar scan
            scan_idx: Index of the scan in the array (0 to num_sectors-1)
            num_sectors: Total number of lidar sectors
        Returns:
            Adjusted distance that provides consistent safety margins
        """
        # Calculate the angle in degrees for easier understanding
        robot_angle_degrees = (scan_idx * 360 / num_sectors)

        # Calculate scaling factors based on front distance (which stays at 1.0)
        front_scale = 1.0
        side_scale = 1.79  # makes side readings comparable to front
        back_scale = 2.10  # makes back readings comparable to front

        # Determine scaling factor based on the robot-relative angle
        if 330 <= robot_angle_degrees or robot_angle_degrees <= 30:  # Front section (0° ±30°)
            scale_factor = front_scale
        elif 150 <= robot_angle_degrees <= 210:  # Back section (180° ±30°)
            scale_factor = back_scale
        else:
            # Side sections
            if 30 < robot_angle_degrees < 150:  # Right side of robot
                if robot_angle_degrees < 90:  # Front-right quadrant
                    # Interpolate from front to side
                    factor = (robot_angle_degrees - 30) / 60
                    scale_factor = front_scale + (side_scale - front_scale) * factor
                else:  # Back-right quadrant
                    # Interpolate from side to back
                    factor = (robot_angle_degrees - 90) / 60
                    scale_factor = side_scale + (back_scale - side_scale) * factor
            else:  # Left side of robot (210° to 330°)
                if robot_angle_degrees < 270:  # Back-left quadrant
                    # Interpolate from back to side
                    factor = (robot_angle_degrees - 210) / 60
                    scale_factor = back_scale - (back_scale - side_scale) * factor
                else:  # Front-left quadrant
                    # Interpolate from side to front
                    factor = (robot_angle_degrees - 270) / 60
                    scale_factor = side_scale - (side_scale - front_scale) * factor

        # Actually apply the scaling factor to the distance
        adjusted_distance = scan_distance * scale_factor

        # if self.step_counter % 20 == 0:
        #     print(f"  Angle {robot_angle_degrees:.1f}°: {scan_distance:.2f}m → {adjusted_distance:.2f}m (scale: {scale_factor:.2f})")

        return adjusted_distance

    def dis_to_wall_to_punishment(self, dt, new_dis):
        """
        Calculate punishment based on adjusted distances to walls
        """
        # Apply distance adjustments based on scan angles
        adjusted_distances = []
        for idx, distance in enumerate(new_dis):
            adjusted = self.scale_distance_by_scan_angle(distance, idx, len(new_dis))
            adjusted_distances.append(adjusted)

        closest = min(adjusted_distances)
        is_terminated = False

        # Define the danger zone
        danger_zone_start = self.rad_of_robot * 2  # Start punishment from 2x radius
        danger_zone_end = self.rad_of_robot  # Max punishment at actual collision

        # Check for immediate collision
        if closest < self.rad_of_robot:
            self.last_wall_punishment = HIT_WALL_PUNISHMENT
            return HIT_WALL_PUNISHMENT, True

        # If outside danger zone, no punishment
        if closest >= danger_zone_start:
            self.last_wall_punishment = 0
            return 0, False

        # In danger zone - normalize to 0-1 range
        # How deep into the danger zone are we (0 = just entered, 1 = at collision boundary)
        danger_fraction = (danger_zone_start - closest) / (danger_zone_start - danger_zone_end)

        # When danger_fraction is small (far from wall): punishment is very small
        # When danger_fraction is near 1 (close to wall): punishment increases rapidly
        punishment_factor = danger_fraction ** WALL_POWER

        # Scale to punishment range
        punishment = punishment_factor * HIT_WALL_PUNISHMENT * CLOSE_TO_WALL_PUNISHMENT * dt

        # Clip to maximum punishment
        punishment = max(punishment, HIT_WALL_PUNISHMENT)

        # Store for visualization
        self.last_wall_punishment = punishment

        return punishment, is_terminated

    def change_in_map_to_reward(self, new_map):
        """Calculate reward based on newly discovered map cells"""
        # Skip if we don't have a previous map to compare
        if self.previous_map is None:
            self.previous_map = new_map.copy()
            self.last_exploration_reward = 0
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

        # Store for visualization
        self.last_exploration_reward = reward

        return reward

    def check_time_and_map_completion(self):
        # Check map exploration condition
        explored_percent = self.percent_explored()
        if explored_percent >= self.explored_threshold:
            print(f"Terminating: {explored_percent * 100:.2f}% of map explored (target: {self.explored_threshold * 100}%)")
            return True

        # Check time-based termination
        elapsed = time.time() - self.episode_start_time
        if elapsed > self.max_episode_duration:
            print(f"Terminating: episode ran for {elapsed:.1f}s (max: {self.max_episode_duration}s)")
            return True

        return False

    def calc_reward(self, time_from_last_env_update, new_dis, new_map):
        """Calculate reward based on time spent, proximity to walls, and exploration"""
        # Time-based continuous punishment
        cont = CONTINUES_PUNISHMENT * time_from_last_env_update
        self.last_cont_punishment = cont
        reward = cont

        # Punishment for being close to walls
        pun, is_terminated = self.dis_to_wall_to_punishment(time_from_last_env_update, new_dis)
        reward += pun

        # Reward for exploring new areas
        exploration_reward = self.change_in_map_to_reward(new_map)
        reward += exploration_reward

        # Reward for moving to not stay in place
        movement_reward = self.movement_to_reward(time_from_last_env_update)
        reward += movement_reward

        # Add penalty for revisiting cells
        revisit_penalty = self.calculate_revisit_penalty(time_from_last_env_update)
        reward += revisit_penalty

        # Check if finished by episode timeout or by map completion
        trunc = self.check_time_and_map_completion()

        # Store total reward for visualization
        self.last_total_reward = reward

        # Update reward visualizer with new data
        self.reward_vis.add_reward_data(
            self.last_cont_punishment,
            self.last_wall_punishment,
            self.last_exploration_reward,
            self.last_movement_reward,
            self.last_revisit_penalty,
            self.last_total_reward
        )

        if is_terminated or trunc:
            # Stop the robot when it hits a wall or episode ends
            print("is_terminated: ", is_terminated, ", is_truncated: ", trunc)
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            self.last_position = None

            return reward, True

        return reward, is_terminated

    def reset(self):
        """Reset the environment - in Gazebo this would typically involve resetting the simulation"""
        # Reset flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.slam_pose_ready = False
        self.should_update_center = True

        self.map_processed = None
        self.pos = None
        self.slam_pose = None
        self.grid_position = None
        self.velocities = None

        # Send stop command
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # IMPORTANT: Explicitly reset the previous map to ensure exploration rewards start fresh
        self.previous_map = None
        self.visit_count_map = None
        self.last_position = None
        self.map_raw = None
        self.visit_count_map = None
        self.center_cell_x = None
        self.center_cell_y = None
        self.current_odom = None
        self.should_update_center = True

        # Reset reward visualization
        self.reward_vis.reset_data()

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
        reward, terminated = self.gazebo_env.calc_reward(
            0.1,  # Fixed time step of 0.1 seconds
            self.gazebo_env.measured_distance_to_walls,
            self.gazebo_env.map_processed
        )

        # Track episode reward
        self.current_episode_reward += reward
        self.step_count += 1
        self.gazebo_env.step_counter = self.step_count

        # Prepare info dictionary for reset handler
        info = {
            'percent_explored': self.gazebo_env.percent_explored(),
            'step_count': self.step_count,
            'episode_reward': self.current_episode_reward
        }

        # Let the reset handler evaluate if we need a reset
        truncated = False

        # Check if this is a self-termination (e.g., time limit or map completion)
        if not terminated:
            truncated = self.gazebo_env.check_time_and_map_completion()

        # Only call the reset handler if something ended the episode
        is_done = terminated or truncated
        if is_done:
            self.reset_handler.update(terminated, truncated)

        return new_state, reward, terminated, truncated, info

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