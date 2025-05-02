import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
import numpy as np
import time
import math
from gymnasium import spaces

from .RewardCalculator import RewardCalculator
from visualizers.MapVisualizationNode import MapVisualizationNode


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
        self.map_raw = None
        self.center_cell_x = None
        self.center_cell_y = None
        self.current_odom = None
        self.should_update_center = True

        # Action space: stop, forward, back, right, left
        self.actions = [0, 1, 2, 3, 4]

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

        self.reward_calculator = RewardCalculator(LINEAR_SPEED, self.rad_of_robot)

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

        if self.reward_calculator.get_total_cells() is None:
            self.reward_calculator.set_total_cells(len(cropped_map))

        # print("updated map")
        if self.observation_space is None:
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

        # print("pos: ", self.grid_position)
        # print("vel: ", self.velocities)

        # Return state with grid position
        return self.grid_position + self.velocities + self.measured_distance_to_walls + self.map_processed

    def get_state_size(self):
        """Get the size of the state vector"""
        return len(self.get_state())

    def get_action_size(self):
        """Get the number of possible actions"""
        return len(self.actions)

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
        self.map_raw = None
        self.center_cell_x = None
        self.center_cell_y = None
        self.current_odom = None
        self.should_update_center = True

        # Wait for new data
        timeout = 5.0  # seconds
        start_time = time.time()

        while not (self.scan_ready and self.map_ready and (self.odom_ready or self.slam_pose_ready)):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                print("Timeout waiting for sensor data during reset")
                break

        self.last_update_time = time.time()

        self.reward_calculator.reward_reset()

        return self.get_state(), {}  # Return state and empty info dict (gym-like interface)