import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import time
import math

from services.map_cropping_service import calc_map_center, crop_map
from services.lidar_data_filter_service import lidar_scan_filter
from services.grid_position_calculator import calc_grid_pos

from visualizers.MapVisualizationNode import MapVisualizationNode


class SensorsProcessor(Node):
    """
    ROS2 Node that interfaces with sensor readings for the gym-like environment
    """

    def __init__(self):
        super().__init__('sensors_processor')

        self.declare_parameter('spawn_location', '')  # Default: empty string means random

        # Get the spawn_location parameter
        self.spawn_location_str = self.get_parameter('spawn_location').get_parameter_value().string_value
        self.get_logger().info(f"Received parameters: spawn_location='{self.spawn_location_str}'")

        self.step_counter = 0

        # visualize the cropped map usually for debug
        print("Creating visualization node...")
        self.vis_node = MapVisualizationNode(publish=False)
        # Create timer to periodically publish the map
        self.pub_crop_timer = self.create_timer(1.0, self.publish_cropped_map)
        print("Visualization node created")

        # Environment state
        self.map_processed = []  # Processed map data for DQL input
        self.pos = None  # [orientation, x, y]
        self.velocities = None  # [vx, va]
        self.slam_pose = None  # Store the latest SLAM pose [orientation, x, y]
        self.grid_position = None  # stores position on grid [sin(x), cos(x), x, y]
        self.measured_distance_to_walls = [10.0] * 16  # distances in sixteenths of circle

        self.map_raw = None
        self.center_cell_x = None
        self.center_cell_y = None
        self.current_odom = None
        self.should_update_center = True

        # Action space: stop, forward, back, right, left
        self.actions = [0, 1, 2, 3, 4]

        # Publishers and subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.slam_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.slam_pose_callback, 10)

        # Data ready flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.slam_pose_ready = False

        print('Sensor Node initialized')

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

        self.measured_distance_to_walls = lidar_scan_filter(ranges, msg.range_max)

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
        self.map_raw = msg
        # Extract map metadata
        resolution = msg.info.resolution  # Typically 0.05m or similar. for me now its 0.15m
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Store the center position (robot starting position) if not already stored
        if self.should_update_center:
            self.center_cell_x, self.center_cell_y = calc_map_center(origin_x, origin_y, width, height, resolution, self.odom_ready, self.pos, self.slam_pose)
            # Reset the flag so we don't update center again until next reset
            self.should_update_center = False

        self.map_processed = crop_map(msg.data, width, height, resolution, self.center_cell_x, self.center_cell_y)
        self.map_ready = True

        # After processing the map, update visualization
        self.vis_node.set_map(self.map_processed, resolution)

        # print("updated map")

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
            print("________________________ NO MAP RAW USING NORMAL POS IF HAPPENS HORRIBLE BUG BUT ONCE IS PROBABLY FINE ________________________")
            return [sin_yaw, cos_yaw, 20, 20]  # Return position with normalized yaw if no map info

        # Get map metadata
        resolution = self.map_raw.info.resolution
        origin_x = self.map_raw.info.origin.position.x
        origin_y = self.map_raw.info.origin.position.y

        # Convert to grid cell coordinates relative to the map origin
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        # Only use cropped map info if map_processed is already initialized
        if self.map_processed and self.center_cell_x is not None and self.center_cell_y is not None:
            crop_size_cells = int(np.sqrt(len(self.map_processed)))
            width = self.map_raw.info.width
            height = self.map_raw.info.height
            grid_x, grid_y = calc_grid_pos(position, grid_x, grid_y, self.center_cell_x, self.center_cell_y, width, height, crop_size_cells)

        # Return with grid position and normalized yaw
        return [sin_yaw, cos_yaw, grid_x, grid_y]

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
        """Reset the sensor processor members"""
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

        return self.get_state(), {}  # Return state and empty info dict (gym-like interface)