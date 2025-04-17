import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, TransformStamped, PoseWithCovarianceStamped
from gymnasium import spaces
import numpy as np
import time
import math
import torch
import random
import tf2_ros
from std_msgs.msg import Empty as EmptyMsg

# Constants
CONTINUES_PUNISHMENT = -2  # amount of punishment for every sec wasted
HIT_WALL_PUNISHMENT = -200
CLOSE_TO_WALL_PUNISHMENT = -0.1
EXPLORATION_REWARD = 1.0

LINEAR_SPEED = 0.3  # irl: 0.3  # m/s
ANGULAR_SPEED = 0.3 * 2  # irl: 0.3  # rad/s

# Predefined spawn positions
SPAWN_POSITIONS = [
    (0.0, 0.0),  # center
    (6.3, 0.0),  # right
    (-6.3, 0.0),  # left
    (0.0, 6.3),  # top
    (0.0, -6.3)  # bottom
]


class GazeboEnv(Node):
    """ROS2 Node that interfaces with Gazebo and provides a gym-like environment interface"""

    def __init__(self, rad_of_robot=0.34):
        super().__init__('gazebo_env_node')

        # Robot properties
        self.rad_of_robot = rad_of_robot * 1.3  # radius from lidar to tip with safety margin
        self.model_name = 'mapping_robot'  # name of the robot model in Gazebo

        # Environment state
        self.previous_map = None
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

        # Reset request publisher (to external reset handler)
        self.reset_request_pub = self.create_publisher(EmptyMsg, '/environment_reset_request', 10)

        # Publishers for robot teleportation
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # TF2 broadcaster for updating robot position in RViz
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Gym-like interface variables
        self.observation_space = None  # Will be initialized after first data is received
        self.action_space = spaces.Discrete(len(self.actions))

        # Data ready flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False

        # Store the latest map for checking changes
        self.latest_map = None
        self.episode_count = 0

        # Timer for environment update (0.1 second interval)
        self.timer = self.create_timer(0.1, self.update_timer_callback)

        print('Gazebo Environment Node initialized')

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
        yaw = msg.pose.pose.orientation.z

        self.pos = [yaw, x, y]
        self.odom_ready = True

    def map_callback(self, msg):
        """Process SLAM map data by cropping a 6m x 6m area centered on the robot's position"""
        # Store raw map data
        self.map_raw = msg
        self.latest_map = msg  # Store for reset purposes

        # Extract map metadata
        resolution = msg.info.resolution  # Typically 0.05m or similar. for me now its 0.15m
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Calculate size of 6m x 6m area in grid cells (maintaining resolution)
        crop_size_meters = 6.0  # 6m x 6m area
        crop_size_cells = int(crop_size_meters / resolution)

        # If robot position isn't available yet, use the center of the map
        if not self.odom_ready:
            robot_cell_x = width // 2
            robot_cell_y = height // 2
        else:
            # Convert robot position to grid cell coordinates
            robot_cell_x = int((self.pos[1] - origin_x) / resolution)
            robot_cell_y = int((self.pos[2] - origin_y) / resolution)

        # Calculate boundaries for cropping (ensuring we don't go out of bounds)
        half_size = crop_size_cells // 2
        min_x = max(0, robot_cell_x - half_size)
        min_y = max(0, robot_cell_y - half_size)
        max_x = min(width, robot_cell_x + half_size)
        max_y = min(height, robot_cell_y + half_size)

        # Calculate actual dimensions of cropped area
        actual_width = max_x - min_x
        actual_height = max_y - min_y

        # Crop the map
        cropped_map = []
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                idx = y * width + x
                if idx < len(msg.data):
                    # Keep unknown as -1, normalize 0-100 scale to 0-1
                    if msg.data[idx] == -1:  # Unknown
                        cropped_map.append(-1.0)
                    else:  # 0-100 scale to 0-1
                        cropped_map.append(float(msg.data[idx]) / 100.0)
                else:
                    # Out of bounds, mark as unknown
                    cropped_map.append(-1.0)

        # Ensure consistent size by padding or truncating
        target_size = crop_size_cells * crop_size_cells
        if len(cropped_map) < target_size:
            cropped_map.extend([-1.0] * (target_size - len(cropped_map)))
        elif len(cropped_map) > target_size:
            cropped_map = cropped_map[:target_size]

        self.map_processed = cropped_map
        self.map_ready = True

        # Initialize observation space if not already done
        if self.observation_space is None:
            obs_size = len(
                self.get_state())  # next lines define max and min of each value in the obs space. note: 4 and not math.pi because there is a chance for a slight error
            self.observation_space = spaces.Box(
                low=np.array([-4, -100, -100] + [0] * 8 + [-1] * len(self.map_processed)),
                high=np.array([4, 100, 100] + [10] * 8 + [1] * len(self.map_processed)),
                dtype=np.float32
            )
            print(f"Observation space initialized with size {obs_size}")

    def update_timer_callback(self):
        """Timer callback to update environment state at 10Hz (0.1 seconds)"""
        if self.scan_ready and self.odom_ready and self.map_ready:
            dt = time.time() - self.last_update_time
            self.last_update_time = time.time()

            # This would typically be called from the DQL agent's update loop
            new_state, reward, is_terminated = self.update_env(
                self.map_processed, self.pos, self.measured_distance_to_walls, dt
            )

            if is_terminated:
                print("Environment terminated (hit wall)")

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

    def dis_to_wall_to_punishment(self, dt, new_dis):
        punishment = 0
        closest = min(new_dis)
        is_terminated = False

        if closest < self.rad_of_robot:  # Robot is too close to a wall
            punishment += HIT_WALL_PUNISHMENT
            is_terminated = True
        elif closest < self.rad_of_robot * 1.3:  # if close but not too close slight punishment
            punishment = (CLOSE_TO_WALL_PUNISHMENT / (closest - self.rad_of_robot) ** 2) * dt

        return punishment, is_terminated

    def change_in_map_to_reward(self, new_map):
        """Calculate reward based on newly discovered map cells"""
        # Skip if we don't have a previous map to compare
        if self.previous_map is None:
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

    def calc_reward(self, time_from_last_env_update, new_dis, new_map):
        """Calculate reward based on time spent, proximity to walls, and exploration"""
        reward = CONTINUES_PUNISHMENT * time_from_last_env_update

        # punishment for being close to walls
        pun, is_terminated = self.dis_to_wall_to_punishment(time_from_last_env_update, new_dis)
        reward += pun

        if is_terminated:  # Robot is too close to a wall
            # Stop the robot when it hits a wall
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            if reward != CONTINUES_PUNISHMENT * 0.1:  # log if reward is not static
                print("reward_t: ", reward, "is_terminated: ", is_terminated)
            return reward, is_terminated

        # reward for exploring new areas
        exploration_reward = self.change_in_map_to_reward(new_map)
        reward += exploration_reward

        if reward != CONTINUES_PUNISHMENT * 0.1:  # log if reward is not static
            print("reward_t: ", reward, "is_terminated: ", is_terminated)

        return reward, is_terminated

    def update_env(self, new_map, new_pos, new_dis, dt=0.1):
        """Update environment state and calculate reward"""
        reward, is_terminated = self.calc_reward(dt, new_dis, new_map)
        self.map_processed = new_map
        self.pos = new_pos
        self.measured_distance_to_walls = new_dis
        new_state = self.get_state()
        return new_state, reward, is_terminated

    def request_slam_reset(self):
        """Request SLAM map reset via the external handler"""
        self.get_logger().info("Requesting SLAM map reset from handler")
        self.reset_request_pub.publish(EmptyMsg())
        # Wait a moment for reset to take effect
        time.sleep(1.0)
        self.previous_map = None  # Reset our internal map tracking
        return True

    def teleport_robot(self, x, y, yaw=0.0):
        """Teleport the robot by publishing to initialpose and using direct commands"""
        try:
            # Approach 1: Try using the gz command line tool for teleportation
            model_name = 'mapping_robot'  # Your robot's name in Gazebo
            try:
                import subprocess
                cmd = [
                    'gz', 'model', '-m', model_name,
                    '-p', f'{x},{y},0.05',  # x,y,z
                    '-o', f'0,0,{yaw}'  # roll,pitch,yaw
                ]
                subprocess.run(cmd, timeout=1.0)
                self.get_logger().info(f"Used gz command line to teleport to ({x}, {y}, {yaw})")
            except Exception as e:
                self.get_logger().warn(f"gz command failed: {e}, trying alternative methods")

            # Approach 2: Create PoseWithCovarianceStamped message
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = self.get_clock().now().to_msg()

            # Set position
            pose_msg.pose.pose.position.x = x
            pose_msg.pose.pose.position.y = y
            pose_msg.pose.pose.position.z = 0.0

            # Set orientation as quaternion
            pose_msg.pose.pose.orientation.z = math.sin(yaw / 2)
            pose_msg.pose.pose.orientation.w = math.cos(yaw / 2)

            # Set covariance (low uncertainty)
            for i in range(36):
                pose_msg.pose.covariance[i] = 0.0
            pose_msg.pose.covariance[0] = 0.001  # x
            pose_msg.pose.covariance[7] = 0.001  # y
            pose_msg.pose.covariance[35] = 0.001  # yaw

            # Publish multiple times to ensure it gets through
            for _ in range(10):
                self.initial_pose_pub.publish(pose_msg)
                time.sleep(0.05)

                # Also publish stop commands
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)

                # Update TF
                self.broadcast_tf(x, y, yaw)

            # Force odometry reset
            odom_msg = Odometry()
            odom_msg.header.frame_id = "odom"
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.child_frame_id = "base_link"

            # Set position
            odom_msg.pose.pose.position.x = x
            odom_msg.pose.pose.position.y = y
            odom_msg.pose.pose.position.z = 0.0

            # Set orientation
            odom_msg.pose.pose.orientation.z = math.sin(yaw / 2)
            odom_msg.pose.pose.orientation.w = math.cos(yaw / 2)

            # Set covariance (low uncertainty)
            for i in range(36):
                odom_msg.pose.covariance[i] = 0.0
            odom_msg.pose.covariance[0] = 0.001  # x
            odom_msg.pose.covariance[7] = 0.001  # y
            odom_msg.pose.covariance[35] = 0.001  # yaw

            # Set zero velocity
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0

            # Publish multiple times to ensure it gets through
            odom_pub = self.create_publisher(Odometry, 'odom', 10)
            for _ in range(5):
                odom_pub.publish(odom_msg)
                time.sleep(0.05)

            self.get_logger().info(f'Robot teleport request sent to ({x}, {y}, {yaw})')
            return True
        except Exception as e:
            self.get_logger().error(f'Error teleporting robot: {str(e)}')
            return False

    def broadcast_tf(self, x, y, yaw):
        """Broadcast TF transform for the new robot position"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        # Set translation
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        # Set rotation (quaternion)
        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def reset(self):
        """Reset the environment with improved teleportation and SLAM reset"""
        self.episode_count += 1
        self.get_logger().info(f"Resetting environment for episode {self.episode_count}...")

        # Send stop command to robot
        stop_cmd = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.05)

        # Request SLAM reset from external handler
        self.get_logger().info("Requesting SLAM map reset from handler")
        self.reset_request_pub.publish(EmptyMsg())

        # Wait for SLAM reset to complete
        time.sleep(1.0)

        # Select a random spawn position
        spawn_pos = random.choice(SPAWN_POSITIONS)
        x, y = spawn_pos
        yaw = random.uniform(-3.14, 3.14)  # Random orientation

        # Use the dedicated teleport publisher
        teleport_pub = self.create_publisher(Pose, '/teleport_robot', 10)

        # Create teleport message
        pose_msg = Pose()
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = 0.05
        pose_msg.orientation.z = math.sin(yaw / 2)
        pose_msg.orientation.w = math.cos(yaw / 2)

        # Publish teleport request multiple times to ensure delivery
        self.get_logger().info(f"Requesting teleportation to ({x}, {y}, {yaw})")
        for _ in range(5):
            teleport_pub.publish(pose_msg)
            time.sleep(0.1)

        # Force our state tracking to update, since teleportation might not work
        self.pos = [yaw, x, y]

        # Wait a moment for teleportation to take effect
        time.sleep(1.0)

        # Wait for sensor data
        timeout = 5.0  # Reduced timeout
        start_time = time.time()
        self.get_logger().info("Waiting for sensor data after reset...")

        # Reset flags to wait for fresh data
        self.scan_ready = False
        self.odom_ready = False
        # Don't wait for map_ready, it might take too long

        spinner_count = 0
        while not (self.scan_ready and self.odom_ready):  # Only wait for scan and odom
            rclpy.spin_once(self, timeout_sec=0.1)
            spinner_count += 1

            if spinner_count % 10 == 0:
                self.get_logger().info(
                    f"Still waiting for data: scan={self.scan_ready}, odom={self.odom_ready}, map={self.map_ready}")

            if time.time() - start_time > timeout:
                self.get_logger().warn("Timeout waiting for sensor data during reset")
                break

        # Even if teleportation didn't work, we'll set our internal state to where we want it
        # so the agent can learn as if teleportation worked
        self.last_update_time = time.time()
        self.get_logger().info("Environment reset complete")

        # Initialize empty map data if necessary
        if not self.map_ready and len(self.map_processed) == 0:
            map_size = self.observation_space.shape[0] - len(self.pos) - len(self.measured_distance_to_walls)
            self.map_processed = [-1.0] * map_size
            self.get_logger().warn(f"Using empty map data of size {map_size}")

        return self.get_state(), {}

    def close(self):
        """Clean up resources"""
        # Send stop command to robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)


class DQLEnv:
    """Adapter class that bridges between GazeboEnv and the DQL agent"""

    def __init__(self, rad_of_robot=0.34):
        # Initialize ROS node for environment
        # Note: rclpy.init() should be called before this from the main node
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