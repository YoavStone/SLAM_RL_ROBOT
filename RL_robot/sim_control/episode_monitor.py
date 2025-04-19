#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from slam_toolbox.srv import Reset
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
import random
import math
import time
import subprocess
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster



class EpisodeMonitor(Node):
    def __init__(self):
        super().__init__('episode_monitor')

        # Declare parameters with default values
        self.declare_parameter('spawn_location', '')  # Default: empty string means random

        self.is_resetting = False
        self.reset_count = 0
        self.last_reset_time = time.time()

        # Get parameter values
        self.spawn_location_str = self.get_parameter('spawn_location').get_parameter_value().string_value
        self.get_logger().info(f"Received parameters: spawn_location='{self.spawn_location_str}'")

        # Robot model name in Gazebo
        self.model_name = 'mapping_robot'

        # Predefined possible random positions
        self.positions = [
            (0.0, 0.0),
            (6.3, 0.0),
            (-6.3, 0.0),
            (0.0, 6.3),
            (0.0, -6.3)
        ]

        # Flag to prevent concurrent reset operations
        self.is_resetting = False

        # Subscribe to episode end signals
        self.subscription = self.create_subscription(
            Empty,
            'episode_end',
            self.episode_callback,
            10
        )

        # Publisher for simulation reset notifications (for DQN agent)
        self.sim_reset_pub = self.create_publisher(
            Empty,
            'simulation_reset',
            10
        )

        # Publisher for cmd_vel (fallback reset method)
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.reset_complete_pub = self.create_publisher(
            Empty,
            'reset_complete',
            10
        )

        # Initialize service clients after a short delay to ensure services are up
        time.sleep(2.0)

        # Client for Gazebo's SetModelState service
        self.set_model_state_client = self.create_client(
            SetModelState,
            '/gazebo/set_model_state'
        )

        # Client for SLAM Toolbox's reset service
        self.clear_slam_map_client = self.create_client(
            Reset,
            '/slam_toolbox/reset'
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Episode Monitor initialized")

        # Initial wait for services
        if not self.set_model_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Gazebo set_model_state service not available on initialization")
        else:
            self.get_logger().info("Gazebo set_model_state service is available")

        if not self.clear_slam_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("SLAM Toolbox reset service not available on initialization")
        else:
            self.get_logger().info("SLAM Toolbox reset service is available")

    def get_random_pose(self):
        """Get a random position and orientation from the predefined list."""
        x, y = random.choice(self.positions)
        yaw = 0.0
        return x, y, yaw

    def parse_spawn_location(self):
        """Parses the spawn_location parameter string 'x,y'."""
        try:
            parts = self.spawn_location_str.split(',')
            if len(parts) == 2:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                yaw = 0.0  # Fixed orientation
                self.get_logger().info(f"Using specified spawn location: x={x}, y={y}")
                return x, y, yaw
            else:
                self.get_logger().warn(
                    f"Invalid format for spawn_location parameter: '{self.spawn_location_str}'. Expected 'x,y'. Falling back to random.")
                return None
        except ValueError:
            self.get_logger().warn(
                f"Could not parse spawn_location parameter: '{self.spawn_location_str}' into floats. Falling back to random.")
            return None
        except Exception as e:
            self.get_logger().error(f"Error parsing spawn_location parameter: {e}. Falling back to random.")
            return None

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0, 0, 0, 0]
        q[0] = sr * cp * cy - cr * sp * sy
        q[1] = cr * sp * cy + sr * cp * sy
        q[2] = cr * cp * sy - sr * sp * cy
        q[3] = cr * cp * cy + sr * sp * sy

        return q

    def reset_robot_position(self):
        """Reset robot position using direct model state command."""
        # Determine position
        if self.spawn_location_str:
            pose_data = self.parse_spawn_location()
            if pose_data is None:
                x, y, yaw = self.get_random_pose()
            else:
                x, y, yaw = pose_data
        else:
            x, y, yaw = self.get_random_pose()

        self.get_logger().info(f"Attempting to teleport robot to x={x:.2f}, y={y:.2f}")

        # First stop the robot completely
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Send it multiple times to ensure it's received
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.05)

        try:
            # For Gazebo Harmonic, use CLI command
            cmd = [
                'gz', 'service', '-s', '/world/empty/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'name: "{self.model_name}", position: {{x: {x}, y: {y}, z: 0.05}}, orientation: {{w: 1.0}}'
            ]

            self.get_logger().info(f"Executing command: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            try:
                stdout, stderr = process.communicate(timeout=2.5)
                output = stdout.decode() if stdout else ""
                error = stderr.decode() if stderr else ""

                if error:
                    self.get_logger().warn(f"Command error: {error}")

                # Sleep to let physics stabilize
                time.sleep(0.5)
                return True
            except subprocess.TimeoutExpired:
                process.kill()
                self.get_logger().warn("Teleport command timed out")
                return False
        except Exception as e:
            self.get_logger().error(f"Error in teleport method: {e}")
            return False

    # Update your reset_slam_map method:
    def reset_slam_map(self):
        """Reset the SLAM map using the slam_toolbox reset service."""
        # First, publish a temporary map->odom transform
        temp_transform = TransformStamped()
        temp_transform.header.stamp = self.get_clock().now().to_msg()
        temp_transform.header.frame_id = "map"
        temp_transform.child_frame_id = "odom"

        # Identity transform
        temp_transform.transform.translation.x = 0.0
        temp_transform.transform.translation.y = 0.0
        temp_transform.transform.translation.z = 0.0
        temp_transform.transform.rotation.x = 0.0
        temp_transform.transform.rotation.y = 0.0
        temp_transform.transform.rotation.z = 0.0
        temp_transform.transform.rotation.w = 1.0

        # Publish transform continuously during reset
        publish_transform = True

        # Start a separate thread to publish the transform
        import threading
        def keep_transform_alive():
            while publish_transform:
                temp_transform.header.stamp = self.get_clock().now().to_msg()
                self.tf_broadcaster.sendTransform(temp_transform)
                time.sleep(0.01)

        transform_thread = threading.Thread(target=keep_transform_alive)
        transform_thread.start()

        # Now try to reset SLAM
        if not self.clear_slam_map_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("SLAM reset service not available")
            publish_transform = False
            transform_thread.join()
            return False

        req = Reset.Request()
        future = self.clear_slam_map_client.call_async(req)

        # Wait for result with timeout
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stop publishing the temporary transform
        publish_transform = False
        transform_thread.join()

        if future.done():
            self.get_logger().info("SLAM map reset via service call succeeded")
            return True
        else:
            self.get_logger().error("Failed to reset SLAM map via service call (timeout)")
            return False

    def episode_callback(self, msg):
        """Handle episode end signal."""
        # Check for too-frequent resets
        current_time = time.time()
        time_since_last = current_time - self.last_reset_time

        if time_since_last < 1.0:  # Less than 1 second since last reset
            self.reset_count += 1
            if self.reset_count > 3:  # If we get more than 3 rapid signals
                self.get_logger().warn(f"Too many resets in quick succession ({self.reset_count}), adding delay")
                time.sleep(2.0)  # Longer cooldown
                self.reset_count = 0
        else:
            self.reset_count = 0  # Reset the counter if enough time has passed

        self.last_reset_time = current_time

        # Prevent concurrent resets
        if self.is_resetting:
            self.get_logger().info("Reset already in progress, ignoring episode_end signal")
            return

        self.get_logger().info("📩 Episode ended signal received — resetting robot")

        self.is_resetting = True
        try:
            self.reset_environment()
        finally:
            time.sleep(0.5)
            self.is_resetting = False

    def reset_environment(self):
        """Reset the robot position and SLAM map."""
        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        time.sleep(0.5)

        # Reset robot position
        position_reset = self.reset_robot_position()

        # Longer delay to let physics stabilize
        self.get_logger().info("Waiting for robot position reset to stabilize...")
        time.sleep(2.0)

        # Reset SLAM map
        self.get_logger().info("Attempting to reset SLAM map...")
        map_reset = self.reset_slam_map()

        # Short delay
        time.sleep(1.0)

        # Allow more time for SLAM to reset
        self.get_logger().info("Waiting for SLAM map reset to complete...")
        time.sleep(3.0)

        # Publish simulation reset notification
        self.get_logger().info("Publishing simulation reset notification")
        self.sim_reset_pub.publish(Empty())

        # Wait for everything to settle
        time.sleep(1.0)

        self.get_logger().info(f"Environment reset completed: position={position_reset}, map={map_reset}")

        # Signal completion
        self.reset_complete_pub.publish(Empty())

    def shutdown_hook(self):
        """Clean up resources when the node is shutting down."""
        self.get_logger().info("Shutting down EpisodeMonitor node")
        # Nothing special to clean up in this implementation


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down')
    finally:
        # Clean up
        node.shutdown_hook()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Episode Monitor shutdown complete")


if __name__ == '__main__':
    main()