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
            'episode_end',  # Make sure this is not prefixed with '/'
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
                yaw = 0.0  # Random orientation
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

        try:
            # For Gazebo Harmonic, try a different command format
            import subprocess

            # Format the command for Gazebo Harmonic properly
            cmd = [
                'gz', 'service', '-s', '/world/empty/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'name: "{self.model_name}", position: {{x: {x}, y: {y}, z: 0.05}}, orientation: {{w: 1.0}}'
            ]

            self.get_logger().info(f"Executing command: {' '.join(cmd)}")

            # Use non-blocking approach with proper timeout handling
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            try:
                stdout, stderr = process.communicate(timeout=2.5)
                output = stdout.decode() if stdout else ""
                error = stderr.decode() if stderr else ""

                if error:
                    self.get_logger().warn(f"Command error: {error}")

                # Even if there's an error, continue with the fallback
                # Just stop the robot via cmd_vel
                self.get_logger().info("Using cmd_vel to stop robot")
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)

                # Sleep a bit to let the stop command take effect
                time.sleep(0.5)

                return True
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                self.get_logger().warn("Teleport command timed out")

                # Fallback - stop the robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                time.sleep(0.5)

                return True
        except Exception as e:
            self.get_logger().error(f"Error in teleport method: {e}")
            # Fallback - just stop the robot
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.5)

            return True

    def reset_slam_map(self):
        """Reset the SLAM map using the slam_toolbox reset service."""
        if not self.clear_slam_map_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("SLAM reset service not available")
            return False

        req = Reset.Request()
        future = self.clear_slam_map_client.call_async(req)

        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.result() is not None:
            self.get_logger().info("SLAM map reset via service call succeeded")
            return True
        else:
            self.get_logger().error("Failed to reset SLAM map via service call")
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
        # Reset robot position
        position_reset = self.reset_robot_position()

        # Longer delay to let physics stabilize
        self.get_logger().info("Waiting for robot position reset to stabilize...")
        time.sleep(2.0)

        # Reset SLAM map
        self.get_logger().info("Attempting to reset SLAM map...")
        map_reset = self.reset_slam_map()

        # Allow more time for SLAM to reset
        self.get_logger().info("Waiting for SLAM map reset to complete...")
        time.sleep(3.0)

        # Publish simulation reset notification
        self.get_logger().info("Publishing simulation reset notification")
        self.sim_reset_pub.publish(Empty())

        self.get_logger().info(f"Environment reset completed: position={position_reset}, map={map_reset}")

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
        print("EpisodeMonitor shutdown complete")


if __name__ == '__main__':
    main()