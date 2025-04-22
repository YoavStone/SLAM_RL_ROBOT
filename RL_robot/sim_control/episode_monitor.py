import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from slam_toolbox.srv import Reset
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist, Vector3
import random
import math
import time
import subprocess
from nav_msgs.msg import Odometry


class EpisodeMonitor(Node):
    def __init__(self):
        super().__init__('episode_monitor')

        # Declare parameters with default values
        self.declare_parameter('spawn_location', '')  # Default: empty string means random

        self.is_resetting = False
        self.reset_count = 0
        self.last_reset_time = time.time()

        # Teleport attempt tracking
        self.teleport_attempt_count = 0
        self.max_teleport_attempts = 10
        self.correction_timeout = 0

        # Timer references
        self.teleport_timer = None
        self.check_timer = None
        self.retry_check_timer = None
        self.retry_teleport_timer = None

        # Current odometry state
        self.current_odom = None
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Get parameter values
        self.spawn_location_str = self.get_parameter('spawn_location').get_parameter_value().string_value
        self.get_logger().info(f"Received parameters: spawn_location='{self.spawn_location_str}'")

        # Robot model name in Gazebo
        self.model_name = 'mapping_robot'

        # Predefined possible random positions
        self.positions = [
            [0.0, 0.0],
            [6.3, 0.0],
            [-6.3, 0.0],
            [0.0, 6.3],
            [0.0, -6.3]
        ]

        # Get parameter values
        self.spawn_location_str = self.get_parameter('spawn_location').get_parameter_value().string_value
        self.get_logger().info(f"Received parameters: spawn_location='{self.spawn_location_str}'")

        # Parse spawn location if provided
        self.target_spawn_position = None
        if self.spawn_location_str:
            try:
                # Try to parse as "x,y" format
                x, y = self.spawn_location_str.split(',')
                self.target_spawn_position = [float(x.strip()), float(y.strip())]
                self.get_logger().info(f"Using provided spawn location: {self.target_spawn_position}")
            except ValueError:
                self.get_logger().warn(
                    f"Could not parse spawn_location '{self.spawn_location_str}'. Using random position.")
                self.target_spawn_position = None

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

        # Publisher for cmd_vel
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

        # Reset command control timer
        self.control_timer = None

        # State for the correction phase
        self.correction_phase = 'position'  # 'position' or 'yaw'

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

    def odom_callback(self, msg):
        """Store the latest odometry data"""
        self.current_odom = msg

    def get_target_position(self):
        """Get the target position for teleportation"""
        yaw = random.choice([0.0, math.pi / 2, math.pi, math.pi * 3 / 2])
        if self.target_spawn_position is not None:
            return [self.target_spawn_position[0], self.target_spawn_position[1], yaw]
        else:
            # Choose a random position from predefined positions
            chosen_position = random.choice(self.positions)
            return [chosen_position[0], chosen_position[1], yaw]

    def teleport_robot(self):
        """Teleport the robot to target position after odom recalibration"""
        # Get target x, y coordinates
        target_x, target_y, yaw = self.get_target_position()
        print(f"Teleporting robot to: ({target_x}, {target_y})")
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        print(f"Using quaternion: w={qw}, z={qz} for yaw={yaw}")


        try:
            # Format the command for Gazebo Harmonic
            cmd = [
                'gz', 'service', '-s', '/world/empty/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req',
                f'name: "{self.model_name}", position: {{x: {target_x}, y: {target_y}, z: 0.0}}, orientation: {{w: {qw}, x: 0.0, y: 0.0, z: {qz}}}'
            ]

            print(f"Executing command: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            try:
                stdout, stderr = process.communicate(timeout=2.5)
                output = stdout.decode() if stdout else ""
                error = stderr.decode() if stderr else ""

                if error:
                    self.get_logger().warn(f"Command error: {error}")

                self.get_logger().info("Teleport successful")
                # Wait a moment for physics to settle
                time.sleep(1.0)

                # After teleportation, reset SLAM and finalize
                self.reset_slam_and_finalize()
                return True

            except subprocess.TimeoutExpired:
                process.kill()
                self.get_logger().warn("Teleport command timed out")
                # Still try to reset SLAM and finalize even if teleport fails
                self.reset_slam_and_finalize()
                return False

        except Exception as e:
            self.get_logger().error(f"Error in teleport method: {e}")
            # Still try to reset SLAM and finalize even if teleport fails
            self.reset_slam_and_finalize()
            return False

    def start_odom_correction(self):
        """Start the control loop to get robot back to zero odom"""
        self.get_logger().info("Starting odom correction control loop")

        # Stop any existing control timer
        if self.control_timer:
            self.control_timer.cancel()

        # First send a stop command to ensure the robot is stationary
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        time.sleep(0.5)

        # Start with position correction phase
        self.correction_phase = 'position'

        # Set timeout for the odom correction
        self.correction_timeout = time.time() + 30.0  # 30 seconds timeout

        # Start the control loop
        self.control_timer = self.create_timer(0.1, self.correction_control_loop)

    def get_current_yaw(self):
        """Extract yaw angle from quaternion orientation"""
        if self.current_odom is None:
            return 0.0

        # Calculate orientation from quaternion
        qx = self.current_odom.pose.pose.orientation.x
        qy = self.current_odom.pose.pose.orientation.y
        qz = self.current_odom.pose.pose.orientation.z
        qw = self.current_odom.pose.pose.orientation.w

        # Convert to Euler angles - yaw (around z-axis)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

    def correction_control_loop(self):
        """Control loop to drive robot back to zero position and yaw"""
        # Check for timeout
        if time.time() > self.correction_timeout:
            self.timer_callback('correction_timeout')
            return

        if self.current_odom is None:
            self.get_logger().warn("No odometry data received yet")
            return

        # Get current position
        current_x = self.current_odom.pose.pose.position.x
        current_y = self.current_odom.pose.pose.position.y
        current_yaw = self.get_current_yaw()

        # Create command
        cmd = Twist()

        if self.correction_phase == 'position':
            # Calculate distance to origin
            distance = math.sqrt(current_x ** 2 + current_y ** 2)

            # If we're close enough to the origin, move to yaw correction phase
            if distance < 0.05:
                self.get_logger().info(f"Reached origin position: ({current_x:.3f}, {current_y:.3f})")

                # Switch to yaw correction phase
                self.correction_phase = 'yaw'

                # Stop the robot before yaw correction
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                time.sleep(0.5)

                # Reset the correction timeout for the yaw phase
                self.correction_timeout = time.time() + 15.0  # 15 seconds for yaw correction

                return

            # Calculate angle to origin
            angle_to_origin = math.atan2(-current_y, -current_x)

            # Calculate angular difference
            angle_diff = angle_to_origin - current_yaw

            # Normalize to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # First focus on turning to face the target
            # Only start moving forward when the robot is approximately facing the target
            if abs(angle_diff) > 0.2:  # About 11.5 degrees
                # Only turn, no forward movement
                cmd.linear.x = 0.0
                cmd.angular.z = angle_diff * 0.5  # Proportional control

                # Limit angular velocity for turning in place
                if abs(cmd.angular.z) > 0.5:
                    cmd.angular.z = 0.5 if cmd.angular.z > 0 else -0.5

                # self.get_logger().info(f"Turning to face target. Position: ({current_x:.2f}, {current_y:.2f}), " + f"Angle diff: {angle_diff:.2f}, Current yaw: {current_yaw:.2f}")
            else:
                # Now that we're facing the target, move forward
                linear_speed = 0.5 * distance  # Lower speed for more precision
                cmd.linear.x = linear_speed

                # Still apply minor angular corrections while moving
                cmd.angular.z = angle_diff * 0.5

                # Limit angular velocity
                if abs(cmd.angular.z) > 0.5:
                    cmd.angular.z = 0.5 if cmd.angular.z > 0 else -0.5

            # self.get_logger().info(f"Position: ({current_x:.2f}, {current_y:.2f}), Distance: {distance:.2f}, " + f"Angle: {angle_to_origin:.2f}, Current yaw: {current_yaw:.2f}, Diff: {angle_diff:.2f}")

        elif self.correction_phase == 'yaw':
            # Yaw correction phase - rotate to yaw=0

            # If we're close enough to zero yaw, stop
            if abs(current_yaw) < 0.02:
                self.get_logger().info(f"Reached target yaw: {current_yaw:.3f}")

                # Stop the robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)

                # Cancel the timer
                self.control_timer.cancel()
                self.control_timer = None

                # Once odom is corrected, teleport the robot to the target position
                self.teleport_robot()
                return

            # Set only angular velocity for pure rotation
            cmd.linear.x = 0.0
            cmd.angular.z = -current_yaw * 0.5  # Proportional control

            # Limit angular velocity
            if abs(cmd.angular.z) > 0.3:
                cmd.angular.z = 0.3 if cmd.angular.z > 0 else -0.3

            # self.get_logger().info(f"Yaw correction: current_yaw={current_yaw:.2f}, cmd.angular.z={cmd.angular.z:.2f}")

        # Publish command
        self.cmd_vel_pub.publish(cmd)

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

        self.get_logger().info("📩 Episode ended signal received — starting reset process")

        self.is_resetting = True
        try:
            # Start with odom correction and then teleport
            self.reset_environment()
        finally:
            self.is_resetting = False

    def reset_environment(self):
        """Reset the environment by first correcting odometry then teleporting"""
        # Directly start the odom correction (no teleport first)
        self.start_odom_correction()

    def timer_callback(self, action_type):
        """Callback for various timers"""
        if action_type == 'correction_timeout':
            # Handle timeout in correction loop
            self.get_logger().warn("Odom correction timeout reached, proceeding to teleport anyway")

            # Cancel the control timer
            if self.control_timer:
                self.control_timer.cancel()
                self.control_timer = None

            # Even if correction fails, still attempt to teleport and continue
            self.teleport_robot()

    def reset_slam_and_finalize(self):
        """Reset SLAM map and finalize the reset process after teleportation"""
        # Reset the SLAM map after robot is at the desired position
        time.sleep(2.0)

        self.get_logger().info("Robot at target position. Now resetting SLAM map...")
        map_reset = self.reset_slam_map()
        time.sleep(5.0)

        # After SLAM reset, finalize the process
        self.finalize_reset()

    def finalize_reset(self):
        """Finalize the reset process after position correction and SLAM reset"""
        # Publish simulation reset notification
        self.get_logger().info("Publishing simulation reset notification")
        self.sim_reset_pub.publish(Empty())

        time.sleep(1.0)

        # Signal reset complete
        self.reset_complete_pub.publish(Empty())
        self.get_logger().info("Reset process completed")

    def shutdown_hook(self):
        """Clean up resources when the node is shutting down."""
        self.get_logger().info("Shutting down EpisodeMonitor node")
        # Cancel all timers
        if self.control_timer:
            self.control_timer.cancel()
        if hasattr(self, 'teleport_timer') and self.teleport_timer:
            self.teleport_timer.cancel()
        if hasattr(self, 'check_timer') and self.check_timer:
            self.check_timer.cancel()
        if hasattr(self, 'retry_check_timer') and self.retry_check_timer:
            self.retry_check_timer.cancel()
        if hasattr(self, 'retry_teleport_timer') and self.retry_teleport_timer:
            self.retry_teleport_timer.cancel()


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