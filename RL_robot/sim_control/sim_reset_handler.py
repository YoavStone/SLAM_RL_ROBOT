import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from slam_toolbox.srv import Reset
from geometry_msgs.msg import Twist, Pose, Quaternion # Still needed for odom correction and internal pose representation
from nav_msgs.msg import Odometry
import random
import math
import time
from std_msgs.msg import String # Import the String message type

# Removed the placeholder TeleportPose class as we are using std_msgs/msg/String


class SimulationResetHandler:
    """
    Enhanced handler that publishes a String message to trigger robot teleportation
    by a separate service node, and performs odometry correction.
    This version works by publishing a message, decoupling the teleport logic.
    """

    def __init__(self, env):
        """
        Initialize the reset handler.

        Args:
            env: The environment object (expected to be a ROS 2 Node) that will be reset
        """
        self.is_first_ep = True

        self.env = env # The ROS 2 Node instance from the environment
        self.logger = env.get_logger() # Use the environment's logger
        self.is_resetting = False

        # Flag to prevent reset attempts before initial data (handled by waiting for odom)
        self.initialized = False

        # Directly use the parameters from EpisodeMonitor (or get them from env if available)
        # For this example, we'll hardcode or use defaults like the original
        self.spawn_location_str = ''  # Default: empty string means random
        self.target_spawn_position = None # Will be set if spawn_location_str is parsed

        # Reset tracking
        self.reset_count = 0
        self.last_reset_time = time.time()

        # Teleport attempt tracking (less relevant with this pattern)
        self.teleport_attempt_count = 0
        self.max_teleport_attempts = 10
        self.correction_timeout = 0

        # Timer references - managed by the handler, created via env node
        self.teleport_timer = None
        self.check_timer = None
        self.retry_check_timer = None
        self.retry_teleport_timer = None
        self.control_timer = None

        # State for the correction phase
        self.correction_phase = 'position'  # 'position' or 'yaw'

        # Current odometry state - will be updated from environment (env.current_odom)
        # Ensure your environment node updates env.current_odom from its odom subscription
        self.current_odom = None # This will be a reference to self.env.current_odom

        # Robot model name in Gazebo - match your naming (needed for the teleport service node)
        self.model_name = 'mapping_robot' # This will be passed in the message or used by the service node

        # Predefined possible random positions
        self.positions = [
            [0.0, 0.0],
            [6.3, 0.0],
            [-6.3, 0.0],
            [0.0, 6.3],
            [0.0, -6.3]
        ]

        # Publisher for the teleport command message
        # Topic name: /teleport_command (or choose a suitable name)
        # Message type: std_msgs/msg/String
        self.teleport_pub = self.env.create_publisher(
            String, # Use String message type
            '/teleport_command',
            10 # QoS history depth
        )
        self.logger.info("Created publisher for /teleport_command using String message")


        # Client for SLAM Toolbox's reset service
        self.clear_slam_map_client = self.env.create_client(
            Reset,
            '/slam_toolbox/reset'
        )

        # Removed wait_for_service from __init__ to allow the node to start faster.
        # The wait for SLAM reset will happen just before calling it.
        self.logger.info("Simulation reset handler initialized.")


    def update(self, terminated, truncated):
        """
        Update episode state and check if reset is needed.
        This method is called by the environment's main loop.

        Args:
            terminated: Whether episode terminated (e.g., collision)
            truncated: Whether episode truncated (e.g., time limit)
        Returns:
            bool: True if episode ended and reset started, False otherwise.
        """
        # Check if episode is done and if we are not already resetting
        is_done = terminated or truncated

        if is_done and not self.is_resetting:
            self.episode_callback(None) # Trigger the reset process
            return True # Indicate that a reset has started

        return False # Indicate that no reset was started

    def episode_callback(self, msg):
        """Handle episode end signal, just like in EpisodeMonitor."""
        # Check for too-frequent resets
        current_time = time.time()
        time_since_last = current_time - self.last_reset_time

        if time_since_last < 1.0:  # Less than 1 second since last reset
            self.reset_count += 1
            if self.reset_count > 3:  # If we get more than 3 rapid signals
                self.logger.warn(f"Too many resets in quick succession ({self.reset_count}), adding delay")
                time.sleep(2.0)  # Longer cooldown
                self.reset_count = 0
        else:
            self.reset_count = 0  # Reset the counter if enough time has passed

        self.last_reset_time = current_time

        # Prevent concurrent resets (already checked in update, but good defensive check)
        if self.is_resetting:
            self.logger.info("Reset already in progress, ignoring episode_end signal")
            return

        self.logger.info("📩 Episode ended signal received — starting reset process")

        self.is_resetting = True
        try:
            # Start with odom correction and then trigger teleport
            self.reset_environment()
        except Exception as e:
             self.logger.error(f"Error during reset process: {e}")
             # Ensure is_resetting is reset even on error
             self.is_resetting = False

    def reset_environment(self):
        """Reset the environment by first correcting odometry then triggering teleport"""
        # Ensure we have odometry data before proceeding with correction
        # The environment node should be updating self.env.current_odom
        if self.is_first_ep:
            self.is_first_ep = False
            self.reset_slam_and_finalize()
            return True

        if self.env.current_odom is None:
            self.logger.warn("Cannot start odom correction - no odometry data yet. Waiting...")
            # We could add a timer here to retry starting correction if odom is still None
            # For simplicity, we'll assume the env will eventually provide odom
            # If odom never arrives, the correction timeout will eventually trigger teleport.
            pass # Just log and wait for odom to appear in the correction loop

        self.logger.info("📩 Starting reset process - first correcting odometry")
        # Start with odom correction
        self.start_odom_correction()
        # Note: reset_environment now just *starts* the process. The teleport and finalization
        # happen within the correction control loop or its timeout handler.
        return True # Indicate that the reset process has been initiated

    def start_odom_correction(self):
        """Start the control loop to get robot back to zero odom"""
        self.logger.info("Starting odom correction control loop")

        # Stop any existing control timer
        if self.control_timer:
            self.control_timer.cancel()

        # First send a stop command to ensure the robot is stationary
        stop_cmd = Twist()
        self.env.cmd_vel_pub.publish(stop_cmd)
        time.sleep(0.5) # Small delay for command to take effect

        # Start with position correction phase
        self.correction_phase = 'position'

        # Set timeout for the odom correction
        self.correction_timeout = time.time() + 30.0  # 30 seconds timeout

        # Start the control loop timer using the environment's node
        self.control_timer = self.env.create_timer(0.1, self.correction_control_loop)

    def get_current_yaw(self):
        """Extract yaw angle from quaternion orientation"""
        # Use the odometry data provided by the environment node
        if self.env.current_odom is None:
            return 0.0

        # Calculate orientation from quaternion
        qx = self.env.current_odom.pose.pose.orientation.x
        qy = self.env.current_odom.pose.pose.orientation.y
        qz = self.env.current_odom.pose.pose.orientation.z
        qw = self.env.current_odom.pose.pose.orientation.w

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

        # Use the odometry data provided by the environment node
        if self.env.current_odom is None:
            self.logger.warn("No odometry data received yet for correction loop.")
            return

        # Get current position and yaw from odometry
        current_x = self.env.current_odom.pose.pose.position.x
        current_y = self.env.current_odom.pose.pose.position.y
        current_yaw = self.get_current_yaw()

        # Create command
        cmd = Twist()

        if self.correction_phase == 'position':
            # Calculate distance to origin
            distance = math.sqrt(current_x ** 2 + current_y ** 2)

            # If we're close enough to the origin, move to yaw correction phase
            if distance < 0.05: # Threshold for position
                self.logger.info(f"Reached origin position: ({current_x:.3f}, {current_y:.3f})")

                # Switch to yaw correction phase
                self.correction_phase = 'yaw'

                # Stop the robot before yaw correction
                stop_cmd = Twist()
                self.env.cmd_vel_pub.publish(stop_cmd)
                time.sleep(0.5) # Small delay

                # Reset the correction timeout for the yaw phase
                self.correction_timeout = time.time() + 15.0  # 15 seconds for yaw correction

                return # Exit this iteration to allow the sleep and phase switch

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
            if abs(angle_diff) > 0.2:  # About 11.5 degrees threshold for turning
                # Only turn, no forward movement
                cmd.linear.x = 0.0
                cmd.angular.z = angle_diff * 0.5  # Proportional control for turning

                # Limit angular velocity for turning in place
                if abs(cmd.angular.z) > 0.5:
                    cmd.angular.z = 0.5 if cmd.angular.z > 0 else -0.5
            else:
                # Now that we're facing the target, move forward
                linear_speed = 0.5 * distance  # Proportional control for linear speed
                 # Limit linear speed
                if abs(linear_speed) > 0.5:
                     linear_speed = 0.5 if linear_speed > 0 else -0.5
                cmd.linear.x = linear_speed

                # Still apply minor angular corrections while moving
                cmd.angular.z = angle_diff * 0.5
                # Limit angular velocity
                if abs(cmd.angular.z) > 0.5:
                    cmd.angular.z = 0.5 if cmd.angular.z > 0 else -0.5

        elif self.correction_phase == 'yaw':
            # Yaw correction phase - rotate to yaw=0

            # If we're close enough to zero yaw, stop
            if abs(current_yaw) < 0.02: # Threshold for yaw
                self.logger.info(f"Reached target yaw: {current_yaw:.3f}")

                # Stop the robot
                stop_cmd = Twist()
                self.env.cmd_vel_pub.publish(stop_cmd)

                # Cancel the timer, odom correction is complete
                self.control_timer.cancel()
                self.control_timer = None

                # Once odom is corrected, trigger teleportation
                self.teleport_robot()
                return # Exit this iteration

            # Set only angular velocity for pure rotation
            cmd.linear.x = 0.0
            cmd.angular.z = -current_yaw * 0.5  # Proportional control for yaw

            # Limit angular velocity
            if abs(cmd.angular.z) > 0.3:
                cmd.angular.z = 0.3 if cmd.angular.z > 0 else -0.3

        # Publish command if correction is still ongoing
        self.env.cmd_vel_pub.publish(cmd)

    def timer_callback(self, action_type):
        """Callback for various timers"""
        if action_type == 'correction_timeout':
            # Handle timeout in correction loop
            self.logger.warn("Odom correction timeout reached, proceeding to trigger teleport anyway")

            # Cancel the control timer
            if self.control_timer:
                self.control_timer.cancel()
                self.control_timer = None

            # Even if correction fails, still attempt to trigger teleport
            self.teleport_robot()

    def get_target_position(self):
        """Get the target position and orientation for teleportation"""
        # Choose a random yaw angle from the predefined set
        yaw = random.choice([0.0, math.pi / 2, math.pi, math.pi * 3 / 2])

        if self.target_spawn_position is not None:
            # Use the provided spawn location if available
            return [self.target_spawn_position[0], self.target_spawn_position[1], yaw]
        else:
            # Choose a random position from predefined positions
            chosen_position = random.choice(self.positions)
            return [chosen_position[0], chosen_position[1], yaw]


    def teleport_robot(self):
        """Publish a String message to trigger teleportation by a separate node."""
        # Get target x, y coordinates and yaw
        target_x, target_y, yaw = self.get_target_position()
        self.logger.info(f"Publishing teleport command to: ({target_x}, {target_y}) with yaw {yaw} as String")

        # Create and populate the String message with pose data
        teleport_data_str = f"{target_x},{target_y},{yaw}"
        teleport_msg = String()
        teleport_msg.data = teleport_data_str

        for _ in range(0, 3):
            self.teleport_pub.publish(teleport_msg)
            # Add a small sleep between publishes if needed, but be mindful of blocking
            time.sleep(0.1)

        self.logger.info("Teleport command (String) published. Waiting for teleport service to act.")

        time.sleep(2.0)

        self.reset_slam_and_finalize()

    def reset_slam_map(self):
        """Reset the SLAM map using the slam_toolbox reset service."""
        # Wait for the service to be available just before calling it.
        self.logger.info("Waiting for SLAM reset service...")
        # Increased timeout to 5 seconds (adjust as needed)
        service_wait_timeout = 5.0
        if not self.clear_slam_map_client.wait_for_service(timeout_sec=service_wait_timeout):
             self.logger.warn(f"SLAM reset service not available after waiting {service_wait_timeout} seconds. Cannot reset map.")
             return False

        self.logger.info("SLAM reset service is available. Proceeding with reset.")

        req = Reset.Request()
        future = self.clear_slam_map_client.call_async(req)

        rclpy.spin_until_future_complete(self.env, future, timeout_sec=5.0)

        if future.result() is not None:
            self.logger.info("SLAM map reset via service call succeeded")
            return True
        else:
            self.logger.error("Failed to reset SLAM map via service call")
            return False

    def reset_slam_and_finalize(self):
        """Reset SLAM map and finalize the reset process after teleportation trigger."""
        # Reset the SLAM map after the teleport command is published.
        # Add a small delay to allow the teleport service node to potentially act
        # and for Gazebo physics to start settling.
        time.sleep(1.0)

        self.logger.info("Teleport command sent. Now resetting SLAM map...")
        map_reset = self.reset_slam_map()
        # Add a delay after SLAM reset for it to take effect
        time.sleep(2.0)

        # After SLAM reset, finalize the process
        self.finalize_reset()

    def finalize_reset(self):
        """Finalize the reset process after position correction and SLAM reset."""

        self.logger.info("Reset process completed")
        self.is_resetting = False # Reset the flag


    def is_reset_in_progress(self):
        """Check if a reset is currently in progress"""
        return self.is_resetting

    def shutdown_hook(self):
        """Clean up resources when the handler is shutting down."""
        self.logger.info("Shutting down SimulationResetHandler")
        # Cancel all timers managed by the handler
        if self.control_timer:
            self.control_timer.cancel()
        # Note: rclpy doesn't provide a direct way to cancel async futures easily
        # Ensure is_resetting is false on shutdown
        self.is_resetting = False
