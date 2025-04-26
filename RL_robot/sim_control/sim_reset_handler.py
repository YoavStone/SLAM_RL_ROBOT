import math
import time
import random
import subprocess
from slam_toolbox.srv import Reset
from geometry_msgs.msg import Twist
import rclpy
import threading
from std_msgs.msg import String
import os


class SpawnPositionHandler:
    def __init__(self, staring_pos):
        self.positions = [
            [0.0, 0.0],
            [6.3, 0.0],
            [-6.3, 0.0],
            [0.0, 6.3],
            [0.0, -6.3]
        ]
        try:
            x, y = staring_pos.split(',')
            self.staring_pos = [float(x.strip()), float(y.strip())]
        except ValueError:
            self.staring_pos = None
            print("no start pos specified teleport to random locations")
        self.target_spawn_position = None

    def get_target_position(self):
        """Get target position and yaw for teleportation"""
        # Random yaw angle
        yaw = random.choice([0.0, math.pi / 2, math.pi, math.pi * 3 / 2])

        if self.staring_pos is not None:
            # Use the provided spawn location if available
            return [self.staring_pos[0], self.staring_pos[1], yaw]
        else:
            # Choose a random position from predefined positions
            chosen_position = random.choice(self.positions)
            return [chosen_position[0], chosen_position[1], yaw]

class SimulationResetHandler:
    """
    A robust reset handler that combines odometry correction like in EpisodeMonitor
    with proper thread management and improved teleport commands.
    """

    def __init__(self, env):
        """Initialize with reference to the environment"""
        # Store environment reference
        self.env = env
        self.logger = env.get_logger()
        self.logger.info("SimulationResetHandler initializing...")

        # Reset state with manual lock
        self._reset_lock = threading.Lock()
        self._is_resetting = False
        self.is_first_ep = True

        # Thread management
        self._reset_thread = None

        # Model configuration
        self.model_name = 'mapping_robot'
        self.world_name = 'empty'

        # Predefined positions for teleportation
        staring_pos = self.env.get_parameter('spawn_location').value
        self.spawn_position_handler = SpawnPositionHandler(staring_pos)

        # SLAM reset client
        self.clear_slam_map_client = self.env.create_client(
            Reset,
            '/slam_toolbox/reset'
        )

        self.env.teleport_pub = self.env.create_publisher(
            String,
            '/teleport_command',
            10
        )

        self.logger.info("SimulationResetHandler initialized")

    def update(self, terminated, truncated):
        """Handle episode termination and trigger reset"""
        is_done = terminated or truncated

        # Use thread-safe check and set to prevent concurrent resets
        if is_done and not self.is_reset_in_progress():
            # Start reset process in background
            self.reset_environment()
            return True

        return False

    def reset_environment_thread_func(self):
        """Reset function that runs in a thread"""
        try:
            self.logger.info("Reset thread started")

            # Stop the robot first
            stop_cmd = Twist()
            self.env.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.5)

            # For first episode, skip odometry correction
            if self.is_first_ep:
                self.is_first_ep = False
                self.direct_teleport()  # Use direct teleport instead of subprocess
            else:
                # Perform odometry correction, then teleport
                self.perform_odometry_correction()

            # All done
            self.logger.info("Reset process completed")

        except Exception as e:
            self.logger.error(f"Error in reset thread: {e}")
        finally:
            # Always release reset flag at the end
            with self._reset_lock:
                self._is_resetting = False
                self._reset_thread = None

    def perform_odometry_correction(self):
        """Drive robot back to zero position and orientation, then teleport"""
        self.logger.info("Starting odometry correction")

        # First send stop command to ensure the robot is stationary
        stop_cmd = Twist()
        self.env.cmd_vel_pub.publish(stop_cmd)
        time.sleep(0.5)

        # Maximum time to allow for odometry correction
        max_correction_time = 45.0  # 45 seconds
        start_time = time.time()

        # Start with position correction phase
        correction_phase = 'position'

        # Continue until time limit or position & orientation are corrected
        while time.time() - start_time < max_correction_time:
            # Check if odometry data is available
            if self.env.current_odom is None:
                self.logger.warn("No odometry data available for correction")
                time.sleep(0.5)
                continue

            # Get current position and orientation
            current_x = self.env.current_odom.pose.pose.position.x
            current_y = self.env.current_odom.pose.pose.position.y
            current_yaw = self.get_current_yaw()

            # Create velocity command
            cmd = Twist()

            if correction_phase == 'position':
                # Calculate distance to origin
                distance = math.sqrt(current_x ** 2 + current_y ** 2)

                # If close enough to origin, switch to yaw correction
                if distance < 0.05:
                    self.logger.info(f"Reached origin position: ({current_x:.3f}, {current_y:.3f})")

                    # Switch to yaw correction
                    correction_phase = 'yaw'

                    # Stop the robot before yaw correction
                    stop_cmd = Twist()
                    self.env.cmd_vel_pub.publish(stop_cmd)
                    time.sleep(0.5)

                    continue  # Skip to next iteration

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
                if abs(angle_diff) > 0.2:
                    # Only turn, no forward movement
                    cmd.linear.x = 0.0
                    cmd.angular.z = angle_diff * 0.5

                    # Limit angular velocity
                    if abs(cmd.angular.z) > 0.5:
                        cmd.angular.z = 0.5 if cmd.angular.z > 0 else -0.5
                else:
                    # Now that we're facing the target, move forward
                    linear_speed = 0.5 * distance
                    # Limit linear speed
                    if abs(linear_speed) > 0.6:
                        linear_speed = 0.6 if linear_speed > 0 else -0.6
                    cmd.linear.x = linear_speed

                    # Still apply minor angular corrections
                    cmd.angular.z = angle_diff * 0.5
                    # Limit angular velocity
                    if abs(cmd.angular.z) > 0.5:
                        cmd.angular.z = 0.5 if cmd.angular.z > 0 else -0.5

            elif correction_phase == 'yaw':
                # Yaw correction phase - rotate to yaw=0

                # If close enough to zero yaw, finish correction
                if abs(current_yaw) < 0.02:
                    self.logger.info(f"Reached target yaw: {current_yaw:.3f}")

                    # Stop the robot
                    stop_cmd = Twist()
                    self.env.cmd_vel_pub.publish(stop_cmd)
                    time.sleep(0.5)

                    # Once odom is corrected, teleport the robot
                    self.direct_teleport()  # Use direct teleport instead of subprocess
                    return  # Exit the correction loop

                # Only rotate, no linear movement
                cmd.linear.x = 0.0
                cmd.angular.z = -current_yaw * 0.5

                # Limit angular velocity
                if abs(cmd.angular.z) > 0.3:
                    cmd.angular.z = 0.3 if cmd.angular.z > 0 else -0.3

            # Publish command
            self.env.cmd_vel_pub.publish(cmd)

            # Brief sleep to allow time for movement
            time.sleep(0.1)

        # If we got here, the correction timed out
        self.logger.warn("Odometry correction timed out, proceeding to teleport anyway")
        self.direct_teleport()  # Use direct teleport instead of subprocess

    def get_current_yaw(self):
        """Extract yaw angle from quaternion orientation"""
        if self.env.current_odom is None:
            return 0.0

        # Get quaternion components
        qx = self.env.current_odom.pose.pose.orientation.x
        qy = self.env.current_odom.pose.pose.orientation.y
        qz = self.env.current_odom.pose.pose.orientation.z
        qw = self.env.current_odom.pose.pose.orientation.w

        # Convert to yaw angle
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

    def direct_teleport(self):
        """
        Teleport the robot by publishing a teleport command to the teleport service.
        """
        # Get target position and orientation
        target_x, target_y, yaw = self.spawn_position_handler.get_target_position()

        # Log the teleport
        print(f"Publishing teleport command to: ({target_x}, {target_y}, {yaw})")

        # Create the teleport command string
        teleport_cmd = f"{target_x},{target_y},{yaw}"

        # Create the message
        msg = String()
        msg.data = teleport_cmd

        # Publish the command to the teleport service
        self.env.teleport_pub.publish(msg)

        # Wait for the teleport to complete
        time.sleep(3.0)

        # Reset SLAM map
        self.reset_slam_map()

    def reset_slam_map(self):
        """Reset the SLAM map with timeout protection"""
        try:
            self.logger.info("Attempting to reset SLAM map")

            # Wait for service with timeout
            if not self.clear_slam_map_client.wait_for_service(timeout_sec=2.0):
                self.logger.warn("SLAM reset service not available, continuing")
                return False

            # Call service with timeout
            req = Reset.Request()
            future = self.clear_slam_map_client.call_async(req)

            # Use a timeout to prevent blocking
            timeout_sec = 3.0
            start_time = time.time()
            while (time.time() - start_time) < timeout_sec:
                if future.done():
                    if future.exception() is not None:
                        self.logger.warn(f"SLAM reset service call failed: {future.exception()}")
                        return False
                    else:
                        self.logger.info("SLAM map reset successful")
                        return True
                time.sleep(0.1)  # Small sleep to avoid tight loop
                rclpy.spin_once(self.env, timeout_sec=0.05)

            # If we get here, the service call timed out
            self.logger.warn(f"SLAM reset service call timed out after {timeout_sec}s")
            return False

        except Exception as e:
            self.logger.error(f"Error in SLAM map reset: {e}")
            return False

    def reset_environment(self):
        """Public method to manually trigger a reset"""
        # This will be called by the DQLEnv.reset() method

        # Thread-safe check and set for reset state
        with self._reset_lock:
            # If already resetting, just wait
            if self._is_resetting:
                self.logger.warn("Reset already in progress, waiting for completion")
                should_start_new_thread = False
            else:
                self._is_resetting = True
                should_start_new_thread = True

                # Clean up any old thread reference
                if self._reset_thread is not None and self._reset_thread.is_alive():
                    self.logger.warn("Old reset thread is still alive, this shouldn't happen")
                    # We won't join it as that would block, but we'll create a new one

        # If we should start a new thread, do so
        if should_start_new_thread:
            # Create and start the thread
            self._reset_thread = threading.Thread(target=self.reset_environment_thread_func)
            self._reset_thread.daemon = True
            self._reset_thread.start()

        # Wait for reset to complete or timeout
        max_wait = 10.0  # seconds
        start_wait = time.time()

        # We'll return after waiting at least 7 seconds or when reset completes
        min_wait = 7.0  # seconds to allow teleport and SLAM reset

        while time.time() - start_wait < max_wait and self.is_reset_in_progress():
            time.sleep(0.1)
            # Don't check too frequently to avoid a tight loop

            # If we've waited at least the minimum time, we can return
            if time.time() - start_wait >= min_wait:
                break

        # If we're still resetting after max wait, log it but continue anyway
        if self.is_reset_in_progress():
            self.logger.warn(f"Reset still in progress after {max_wait}s, but returning to allow training to continue")

        return True

    def is_reset_in_progress(self):
        """Thread-safe check if reset is in progress"""
        with self._reset_lock:
            return self._is_resetting