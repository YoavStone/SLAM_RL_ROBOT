import math
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

from .MotorsSynchronizer import MotorsSynchronizer
from .MotorsController import MotorsController


class RobotPositionCalculator:
    def __init__(self, motors_synchronizer, wheel_radius, wheel_separation, ticks_per_revolution):
        # Initialize position calculator
        self.motors_synchronizer = motors_synchronizer
        self.last_right_pos = 0
        self.last_left_pos = 0
        self.last_time = time.time()

        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation
        self.ticks_per_revolution = ticks_per_revolution

        # Robot pose (x, y, theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def create_odom_message(self, msg_time):
        # Get current motor positions
        right_pos, left_pos = self.motors_synchronizer.get_motors_pos()

        # Calculate change in encoder ticks
        delta_right = right_pos - self.last_right_pos
        delta_left = left_pos - self.last_left_pos

        # Update stored positions
        self.last_right_pos = right_pos
        self.last_left_pos = left_pos

        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt == 0:
            return  # Avoid division by zero

        # Convert ticks to distance traveled by each wheel
        right_distance = 2 * math.pi * self.wheel_radius * delta_right / self.ticks_per_revolution
        left_distance = 2 * math.pi * self.wheel_radius * delta_left / self.ticks_per_revolution

        # Calculate linear and angular displacement
        linear_displacement = (right_distance + left_distance) / 2.0
        angular_displacement = (right_distance - left_distance) / self.wheel_separation

        # Update pose
        self.theta += angular_displacement
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Update position based on heading
        self.x += linear_displacement * math.cos(self.theta)
        self.y += linear_displacement * math.sin(self.theta)

        # Calculate velocities
        linear_velocity = linear_displacement / dt
        angular_velocity = angular_displacement / dt

        # Create and publish odometry message
        odom = Odometry()
        odom.header.stamp = msg_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        # Set position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        # Set orientation (quaternion from yaw)
        cy = math.cos(self.theta * 0.5)
        sy = math.sin(self.theta * 0.5)
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = sy
        odom.pose.pose.orientation.w = cy

        # Set velocity
        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.angular.z = angular_velocity

        # if abs(delta_right) > 0 or abs(delta_left) > 0:
        #     print(f'Position: ({self.x:.3f}, {self.y:.3f}, {self.theta:.3f}), R: {right_pos}, L: {left_pos}')

        # return the message
        return odom