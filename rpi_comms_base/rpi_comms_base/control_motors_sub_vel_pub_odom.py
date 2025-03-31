import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math
from . import L298nDriver
from . import MotorController


class RobotControlNode(Node):
    def __init__(self):
        print("will initialized robot control node")
        super().__init__('robot_control_node')

        # Initialize motor driver
        # Define GPIO pins
        in1 = 17
        in2 = 27
        in3 = 23
        in4 = 24

        pwmR = 22
        pwmL = 25

        ena1 = 19
        enb1 = 26
        ena2 = 13
        enb2 = 6

        FREQ = 1000

        self.driver = L298nDriver.L298nDriver(in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2)
        self.driver.call_encoder_interrupt()

        # Initialize velocity subscriber
        self.velocity_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.velocity_callback,
            10)

        # Initialize position publisher
        self.odom_publisher = self.create_publisher(
            Odometry,
            'odom_robot',
            10)

        # Set up timer for regular position updates
        self.publisher_timer = self.create_timer(0.1, self.publish_position)  # 10Hz update rate
        self.closed_loop_speed_control_timer = self.create_timer(
            0.05, self.motor_controller.closed_loop_control_speed)  # 20Hz update rate

        # Set up parameters for odometry calculation
        self.wheel_radius = 0.034  # meters
        self.wheel_separation = 0.34  # meters
        self.ticks_per_revolution = 170  # based on your encoder
        self.last_right_pos = 0
        self.last_left_pos = 0

        # Robot pose (x, y, theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Speed multiplier factor
        self.linear_speed_factor = 1.0
        self.turn_speed_factor = 1.0

        self.motor_controller = MotorController.MotorController(self.driver, self.ticks_per_revolution)

        # Time tracking
        self.last_time = time.time()

        print('Robot Control Node has been initialized')

    def convert_cmd_vel_to_motor_speeds(self, linear_vel, angular_vel):
        """
        Convert linear and angular velocities from cmd_vel into desired
        left and right wheel speeds in rad/s. Signs indicate direction.

        Args:
            linear_vel (float): Linear velocity in m/s
            angular_vel (float): Angular velocity in rad/s

        Returns:
            tuple: (right_wheel_speed, left_wheel_speed) in rad/s
                  Positive values for forward, negative for backward
        """
        # Calculate wheel velocities based on differential drive kinematics
        # For a differential drive robot:
        # v_r = v + ω*L/2
        # v_l = v - ω*L/2
        # where v is linear velocity, ω is angular velocity, L is wheel separation

        # Calculate linear velocities for each wheel
        v_right = linear_vel + (angular_vel * self.wheel_separation / 2)
        v_left = linear_vel - (angular_vel * self.wheel_separation / 2)

        # Convert to angular velocity (rad/s)
        # The sign will correctly indicate direction:
        # positive for forward, negative for backward
        right_wheel_speed = v_right / self.wheel_radius
        left_wheel_speed = v_left / self.wheel_radius

        return right_wheel_speed, left_wheel_speed

    def velocity_callback(self, msg):
        speed = msg.linear.x
        turn = msg.angular.z

        if abs(speed) < 0.05 and abs(turn) < 0.05:
            # Stop if both speed and turn are very small
            speed = 0.0
            turn = 0.0
        elif abs(speed) > 0.0:
            # Primarily moving forward/backward
            turn = 0.0
        else:
            # Primarily turning
            speed = 0.0

        print(f'Received velocity command: linear={speed:.2f}, angular={turn:.2f}')

        self.motor_controller.r_motor_desired_speed, self.motor_controller.l_motor_desired_speed = self.convert_cmd_vel_to_motor_speeds(speed, turn)

        right_wheel_speed, left_wheel_speed = self.motor_controller.get_motor_speeds()

        print('desired: ', self.motor_controller.r_motor_desired_speed, self.motor_controller.l_motor_desired_speed)
        print('current: ', right_wheel_speed, left_wheel_speed)

        # Convert received velocities to motor commands
        self.convert_vel_to_motor_dir(speed, turn)

    def convert_vel_to_motor_dir(self, speed, turn):
        """
        Convert linear and angular velocity to motor dir
        """
        if speed == 0.0 == turn:
            self.driver.stop()
        elif abs(speed) > 0:
            # Primarily moving forward/backward
            if speed > 0:
                self.driver.go_forward()
            else:
                self.driver.go_backwards()
        else:
            # Primarily turning
            if turn > 0:
                self.driver.turn_left()
            else:
                self.driver.turn_right()

    def publish_position(self):
        # Get current motor positions
        right_pos, left_pos = self.driver.get_motors_pos()

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
        odom.header.stamp = self.get_clock().now().to_msg()
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

        # Publish the message
        self.odom_publisher.publish(odom)

        if abs(delta_right) > 0 or abs(delta_left) > 0:
            print(f'Position: ({self.x:.3f}, {self.y:.3f}, {self.theta:.3f}), R: {right_pos}, L: {left_pos}')


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()