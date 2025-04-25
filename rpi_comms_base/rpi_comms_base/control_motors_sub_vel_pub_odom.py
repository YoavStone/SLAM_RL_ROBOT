import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

from .MotorsSynchronizer import MotorsSynchronizer
from .MotorsController import MotorsController
from .RobotPositionCalculator import RobotPositionCalculator


class RobotControlNode(Node):
    def __init__(self):
        print("will initialized robot control node")
        super().__init__('robot_control_node')

        # Initialize motors synchronizer
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

        self.motors_synchronizer = MotorsSynchronizer(in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2)
        self.motors_synchronizer.call_encoder_interrupt()

        # Initialize velocity subscriber
        self.velocity_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.velocity_callback,
            10
        )

        # Initialize position publisher
        self.odom_publisher = self.create_publisher(
            Odometry,
            'odom_robot',
            10
        )

        # Set up parameters for odometry calculation
        self.wheel_radius = 0.034  # meters
        self.wheel_separation = 0.34  # meters
        self.ticks_per_revolution = 170  # based on your encoder

        # Speed multiplier factor
        self.linear_speed_factor = 1.0
        self.turn_speed_factor = 1.0

        self.motor_controller = MotorsController(self.motors_synchronizer, self.ticks_per_revolution)
        self.closed_loop_speed_control_timer = self.create_timer(
            0.05, self.motor_controller.closed_loop_control_speed)  # 20Hz update rate

        # Time tracking
        self.last_time = time.time()

        self.odom_calculator = RobotPositionCalculator(self.motors_synchronizer, self.wheel_radius, self.wheel_separation, self.ticks_per_revolution)

        # Set up timer for regular position updates
        self.publisher_timer = self.create_timer(0.1, self.publish_position)  # 10Hz update rate

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

        right_wheel_speed, left_wheel_speed = self.motor_controller.get_motors_speeds()

        # print('desired: ', self.motor_controller.r_motor_desired_speed, self.motor_controller.l_motor_desired_speed)
        # print('current: ', right_wheel_speed, left_wheel_speed)

        # Convert received velocities to motor commands
        self.convert_vel_to_motor_dir(speed, turn)

    def convert_vel_to_motor_dir(self, speed, turn):
        """
        Convert linear and angular velocity to motor dir
        """
        if speed == 0.0 == turn:
            self.motors_synchronizer.stop()
        elif abs(speed) > 0:
            # Primarily moving forward/backward
            if speed > 0:
                self.motors_synchronizer.go_forward()
            else:
                self.motors_synchronizer.go_backwards()
        else:
            # Primarily turning
            if turn > 0:
                self.motors_synchronizer.turn_left()
            else:
                self.motors_synchronizer.turn_right()

    def publish_position(self):
        # Get current motor positions
        odom = self.odom_calculator.create_odom_message(self.get_clock().now().to_msg())
        # Publish the message
        self.odom_publisher.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()