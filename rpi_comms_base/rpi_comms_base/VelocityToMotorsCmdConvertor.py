import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class VelocityToMotorsCmdConvertor:
    def __init__(self, motors_synchronizer, motor_controller, wheel_separation, wheel_radius):
        self.motors_synchronizer = motors_synchronizer
        self.motor_controller = motor_controller

        self.wheel_separation = wheel_separation
        self.wheel_radius = wheel_radius


    def convert_vel_to_motor_dir(self, msg):
        """
        Convert msg to linear and angular velocity to motor dir
        """
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

        # Convert received velocities to motor commands
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