import math
import time

from .MotorsSynchronizer import MotorsSynchronizer


class MotorsController:
    def __init__(self, motors_synchronizer, ticks_per_revolution, wheel_radius):

        self.motors_synchronizer = motors_synchronizer

        self.wheel_radius = wheel_radius

        self.last_right_pos = 0.0
        self.last_left_pos = 0.0

        self.right_wheel_speed = 0.0
        self.left_wheel_speed = 0.0

        self.r_motor_desired_speed = 0.0
        self.l_motor_desired_speed = 0.0

        self.pwm_change_factor = 0.3  # adjust how much the inaccuracy in the speed difference affects the pwm change

        self.max_speed = 0.8
        self.min_pwm = 0.15

        self.ticks_per_revolution = ticks_per_revolution

        self.last_time = time.time()

    def get_motors_speeds(self):
        """
        Calculate the current speed of each motor in rad/s based on encoder readings
        over a small time delta.

        Returns:
            tuple: (right_wheel_speed, left_wheel_speed) in rad/s
        """
        # Get current motor positions
        right_pos, left_pos = self.motors_synchronizer.get_motors_pos()

        # Calculate change in encoder ticks
        delta_right = right_pos - self.last_right_pos
        delta_left = left_pos - self.last_left_pos

        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_time

        # Avoid division by zero
        if dt < 0.005:
            return 0.0, 0.0

        self.last_right_pos = right_pos
        self.last_left_pos = left_pos

        self.last_time = time.time()

        # Convert ticks to spins per sec
        self.right_wheel_speed = (delta_right / self.ticks_per_revolution) / dt
        self.left_wheel_speed = (delta_left / self.ticks_per_revolution) / dt

        return self.right_wheel_speed, self.left_wheel_speed

    def closed_loop_control_speed(self):

        self.get_motors_speeds()
        print("real: ", self.right_wheel_speed)
        print("des: ", self.r_motor_desired_speed)

        if self.r_motor_desired_speed <= 0.05 >= self.l_motor_desired_speed:
            self.motors_synchronizer.stop()
            self.motors_synchronizer.set_pwm(0.0, 0.0)
            return

        dt_speeds_r = self.r_motor_desired_speed - self.right_wheel_speed
        dt_speeds_l = self.l_motor_desired_speed - self.left_wheel_speed

        error_speed_r = dt_speeds_r / self.r_motor_desired_speed
        error_speed_l = dt_speeds_l / self.l_motor_desired_speed

        # make between -1 and 1
        error_speed_r = max(-1.0, min(error_speed_r, 1.0))
        error_speed_l = max(-1.0, min(error_speed_l, 1.0))

        print("dt: ", dt_speeds_r)
        print("err: ", error_speed_r)

        # Get current PWM values (absolute)
        pwm_r = abs(self.motors_synchronizer.R_Motor.pwm)
        pwm_l = abs(self.motors_synchronizer.L_Motor.pwm)

        if pwm_r != 0.0 != pwm_l:
            error_pwm_r = pwm_r * error_speed_r * self.pwm_change_factor
            error_pwm_l = pwm_l * error_speed_l * self.pwm_change_factor
        else:
            error_pwm_r = 1.0 * error_speed_r * self.pwm_change_factor
            error_pwm_l = 1.0 * error_speed_l * self.pwm_change_factor

        new_pwm_r = min(self.max_speed, abs(pwm_r + error_pwm_r))
        new_pwm_l = min(self.max_speed, abs(pwm_l + error_pwm_l))

        print("pwm: ", pwm_r)
        print("new pwm: ", new_pwm_r)

        self.motors_synchronizer.set_pwm(new_pwm_r, new_pwm_l)