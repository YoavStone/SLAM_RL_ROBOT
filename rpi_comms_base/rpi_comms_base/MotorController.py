import math
import time

from . import L298nDriver


class MotorController:
    def __init__(self, driver, ticks_per_revolution):

        self.driver = driver

        self.last_right_pos = 0.0
        self.last_left_pos = 0.0

        self.right_wheel_speed = 0.0
        self.left_wheel_speed = 0.0

        self.r_motor_desired_speed = 0.0
        self.l_motor_desired_speed = 0.0

        self.pwm_change_factor = 1.0  # adjust how much the inaccuracy in the speed difference affects the pwm change

        self.ticks_per_revolution = ticks_per_revolution

        self.last_time = time.time()


    def get_motor_speeds(self):
        """
        Calculate the current speed of each motor in rad/s based on encoder readings
        over a small time delta.

        Returns:
            tuple: (right_wheel_speed, left_wheel_speed) in rad/s
        """
        # Get current motor positions
        right_pos, left_pos = self.driver.get_motors_pos()

        # Calculate change in encoder ticks
        delta_right = right_pos - self.last_right_pos
        delta_left = left_pos - self.last_left_pos

        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_time

        # Avoid division by zero
        if dt < 0.001:
            return 0.0, 0.0

        self.last_right_pos = right_pos
        self.last_left_pos = left_pos

        self.last_time = time.time()

        # Convert ticks to angular velocity (rad/s)
        self.right_wheel_speed = (delta_right / self.ticks_per_revolution) * 2 * math.pi / dt
        self.left_wheel_speed = (delta_left / self.ticks_per_revolution) * 2 * math.pi / dt

        return self.right_wheel_speed, self.left_wheel_speed


    def closed_loop_control_speed(self):

        if self.r_motor_desired_speed == 0.0 == self.l_motor_desired_speed:
            return

        dt_speeds_r = self.r_motor_desired_speed - self.right_wheel_speed
        dt_speeds_l = self.l_motor_desired_speed - self.left_wheel_speed

        error_speed_r = dt_speeds_r / self.r_motor_desired_speed
        error_speed_l = dt_speeds_l / self.l_motor_desired_speed

        pwm_r = self.driver.R_Motor.pwm
        pwm_l = self.driver.L_Motor.pwm

        if pwm_r != 0.0 != pwm_l:
            error_pwm_r = pwm_r * error_speed_r * self.pwm_change_factor
            error_pwm_l = pwm_l * error_speed_l * self.pwm_change_factor
        else:
            error_pwm_r = 1.0 * error_speed_r * self.pwm_change_factor
            error_pwm_l = 1.0 * error_speed_l * self.pwm_change_factor

        new_pwm_r = abs(min(1.0, pwm_r + error_pwm_r))
        new_pwm_l = abs(min(1.0, pwm_l + error_pwm_l))

        # should never happen but here just in case:
        if new_pwm_r < 0.05 and new_pwm_l < 0.05:
            self.driver.stop()
            return

        self.driver.set_pwm(new_pwm_r, new_pwm_l)