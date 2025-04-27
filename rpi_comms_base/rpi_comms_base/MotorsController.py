import math
import time

from .MotorsSynchronizer import MotorsSynchronizer


class MotorsController:
    def __init__(self, motors_synchronizer, ticks_per_revolution):

        self.motors_synchronizer = motors_synchronizer

        self.last_right_pos = 0.0
        self.last_left_pos = 0.0

        self.right_wheel_speed = 0.0
        self.left_wheel_speed = 0.0

        self.r_motor_desired_speed = 0.0
        self.l_motor_desired_speed = 0.0

        self.pwm_change_factor = 0.15  # adjust how much the inaccuracy in the speed difference affects the pwm change

        self.max_speed = 0.8

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

        # Get desired speeds (absolute values)
        abs_target_r = abs(self.r_motor_desired_speed)
        abs_target_l = abs(self.l_motor_desired_speed)

        # If both motors should be stopped, just set PWM to 0
        if abs_target_r <= 0.05 and abs_target_l <= 0.05:
            self.motors_synchronizer.stop()
            self.motors_synchronizer.set_pwm(0.0, 0.0)
            return

        # Calculate speed differences and errors
        dt_speeds_r = abs_target_r - abs(self.right_wheel_speed)
        dt_speeds_l = abs_target_l - abs(self.left_wheel_speed)

        # Calculate relative errors (protect against division by zero)
        if abs_target_r > 0.05:
            error_speed_r = dt_speeds_r / abs_target_r
        else:
            error_speed_r = 0.0

        if abs_target_l > 0.05:
            error_speed_l = dt_speeds_l / abs_target_l
        else:
            error_speed_l = 0.0

        # Limit error to reasonable range
        error_speed_r = max(-1.0, min(error_speed_r, 1.0))
        error_speed_l = max(-1.0, min(error_speed_l, 1.0))

        # Get current PWM values (always positive)
        current_pwm_r = abs(self.motors_synchronizer.R_Motor.pwm)
        current_pwm_l = abs(self.motors_synchronizer.L_Motor.pwm)

        # Calculate PWM adjustments
        if current_pwm_r > 0.01:
            error_pwm_r = current_pwm_r * error_speed_r * self.pwm_change_factor
        else:
            # If PWM is essentially zero, start with minimum PWM
            error_pwm_r = 1.0 * error_speed_r * self.pwm_change_factor * 2


        if current_pwm_l > 0.01:
            error_pwm_l = current_pwm_l * error_speed_l * self.pwm_change_factor
        else:
            # If PWM is essentially zero, start with minimum PWM
            error_pwm_l = 1.0 * error_speed_l * self.pwm_change_factor * 2

        # Calculate new PWM values (always positive)
        new_pwm_r = current_pwm_r + error_pwm_r
        new_pwm_l = current_pwm_l + error_pwm_l

        # Apply limits
        new_pwm_r = min(self.max_speed, new_pwm_r)
        new_pwm_l = min(self.max_speed, new_pwm_l)

        # Set final PWM values (always positive)
        self.motors_synchronizer.set_pwm(new_pwm_r, new_pwm_l)