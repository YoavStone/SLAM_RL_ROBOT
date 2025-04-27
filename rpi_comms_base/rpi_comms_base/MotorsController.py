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

        # Reduced from 0.55 to avoid over-correction at high speeds
        self.pwm_change_factor = 0.3

        self.max_speed = 0.8
        # Minimum PWM to overcome static friction
        self.min_speed = 0.1

        self.ticks_per_revolution = ticks_per_revolution

        self.last_time = time.time()

        # Add speed filtering to reduce noise
        self.filter_alpha = 0.7  # 0-1, higher = more weight to new readings

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

        # Avoid division by zero and unreasonably small time steps
        if dt < 0.01:
            return self.right_wheel_speed, self.left_wheel_speed

        self.last_right_pos = right_pos
        self.last_left_pos = left_pos

        self.last_time = current_time  # Using the actual time stored earlier

        # Calculate new speeds
        new_right_speed = (delta_right / self.ticks_per_revolution) * 2 * math.pi / dt
        new_left_speed = (delta_left / self.ticks_per_revolution) * 2 * math.pi / dt

        # Apply low-pass filter to smooth speed readings (especially important at high speeds)
        self.right_wheel_speed = self.filter_alpha * new_right_speed + (1 - self.filter_alpha) * self.right_wheel_speed
        self.left_wheel_speed = self.filter_alpha * new_left_speed + (1 - self.filter_alpha) * self.left_wheel_speed

        return self.right_wheel_speed, self.left_wheel_speed

    def closed_loop_control_speed(self):

        self.get_motors_speeds()

        # If both motors should be stopped, just set PWM to 0
        if self.r_motor_desired_speed == 0.0 and self.l_motor_desired_speed == 0.0:
            self.motors_synchronizer.set_pwm(0.0, 0.0)
            return

        # Calculate speed differences
        dt_speeds_r = self.r_motor_desired_speed - self.right_wheel_speed
        dt_speeds_l = self.l_motor_desired_speed - self.left_wheel_speed

        # Get current PWM values
        pwm_r = self.motors_synchronizer.R_Motor.pwm
        pwm_l = self.motors_synchronizer.L_Motor.pwm

        # Calculate relative error only if desired speed is significant
        if abs(self.r_motor_desired_speed) > 0.05:
            error_speed_r = dt_speeds_r / abs(self.r_motor_desired_speed)
        else:
            error_speed_r = 0

        if abs(self.l_motor_desired_speed) > 0.05:
            error_speed_l = dt_speeds_l / abs(self.l_motor_desired_speed)
        else:
            error_speed_l = 0

        # Calculate PWM adjustment, but with rate limiting for high speeds
        if pwm_r != 0.0 and pwm_l != 0.0:
            # Limit error correction at high speeds
            max_correction = 0.1  # Maximum allowed correction per cycle

            error_pwm_r = pwm_r * error_speed_r * self.pwm_change_factor
            error_pwm_l = pwm_l * error_speed_l * self.pwm_change_factor

            # Limit correction to prevent oscillations
            error_pwm_r = max(-max_correction, min(error_pwm_r, max_correction))
            error_pwm_l = max(-max_correction, min(error_pwm_l, max_correction))
        else:
            # Initial PWM values if current PWM is zero
            error_pwm_r = 0.15 * error_speed_r * self.pwm_change_factor
            error_pwm_l = 0.15 * error_speed_l * self.pwm_change_factor

        # Determine sign to preserve direction
        r_sign = 1 if self.r_motor_desired_speed >= 0 else -1
        l_sign = 1 if self.l_motor_desired_speed >= 0 else -1

        # Calculate new PWM values (absolute)
        new_pwm_r_abs = abs(pwm_r + error_pwm_r)
        new_pwm_l_abs = abs(pwm_l + error_pwm_l)

        # Apply minimum PWM to overcome friction if motors should be moving
        if abs(self.r_motor_desired_speed) > 0.05:
            new_pwm_r_abs = max(self.min_speed, new_pwm_r_abs)

        if abs(self.l_motor_desired_speed) > 0.05:
            new_pwm_l_abs = max(self.min_speed, new_pwm_l_abs)

        # Apply maximum speed limit
        new_pwm_r_abs = min(self.max_speed, new_pwm_r_abs)
        new_pwm_l_abs = min(self.max_speed, new_pwm_l_abs)

        # Apply sign to maintain direction
        new_pwm_r = new_pwm_r_abs * r_sign
        new_pwm_l = new_pwm_l_abs * l_sign

        # Set new PWM values
        self.motors_synchronizer.set_pwm(new_pwm_r, new_pwm_l)