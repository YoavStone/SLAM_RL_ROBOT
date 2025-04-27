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

        # Use a single correction factor for simplicity and consistency
        self.pwm_change_factor = 0.3

        self.max_speed = 0.8
        self.min_speed = 0.15  # Slightly increased for better starting torque

        self.ticks_per_revolution = ticks_per_revolution

        self.last_time = time.time()

        # Add speed filtering to reduce noise
        self.filter_alpha = 0.7  # 0-1, higher = more weight to new readings

        # Add calibration for speed-to-PWM relationship
        # This helps address motor asymmetry without direction-specific parameters
        self.speed_calibration = 0.05  # Experimental value - adjust based on testing

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

        # Apply low-pass filter to smooth speed readings
        self.right_wheel_speed = self.filter_alpha * new_right_speed + (1 - self.filter_alpha) * self.right_wheel_speed
        self.left_wheel_speed = self.filter_alpha * new_left_speed + (1 - self.filter_alpha) * self.left_wheel_speed

        return self.right_wheel_speed, self.left_wheel_speed

    def closed_loop_control_speed(self):
        self.get_motors_speeds()

        # If both motors should be stopped, just set PWM to 0
        if abs(self.r_motor_desired_speed) < 0.01 and abs(self.l_motor_desired_speed) < 0.01:
            self.motors_synchronizer.set_pwm(0.0, 0.0)
            return

        # Get current speeds (we'll work with absolute values and apply direction later)
        abs_right_speed = abs(self.right_wheel_speed)
        abs_left_speed = abs(self.left_wheel_speed)

        # Get target speeds
        abs_right_target = abs(self.r_motor_desired_speed)
        abs_left_target = abs(self.l_motor_desired_speed)

        # Calculate speed differences (target - actual)
        speed_diff_r = abs_right_target - abs_right_speed
        speed_diff_l = abs_left_target - abs_left_speed

        # Calculate proportional error term
        # This uses actual speed difference rather than a relative percentage
        # This approach is more consistent across different speeds
        error_r = speed_diff_r * self.pwm_change_factor
        error_l = speed_diff_l * self.pwm_change_factor

        # Get current PWM values (absolute)
        current_pwm_r = abs(self.motors_synchronizer.R_Motor.pwm)
        current_pwm_l = abs(self.motors_synchronizer.L_Motor.pwm)

        # If PWM is very low, use a starting value based on desired speed
        if current_pwm_r < 0.05:
            # Base initial PWM on desired speed with a minimum value
            current_pwm_r = max(self.min_speed, abs_right_target * self.speed_calibration)

        if current_pwm_l < 0.05:
            # Base initial PWM on desired speed with a minimum value
            current_pwm_l = max(self.min_speed, abs_left_target * self.speed_calibration)

        # Limit maximum change per cycle to prevent oscillations
        max_change = 0.1
        error_r = max(-max_change, min(error_r, max_change))
        error_l = max(-max_change, min(error_l, max_change))

        # Calculate new PWM values
        new_pwm_r = current_pwm_r + error_r
        new_pwm_l = current_pwm_l + error_l

        # Apply minimum PWM if motors should be moving
        if abs_right_target > 0.05:
            new_pwm_r = max(self.min_speed, new_pwm_r)

        if abs_left_target > 0.05:
            new_pwm_l = max(self.min_speed, new_pwm_l)

        # Apply maximum limit
        new_pwm_r = min(self.max_speed, new_pwm_r)
        new_pwm_l = min(self.max_speed, new_pwm_l)

        # Apply direction
        r_dir = 1 if self.r_motor_desired_speed >= 0 else -1
        l_dir = 1 if self.l_motor_desired_speed >= 0 else -1

        # Set final PWM values
        self.motors_synchronizer.set_pwm(new_pwm_r * r_dir, new_pwm_l * l_dir)