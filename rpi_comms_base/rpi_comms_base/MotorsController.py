import math
import time


class MotorsController:
    def __init__(self, motors_synchronizer, ticks_per_revolution):
        self.motors_synchronizer = motors_synchronizer
        self.ticks_per_revolution = ticks_per_revolution

        # Motor state tracking
        self.last_right_pos = 0.0
        self.last_left_pos = 0.0
        self.right_wheel_speed = 0.0
        self.left_wheel_speed = 0.0
        self.last_time = time.time()

        # Target speeds
        self.r_motor_desired_speed = 0.0
        self.l_motor_desired_speed = 0.0

        # PID controller parameters
        self.kp = 0.5  # Proportional gain
        self.ki = 0.2  # Integral gain
        self.kd = 0.05  # Derivative gain

        # Error history
        self.r_prev_error = 0.0
        self.l_prev_error = 0.0
        self.r_integral = 0.0
        self.l_integral = 0.0

        # Anti-windup limits
        self.max_integral = 0.3

        # Speed limits
        self.max_speed = 0.8
        self.min_speed = 0.05  # Minimum PWM to overcome static friction

        # Reset timing
        self.pid_last_time = time.time()

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

        # Avoid division by zero and ensure reasonable dt
        if dt < 0.001:
            return self.right_wheel_speed, self.left_wheel_speed  # Return previous values

        # Update state
        self.last_right_pos = right_pos
        self.last_left_pos = left_pos
        self.last_time = current_time

        # Apply low-pass filter to reduce noise (weighted average)
        alpha = 0.7  # Filter coefficient (0-1) - higher means more weight on new reading
        new_right_speed = (delta_right / self.ticks_per_revolution) * 2 * math.pi / dt
        new_left_speed = (delta_left / self.ticks_per_revolution) * 2 * math.pi / dt

        # Update speeds with filtering
        self.right_wheel_speed = alpha * new_right_speed + (1 - alpha) * self.right_wheel_speed
        self.left_wheel_speed = alpha * new_left_speed + (1 - alpha) * self.left_wheel_speed

        return self.right_wheel_speed, self.left_wheel_speed

    def closed_loop_control_speed(self):
        # Get current speeds
        self.get_motors_speeds()

        # If both desired speeds are zero, stop motors and reset PID state
        if self.r_motor_desired_speed == 0.0 and self.l_motor_desired_speed == 0.0:
            self.motors_synchronizer.set_pwm(0.0, 0.0)
            self.r_integral = 0.0
            self.l_integral = 0.0
            self.r_prev_error = 0.0
            self.l_prev_error = 0.0
            return

        # Calculate time since last PID update
        current_time = time.time()
        dt = current_time - self.pid_last_time
        self.pid_last_time = current_time

        # Ensure dt is reasonable
        if dt < 0.001 or dt > 0.1:
            dt = 0.05  # Use expected value if dt is too small or too large

        # Calculate errors
        r_error = self.r_motor_desired_speed - self.right_wheel_speed
        l_error = self.l_motor_desired_speed - self.left_wheel_speed

        # Update integral term with anti-windup
        self.r_integral += r_error * dt
        self.l_integral += l_error * dt

        # Limit integral terms to prevent windup
        self.r_integral = max(-self.max_integral, min(self.r_integral, self.max_integral))
        self.l_integral = max(-self.max_integral, min(self.l_integral, self.max_integral))

        # Calculate derivative term
        r_derivative = (r_error - self.r_prev_error) / dt
        l_derivative = (l_error - self.l_prev_error) / dt

        # Save current error for next iteration
        self.r_prev_error = r_error
        self.l_prev_error = l_error

        # Calculate PID output
        r_output = (self.kp * r_error) + (self.ki * self.r_integral) + (self.kd * r_derivative)
        l_output = (self.kp * l_error) + (self.ki * self.l_integral) + (self.kd * l_derivative)

        # Get current PWM values
        pwm_r = self.motors_synchronizer.R_Motor.pwm
        pwm_l = self.motors_synchronizer.L_Motor.pwm

        # Calculate new PWM values
        # Handle sign separately to maintain direction
        r_sign = 1 if self.r_motor_desired_speed >= 0 else -1
        l_sign = 1 if self.l_motor_desired_speed >= 0 else -1

        # Calculate absolute new PWM values
        new_pwm_r_abs = abs(pwm_r) + r_output
        new_pwm_l_abs = abs(pwm_l) + l_output

        # Apply minimum PWM to overcome static friction if motors should be moving
        if abs(self.r_motor_desired_speed) > 0.01:
            new_pwm_r_abs = max(self.min_speed, new_pwm_r_abs)

        if abs(self.l_motor_desired_speed) > 0.01:
            new_pwm_l_abs = max(self.min_speed, new_pwm_l_abs)

        # Apply maximum speed limit
        new_pwm_r_abs = min(self.max_speed, new_pwm_r_abs)
        new_pwm_l_abs = min(self.max_speed, new_pwm_l_abs)

        # Apply sign to maintain direction
        new_pwm_r = new_pwm_r_abs * r_sign
        new_pwm_l = new_pwm_l_abs * l_sign

        # Set motor PWM values
        self.motors_synchronizer.set_pwm(new_pwm_r, new_pwm_l)