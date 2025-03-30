from . import Motor
import time
import numpy as np


class PIDController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.01, sample_time=0.1):
        # PID constants
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        # Initialize variables
        self.sample_time = sample_time  # seconds
        self.last_time = time.time()
        self.last_error = 0
        self.integral = 0
        self.setpoint = 0
        self.output_limits = (0, 1.0)  # PWM limits

    def compute(self, current_value):
        """Calculate PID output value for given reference input and feedback"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt < self.sample_time:
            return None  # Not enough time has passed

        # Calculate error
        error = self.setpoint - current_value

        # Calculate integral and prevent windup
        self.integral += error * dt
        if self.output_limits:
            self.integral = max(min(self.integral, self.output_limits[1]), self.output_limits[0])

        # Calculate derivative
        derivative = (error - self.last_error) / dt if dt > 0 else 0

        # Calculate output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Apply limits to output
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])

        # Remember variables for next calculation
        self.last_time = current_time
        self.last_error = error

        return output

    def set_target(self, setpoint):
        """Set the target value for the controller"""
        self.setpoint = setpoint

    def reset(self):
        """Reset the controller"""
        self.last_error = 0
        self.integral = 0


class L298nDriver:
    def __init__(self, in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2):
        # Initialize motors
        self.R_Motor = Motor.Motor(in1, in2, pwmR, FREQ, ena1, enb1)
        self.L_Motor = Motor.Motor(in3, in4, pwmL, FREQ, ena2, enb2)

        # Motor speed controllers
        self.R_pid = PIDController(kp=0.2, ki=0.05, kd=0.01)
        self.L_pid = PIDController(kp=0.2, ki=0.05, kd=0.01)

        # Physical parameters
        self.wheel_radius = 0.034  # meters
        self.wheel_separation = 0.34  # meters
        self.ticks_per_revolution = 170  # encoder ticks per revolution
        self.max_linear_speed = 1  # max speed in m/s at full PWM

        # Speed tracking
        self.last_R_pos = 0
        self.last_L_pos = 0
        self.last_time = time.time()
        self.update_interval = 0.1  # seconds

        # Target speeds
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0

        # Current speeds
        self.current_R_velocity = 0.0
        self.current_L_velocity = 0.0

    def go_forward(self):
        self.R_Motor.forward()
        self.L_Motor.forward()

    def go_backwards(self):
        self.R_Motor.backwards()
        self.L_Motor.backwards()

    def stop(self):
        self.R_Motor.stop()
        self.L_Motor.stop()
        self.reset_controllers()

    def turn_right(self):
        self.R_Motor.backwards()
        self.L_Motor.forward()

    def turn_left(self):
        self.R_Motor.forward()
        self.L_Motor.backwards()

    def set_velocity(self, linear_velocity, angular_velocity):
        """
        Set target linear and angular velocity and convert to wheel velocities

        Args:
            linear_velocity: Target linear velocity in m/s
            angular_velocity: Target angular velocity in rad/s
        """
        # Store target velocities
        self.target_linear_velocity = linear_velocity
        self.target_angular_velocity = angular_velocity

        # Calculate left and right wheel velocities using differential drive kinematics
        right_wheel_velocity = (linear_velocity + (angular_velocity * self.wheel_separation / 2.0)) / self.wheel_radius
        left_wheel_velocity = (linear_velocity - (angular_velocity * self.wheel_separation / 2.0)) / self.wheel_radius

        # Convert to target encoder counts per second
        right_ticks_per_sec = right_wheel_velocity * self.ticks_per_revolution / (2 * np.pi)
        left_ticks_per_sec = left_wheel_velocity * self.ticks_per_revolution / (2 * np.pi)

        # Set PID controller targets
        self.R_pid.set_target(abs(right_ticks_per_sec))
        self.L_pid.set_target(abs(left_ticks_per_sec))

        # Set direction based on calculated velocities
        if right_wheel_velocity >= 0 and left_wheel_velocity >= 0:
            self.go_forward()
        elif right_wheel_velocity < 0 and left_wheel_velocity < 0:
            self.go_backwards()
        elif right_wheel_velocity < 0 and left_wheel_velocity >= 0:
            self.turn_right()
        elif right_wheel_velocity >= 0 and left_wheel_velocity < 0:
            self.turn_left()
        else:
            self.stop()

    def set_speed(self, speed):
        """Legacy method - converts a simple speed parameter to linear velocity"""
        # Convert speed (0-1) to approximate linear velocity in m/s
        self.set_velocity(speed * self.max_linear_speed, 0.0)

    def update_speed_control(self):
        """
        Update motor speeds based on encoder feedback
        This should be called frequently, preferably in a timer callback
        """
        current_time = time.time()
        dt = current_time - self.last_time

        if dt < self.update_interval:
            return  # Not enough time has passed

        # Get current encoder positions
        right_pos, left_pos = self.get_motors_pos()

        # Calculate change in encoder ticks
        delta_right = right_pos - self.last_R_pos
        delta_left = left_pos - self.last_L_pos

        # Calculate current speed in ticks per second
        right_ticks_per_sec = delta_right / dt
        left_ticks_per_sec = delta_left / dt

        # Update current velocities in m/s
        self.current_R_velocity = (right_ticks_per_sec * 2 * np.pi * self.wheel_radius) / self.ticks_per_revolution
        self.current_L_velocity = (left_ticks_per_sec * 2 * np.pi * self.wheel_radius) / self.ticks_per_revolution

        # Store current values for next iteration
        self.last_R_pos = right_pos
        self.last_L_pos = left_pos
        self.last_time = current_time

        # Use PID controllers to adjust PWM values
        right_pwm = self.R_pid.compute(abs(right_ticks_per_sec))
        left_pwm = self.L_pid.compute(abs(left_ticks_per_sec))

        # Apply new PWM values if computed
        if right_pwm is not None:
            self.R_Motor.change_speed(right_pwm)
        if left_pwm is not None:
            self.L_Motor.change_speed(left_pwm)

        # Print debug info
        print(f"Target: {self.target_linear_velocity:.3f} m/s, {self.target_angular_velocity:.3f} rad/s")
        print(f"Current: R={self.current_R_velocity:.3f} m/s, L={self.current_L_velocity:.3f} m/s")
        print(f"PWM: R={right_pwm:.3f}, L={left_pwm:.3f}")

    def call_encoder_interrupt(self):
        self.R_Motor.call_encoder_int()
        self.L_Motor.call_encoder_int()

    def get_motors_pos(self):
        return self.R_Motor.get_pos(), self.L_Motor.get_pos()

    def reset_controllers(self):
        """Reset PID controllers"""
        self.R_pid.reset()
        self.L_pid.reset()
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0