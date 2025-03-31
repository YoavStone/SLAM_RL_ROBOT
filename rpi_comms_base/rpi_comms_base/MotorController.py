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

        # Convert ticks to angular velocity (rad/s)
        self.right_wheel_speed = (delta_right / self.ticks_per_revolution) * 2 * math.pi / dt
        self.left_wheel_speed = (delta_left / self.ticks_per_revolution) * 2 * math.pi / dt

        return self.right_wheel_speed, self.left_wheel_speed