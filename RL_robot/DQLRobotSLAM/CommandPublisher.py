from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


# Constants for robot vel
from constants.constants import (
    LINEAR_SPEED,
    ANGULAR_SPEED,
)


def action_to_cmd(action):
    """Convert action index to Twist command"""
    cmd = Twist()

    if action == 0:  # Stop
        pass  # All values are initialized to 0
    elif action == 1:  # Forward
        cmd.linear.x = LINEAR_SPEED
    elif action == 2:  # Back
        cmd.linear.x = -LINEAR_SPEED
    elif action == 3:  # Right
        cmd.angular.z = -ANGULAR_SPEED
    elif action == 4:  # Left
        cmd.angular.z = ANGULAR_SPEED

    return cmd


class CommandPublisher(Node):
    def __init__(self):
        super().__init__('command_publisher')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def execute_action(self, action):
        """Execute action by publishing to cmd_vel"""
        cmd = action_to_cmd(action)
        self.cmd_vel_pub.publish(cmd)
