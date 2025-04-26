#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import time


class TeleportService(Node):
    """
    A dedicated service node for teleporting the robot in Gazebo.
    This node runs in its own process with its own environment.
    """

    def __init__(self):
        super().__init__('teleport_service')

        # Subscribe to teleport commands
        self.subscription = self.create_subscription(
            String,
            '/teleport_command',
            self.teleport_callback,
            10
        )

        self.get_logger().info("Teleport service initialized")

    def teleport_callback(self, msg):
        """Handle teleport commands"""
        # Format: "x,y,yaw"
        try:
            x_str, y_str, yaw_str = msg.data.split(',')
            x = float(x_str.strip())
            y = float(y_str.strip())
            yaw = float(yaw_str.strip())

            self.get_logger().info(f"Teleporting robot to: ({x}, {y}, {yaw})")
            self.execute_teleport(x, y, yaw)

        except Exception as e:
            self.get_logger().error(f"Error processing teleport command: {e}")

    def execute_teleport(self, x, y, yaw):
        """Execute teleport command with retry logic"""
        import math

        # Calculate quaternion
        import math
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)

        # Try teleport command
        cmd = [
            'gz', 'service', '-s', '/world/empty/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req',
            f'name: "mapping_robot", position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: {qw}, x: 0.0, y: 0.0, z: {qz}}}'
        ]

        self.get_logger().info(f"Executing command: {' '.join(cmd)}")

        # Try the command multiple times
        for attempt in range(3):
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=3.0)

                output = stdout.decode() if stdout else ""
                error = stderr.decode() if stderr else ""

                if error:
                    self.get_logger().warn(f"Command error (attempt {attempt + 1}): {error}")
                    time.sleep(1.0)
                    continue

                self.get_logger().info(f"Teleport successful (attempt {attempt + 1})")
                return True

            except Exception as e:
                self.get_logger().error(f"Command exception (attempt {attempt + 1}): {e}")
                time.sleep(1.0)

        self.get_logger().error("All teleport attempts failed")
        return False


def main(args=None):
    rclpy.init(args=args)
    teleport_service = TeleportService()

    try:
        rclpy.spin(teleport_service)
    except KeyboardInterrupt:
        pass
    finally:
        teleport_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()