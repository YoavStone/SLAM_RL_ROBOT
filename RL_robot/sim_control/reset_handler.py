import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import subprocess
import time


class SimpleResetHandler(Node):
    """Node that handles clearing the SLAM map without restarting the node"""

    def __init__(self):
        super().__init__('simple_reset_handler')

        # Create a subscriber for reset requests
        self.reset_sub = self.create_subscription(
            Empty,
            '/environment_reset_request',
            self.reset_callback,
            10
        )

        # Publisher to send clear_map requests
        self.clear_map_pub = self.create_publisher(
            Empty,
            '/slam_toolbox/clear_map',
            10
        )

        # Look for available SLAM services
        self.find_slam_services()

        self.get_logger().info("Simple reset handler initialized")

    def find_slam_services(self):
        """Find available SLAM services for later use"""
        try:
            result = subprocess.run(['ros2', 'service', 'list'],
                                    capture_output=True, text=True, timeout=2.0)

            if result.returncode == 0:
                services = result.stdout.strip().split('\n')
                slam_services = [s for s in services if 'slam' in s.lower()]

                if slam_services:
                    self.get_logger().info("Found SLAM services:")
                    for service in slam_services:
                        self.get_logger().info(f"  - {service}")
                else:
                    self.get_logger().warn("No SLAM services found")
            else:
                self.get_logger().error(f"Failed to list services: {result.stderr}")

        except Exception as e:
            self.get_logger().error(f"Error finding SLAM services: {e}")

    def reset_callback(self, msg):
        """Handle reset request by clearing the map"""
        self.get_logger().info("Received reset request, clearing SLAM map")

        # Method 1: Publish to clear_map topic (multiple times)
        for _ in range(10):
            self.clear_map_pub.publish(Empty())
            time.sleep(0.1)

        # Method 2: Try using ROS 2 CLI
        try:
            self.get_logger().info("Trying service call via CLI")
            subprocess.run(
                ['ros2', 'service', 'call', '/slam_toolbox/clear_map', 'std_srvs/srv/Empty', '{}'],
                timeout=2.0
            )
        except Exception as e:
            self.get_logger().warn(f"Service call via CLI failed: {e}")

        # Method 3: Try topic publication via CLI
        try:
            self.get_logger().info("Trying topic pub via CLI")
            subprocess.run(
                ['ros2', 'topic', 'pub', '--once', '/slam_toolbox/clear_map', 'std_msgs/msg/Empty', '{}'],
                timeout=2.0
            )
        except Exception as e:
            self.get_logger().warn(f"Topic pub via CLI failed: {e}")

        self.get_logger().info("Map clearing attempts completed")


def main(args=None):
    rclpy.init(args=args)
    handler = SimpleResetHandler()

    try:
        rclpy.spin(handler)
    except KeyboardInterrupt:
        pass
    finally:
        handler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()