import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math


class LaserScanRotator(Node):
    def __init__(self):
        super().__init__('laser_scan_rotator')

        # Subscribe to the original laser scan
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan_not_rotated',  # Original scan topic
            self.scan_callback,
            10
        )

        # Publisher for the rotated scan
        self.scan_publisher = self.create_publisher(
            LaserScan,
            'scan',  # New topic with rotated data
            10
        )

        self.get_logger().info('Laser scan rotator node initialized')

    def scan_callback(self, msg):
        # Create a new LaserScan message
        rotated_scan = LaserScan()

        # Copy all the header and metadata
        rotated_scan.header = msg.header
        rotated_scan.angle_min = msg.angle_min
        rotated_scan.angle_max = msg.angle_max
        rotated_scan.angle_increment = msg.angle_increment
        rotated_scan.time_increment = msg.time_increment
        rotated_scan.scan_time = msg.scan_time
        rotated_scan.range_min = msg.range_min
        rotated_scan.range_max = msg.range_max

        # Step 1: Rotate the ranges array by 180 degrees (shift by half the array length)
        ranges_length = len(msg.ranges)
        half_length = ranges_length // 2
        rotated_ranges = msg.ranges[half_length:] + msg.ranges[:half_length]

        # Step 2: Mirror flip the rotated array (reverse the entire array)
        rotated_scan.ranges = list(reversed(rotated_ranges))

        # Do the same for intensities if they exist
        if len(msg.intensities) > 0:
            rotated_intensities = msg.intensities[half_length:] + msg.intensities[:half_length]
            rotated_scan.intensities = list(reversed(rotated_intensities))
        else:
            rotated_scan.intensities = []

        # Publish the rotated scan
        self.scan_publisher.publish(rotated_scan)


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanRotator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down laser scan rotator node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()