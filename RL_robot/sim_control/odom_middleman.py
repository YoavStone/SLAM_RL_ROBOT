import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, PoseWithCovariance, TwistWithCovariance
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Empty
import numpy as np
import math
import time


class OdomMiddleman(Node):
    """
    Middleman node that subscribes to odom_diff from diff_drive,
    applies corrections, and republishes to odom.
    """

    def __init__(self):
        super().__init__('odom_middleman')

        # Subscribe to reset signals
        self.reset_sub = self.create_subscription(
            Empty,
            'episode_end',
            self.publish_zeroed_odom,
            10
        )

        # Publisher for corrected odometry
        self.odom_pub = self.create_publisher(
            Odometry,
            'odom',
            10
        )

        self.resetting = True

        # TF broadcaster for corrected transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing at a fixed rate
        self.get_logger().info('Odometry Middleman Node initialized')

    def publish_zeroed_odom(self):
        """
        Publish zeroed odometry immediately after reset.
        """
        self.resetting = True

        self.get_logger().info('Received reset signal, updating odometry offsets')

        zero_odom = Odometry()
        zero_odom.header.stamp = self.get_clock().now().to_msg()
        zero_odom.header.frame_id = 'odom'
        zero_odom.child_frame_id = 'base_footprint'

        # Set position to zero
        zero_odom.pose.pose.position.x = 0.0
        zero_odom.pose.pose.position.y = 0.0
        zero_odom.pose.pose.position.z = 0.0

        # Set orientation to identity quaternion
        zero_odom.pose.pose.orientation.x = 0.0
        zero_odom.pose.pose.orientation.y = 0.0
        zero_odom.pose.pose.orientation.z = 0.0
        zero_odom.pose.pose.orientation.w = 1.0

        # Set twist to zero
        zero_odom.twist.twist.linear.x = 0.0
        zero_odom.twist.twist.linear.y = 0.0
        zero_odom.twist.twist.linear.z = 0.0
        zero_odom.twist.twist.angular.x = 0.0
        zero_odom.twist.twist.angular.y = 0.0
        zero_odom.twist.twist.angular.z = 0.0

        # Publish zeroed odometry
        self.odom_pub.publish(zero_odom)

        # Also publish zeroed TF
        zero_tf = TransformStamped()
        zero_tf.header.stamp = self.get_clock().now().to_msg()
        zero_tf.header.frame_id = 'odom'
        zero_tf.child_frame_id = 'base_footprint'

        zero_tf.transform.translation.x = 0.0
        zero_tf.transform.translation.y = 0.0
        zero_tf.transform.translation.z = 0.0

        zero_tf.transform.rotation.x = 0.0
        zero_tf.transform.rotation.y = 0.0
        zero_tf.transform.rotation.z = 0.0
        zero_tf.transform.rotation.w = 1.0

        # Publish zero transform multiple times to ensure it's applied
        for _ in range(100):
            self.tf_broadcaster.sendTransform(zero_tf)
            time.sleep(0.2)

        self.get_logger().info('Published zeroed odometry after reset')

        self.resetting = False


def main(args=None):
    rclpy.init(args=args)
    node = OdomMiddleman()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()