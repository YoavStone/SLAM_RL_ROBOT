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

        # Parameters
        self.declare_parameter('publish_rate', 30.0)
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        # Variables to track the offset between raw odometry and corrected odometry
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.yaw_offset = 0.0

        # Most recent odometry data
        self.latest_odom = None
        self.latest_odom_time = None

        # Subscribe to raw odometry from diff_drive
        self.odom_diff_sub = self.create_subscription(
            Odometry,
            'odom_diff',
            self.odom_diff_callback,
            10
        )

        # Subscribe to tf from diff_drive
        self.tf_diff_sub = self.create_subscription(
            TFMessage,
            'tf_diff',
            self.tf_diff_callback,
            10
        )

        # Subscribe to reset signals
        self.reset_sub = self.create_subscription(
            Empty,
            'episode_end',
            self.reset_callback,
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
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_corrected_odom)

        self.get_logger().info('Odometry Middleman Node initialized')

    def odom_diff_callback(self, msg):
        """
        Process incoming raw odometry data from diff_drive.
        """
        self.latest_odom = msg
        self.latest_odom_time = self.get_clock().now()

        # Don't publish here - let the timer handle it for consistent rate

    def tf_diff_callback(self, msg):
        """
        Process incoming tf data from diff_drive.
        """
        if self.resetting:
            # Find the odom_diff -> base_footprint transform
            for transform in msg.transforms:
                if (transform.header.frame_id == 'odom_diff' and
                        transform.child_frame_id == 'base_footprint'):
                    # No need to store this - just modify and republish immediately
                    corrected_tf = TransformStamped()
                    corrected_tf.header.stamp = transform.header.stamp
                    corrected_tf.header.frame_id = 'odom'  # Change the frame_id
                    corrected_tf.child_frame_id = transform.child_frame_id

                    # Apply the correction to the transform
                    x_raw = transform.transform.translation.x
                    y_raw = transform.transform.translation.y

                    # Extract yaw from quaternion
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w

                    # Convert to yaw
                    siny_cosp = 2.0 * (qw * qz + qx * qy)
                    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw_raw = math.atan2(siny_cosp, cosy_cosp)

                    # Apply offsets
                    x_corrected = x_raw - self.x_offset
                    y_corrected = y_raw - self.y_offset
                    yaw_corrected = yaw_raw - self.yaw_offset

                    # Convert back to quaternion
                    cy = math.cos(yaw_corrected * 0.5)
                    sy = math.sin(yaw_corrected * 0.5)
                    cp = math.cos(0.0)  # pitch = 0
                    sp = math.sin(0.0)
                    cr = math.cos(0.0)  # roll = 0
                    sr = math.sin(0.0)

                    corrected_tf.transform.rotation.x = sr * cp * cy - cr * sp * sy
                    corrected_tf.transform.rotation.y = cr * sp * cy + sr * cp * sy
                    corrected_tf.transform.rotation.z = cr * cp * sy - sr * sp * cy
                    corrected_tf.transform.rotation.w = cr * cp * cy + sr * sp * sy

                    # Set translation
                    corrected_tf.transform.translation.x = x_corrected
                    corrected_tf.transform.translation.y = y_corrected
                    corrected_tf.transform.translation.z = transform.transform.translation.z

                    # Publish the corrected transform
                    self.tf_broadcaster.sendTransform(corrected_tf)
                    break

    def publish_corrected_odom(self):
        """
        Publish corrected odometry message at a fixed rate.
        """
        if self.latest_odom is None:
            return  # No data yet

        # Create a new odometry message
        corrected_odom = Odometry()
        corrected_odom.header.stamp = self.get_clock().now().to_msg()
        corrected_odom.header.frame_id = 'odom'  # Changed from odom_diff
        corrected_odom.child_frame_id = 'base_footprint'

        # Copy covariance data
        corrected_odom.pose.covariance = self.latest_odom.pose.covariance
        corrected_odom.twist.covariance = self.latest_odom.twist.covariance

        # Apply correction to position
        x_raw = self.latest_odom.pose.pose.position.x
        y_raw = self.latest_odom.pose.pose.position.y
        z_raw = self.latest_odom.pose.pose.position.z

        corrected_odom.pose.pose.position.x = x_raw - self.x_offset
        corrected_odom.pose.pose.position.y = y_raw - self.y_offset
        corrected_odom.pose.pose.position.z = z_raw

        # Extract orientation as quaternion
        qx = self.latest_odom.pose.pose.orientation.x
        qy = self.latest_odom.pose.pose.orientation.y
        qz = self.latest_odom.pose.pose.orientation.z
        qw = self.latest_odom.pose.pose.orientation.w

        # Convert to yaw angle
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_raw = math.atan2(siny_cosp, cosy_cosp)

        # Apply yaw offset
        yaw_corrected = yaw_raw - self.yaw_offset

        # Convert back to quaternion
        cy = math.cos(yaw_corrected * 0.5)
        sy = math.sin(yaw_corrected * 0.5)
        cp = math.cos(0.0)  # pitch = 0
        sp = math.sin(0.0)
        cr = math.cos(0.0)  # roll = 0
        sr = math.sin(0.0)

        corrected_odom.pose.pose.orientation.x = sr * cp * cy - cr * sp * sy
        corrected_odom.pose.pose.orientation.y = cr * sp * cy + sr * cp * sy
        corrected_odom.pose.pose.orientation.z = cr * cp * sy - sr * sp * cy
        corrected_odom.pose.pose.orientation.w = cr * cp * cy + sr * sp * sy

        # Copy twist (velocity) data - no correction needed
        corrected_odom.twist.twist = self.latest_odom.twist.twist

        # Publish corrected odometry
        self.odom_pub.publish(corrected_odom)

    def reset_callback(self, msg):
        print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        """
        Handle reset signal to update offsets.
        """
        self.resetting = True

        self.get_logger().info('Received reset signal, updating odometry offsets')

        # We need current odometry to calculate offsets
        if self.latest_odom is None:
            self.get_logger().warn('Cannot reset odometry - no data received yet')
            return

        # Calculate new offsets to zero out the current position
        self.x_offset = self.latest_odom.pose.pose.position.x
        self.y_offset = self.latest_odom.pose.pose.position.y

        # Extract current yaw
        qx = self.latest_odom.pose.pose.orientation.x
        qy = self.latest_odom.pose.pose.orientation.y
        qz = self.latest_odom.pose.pose.orientation.z
        qw = self.latest_odom.pose.pose.orientation.w

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Update yaw offset
        self.yaw_offset = current_yaw

        self.get_logger().info(
            f'Updated odometry offsets: x={self.x_offset:.2f}, y={self.y_offset:.2f}, yaw={self.yaw_offset:.2f}')

        # Immediately publish zeroed odometry
        self.publish_zeroed_odom()

        self.resetting = False

    def publish_zeroed_odom(self):
        """
        Publish zeroed odometry immediately after reset.
        """
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