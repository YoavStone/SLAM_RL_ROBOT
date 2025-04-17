import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
import math
import time
import subprocess


class TeleportService(Node):
    """Service to teleport the robot in Gazebo Harmonic"""

    def __init__(self):
        super().__init__('teleport_service')

        # Create necessary publishers
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # For directly publishing to odometry (force reset)
        self.odom_pub = self.create_publisher(
            Odometry,
            'odom',
            10
        )

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('Teleport service initialized')

    def broadcast_tf(self, x, y, yaw):
        """Broadcast TF transform for the robot position"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        # Set translation
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        # Set rotation (quaternion)
        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)

    def teleport_robot(self, x, y, yaw=0.0):
        """Teleportation for Gazebo Harmonic using alternative methods"""
        try:
            # For Gazebo Harmonic, we'll use a combination of approaches

            # Approach 1: Try using the gz command line tool for teleportation
            model_name = 'mapping_robot'  # Your robot's name in Gazebo
            try:
                cmd = [
                    'gz', 'model', '-m', model_name,
                    '-p', f'{x},{y},0.05',  # x,y,z
                    '-o', f'0,0,{yaw}'  # roll,pitch,yaw
                ]
                subprocess.run(cmd, timeout=1.0)
                self.get_logger().info(f"Used gz command line to teleport to ({x}, {y}, {yaw})")
            except Exception as e:
                self.get_logger().warn(f"gz command failed: {e}, trying alternative methods")

            # Approach 2: Use initialpose (works with navigation stack)
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = self.get_clock().now().to_msg()

            # Set position
            pose_msg.pose.pose.position.x = x
            pose_msg.pose.pose.position.y = y
            pose_msg.pose.pose.position.z = 0.0

            # Set orientation as quaternion
            pose_msg.pose.pose.orientation.z = math.sin(yaw / 2)
            pose_msg.pose.pose.orientation.w = math.cos(yaw / 2)

            # Set covariance (very low uncertainty)
            for i in range(36):
                pose_msg.pose.covariance[i] = 0.0
            pose_msg.pose.covariance[0] = 0.001  # x
            pose_msg.pose.covariance[7] = 0.001  # y
            pose_msg.pose.covariance[35] = 0.001  # yaw

            # Publish multiple times with delays
            for _ in range(10):
                self.initial_pose_pub.publish(pose_msg)
                time.sleep(0.05)

            self.get_logger().info(f'Robot teleport via initialpose to ({x}, {y}, {yaw})')

            # Approach 3: Force odometry reset
            odom_msg = Odometry()
            odom_msg.header.frame_id = "odom"
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.child_frame_id = "base_link"

            # Set position
            odom_msg.pose.pose.position.x = x
            odom_msg.pose.pose.position.y = y
            odom_msg.pose.pose.position.z = 0.0

            # Set orientation
            odom_msg.pose.pose.orientation.z = math.sin(yaw / 2)
            odom_msg.pose.pose.orientation.w = math.cos(yaw / 2)

            # Set covariance (low uncertainty)
            for i in range(36):
                odom_msg.pose.covariance[i] = 0.0
            odom_msg.pose.covariance[0] = 0.01  # x
            odom_msg.pose.covariance[7] = 0.01  # y
            odom_msg.pose.covariance[35] = 0.01  # yaw

            # Set zero velocity
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0

            # Publish multiple times to ensure it gets through
            for _ in range(5):
                self.odom_pub.publish(odom_msg)
                time.sleep(0.1)

            # Approach 4: Direct TF broadcast
            for _ in range(10):
                self.broadcast_tf(x, y, yaw)
                time.sleep(0.05)

            # Approach 5: Publish stop commands to ensure robot doesn't move
            stop_cmd = Twist()
            for _ in range(5):
                self.cmd_vel_pub.publish(stop_cmd)
                time.sleep(0.05)

            self.get_logger().info(f'Completed teleport sequence for ({x}, {y}, {yaw})')
            return True
        except Exception as e:
            self.get_logger().error(f'Error teleporting robot: {str(e)}')
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