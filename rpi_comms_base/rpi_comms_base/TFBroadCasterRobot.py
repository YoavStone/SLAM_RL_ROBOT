import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from transforms3d.euler import quat2euler, euler2quat  # You'll need pip install transforms3d


class TFBroadCasterRobot(Node):
    def __init__(self):
        super().__init__('tf_board_cast_node_for_robot')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_footprint'

        # Copy position from odometry
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z

        # Extract quaternion and convert to Euler angles
        q = [msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w]

        roll, pitch, yaw = quat2euler(q)

        # Add 180 degrees to the yaw (rotate around Z axis)
        yaw += math.pi

        # Convert back to quaternion
        q_new = euler2quat(roll, pitch, yaw)

        # Set the new orientation
        transform.transform.rotation.x = q_new[0]
        transform.transform.rotation.y = q_new[1]
        transform.transform.rotation.z = q_new[2]
        transform.transform.rotation.w = q_new[3]

        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = TFBroadCaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("clean up command robot node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()