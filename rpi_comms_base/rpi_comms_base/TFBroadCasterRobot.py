import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import qmult


class TFBroadCasterRobot(Node):
    def __init__(self):
        super().__init__('tf_board_cast_node_for_robot')
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create a rotation quaternion for 180 degrees around Z axis
        # Order in transforms3d is [w, x, y, z]
        self.rot_180_z = euler2quat(0, 0, math.pi)

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

        # Get original position
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        pos_z = msg.pose.pose.position.z

        # Invert X and Y for 180-degree rotation
        transform.transform.translation.x = -pos_x
        transform.transform.translation.y = -pos_y
        transform.transform.translation.z = pos_z

        # Get original orientation as [w, x, y, z] for transforms3d
        q_orig = [
            msg.pose.pose.orientation.w,  # Note: w first for transforms3d
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ]

        # Apply 180-degree rotation around Z-axis
        # qmult multiplies quaternions
        q_new = qmult(self.rot_180_z, q_orig)

        # Set the new orientation (convert back to [x, y, z, w] for ROS)
        transform.transform.rotation.w = q_new[0]
        transform.transform.rotation.x = q_new[1]
        transform.transform.rotation.y = q_new[2]
        transform.transform.rotation.z = q_new[3]

        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = TFBroadCasterRobot()
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