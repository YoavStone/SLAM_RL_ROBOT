import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from transforms3d.euler import quat2euler, euler2quat


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

        # Invert the X and Y positions (flip 180 degrees)
        transform.transform.translation.x = -msg.pose.pose.position.x
        transform.transform.translation.y = -msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z

        # For orientation, we need to handle it differently
        # Rotate 180 degrees around the Z axis (invert both x and y)
        q_orig = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

        # This approach inverts the X and Y components of rotation
        # which effectively produces a 180-degree rotation around Z
        q_new = [
            -q_orig[0],  # Invert X
            -q_orig[1],  # Invert Y
            q_orig[2],  # Keep Z the same
            q_orig[3]  # Keep W the same
        ]

        # Normalize the quaternion to ensure it remains valid
        magnitude = math.sqrt(sum(x * x for x in q_new))
        q_new = [x / magnitude for x in q_new]

        transform.transform.rotation.x = q_new[0]
        transform.transform.rotation.y = q_new[1]
        transform.transform.rotation.z = q_new[2]
        transform.transform.rotation.w = q_new[3]

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