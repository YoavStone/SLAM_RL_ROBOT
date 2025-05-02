import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


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
        # Create and broadcast TF transform from odom to base_footprint
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_footprint'

        # Copy position from received odometry
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z

        # Copy orientation from received odometry
        transform.transform.rotation = msg.pose.pose.orientation

        # Broadcast the transform
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