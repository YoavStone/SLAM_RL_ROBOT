import rclpy
from rclpy.node import Node

from .RobotControlNode import RobotControlNode


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RobotControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("clean up command robot node")
        if node is not None:
            node.destroy_node()
            print("Node destroyed.")
        if rclpy.ok():
            rclpy.shutdown()
            print("Node shut down.")

    return 0


if __name__ == '__main__':
    main()