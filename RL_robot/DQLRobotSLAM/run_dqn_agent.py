import rclpy
from rclpy.node import Node

from .DQLAgent import DQLAgent


def main(args=None):
    rclpy.init(args=args)

    agent_node = DQLAgent()

    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        print("Node stopped cleanly")
    finally:
        # Clean up
        agent_node.env.close()
        agent_node.destroy_node()
        rclpy.shutdown()

        return 0


if __name__ == "__main__":
    main()