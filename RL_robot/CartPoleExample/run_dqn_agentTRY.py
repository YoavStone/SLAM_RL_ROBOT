import torch
import gymnasium as gym
import rclpy
from rclpy.node import Node

from .DQLAgentTRY import DQLAgent


class DQLAgentNode(Node):
    def __init__(self):
        super().__init__('dql_agent_node')

        # ROS2 parameters
        self.declare_parameter('learning_mode', True)
        self.declare_parameter('model_path', '')

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value

        print(f"Learning mode: {self.learning_mode}")
        print(f"Model path: {self.model_path}")

        # Initialize agent
        self.agent = DQLAgent(learning_mode=self.learning_mode, model_path=self.model_path)

        # Start the agent
        if self.learning_mode:
            print("Starting agent in learning mode")
            self.agent.init_replay_buffer()
            self.agent.train()
        else:
            print("Starting agent in execution mode")
            self.agent.run()


def main(args=None):
    rclpy.init(args=args)
    node = DQLAgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped cleanly")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()