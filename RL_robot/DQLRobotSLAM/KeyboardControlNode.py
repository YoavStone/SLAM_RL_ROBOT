#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
import sys
import termios
import tty
import threading
import time


class KeyboardControlNode(Node):
    """Node for keyboard control of demonstration mode"""

    def __init__(self):
        super().__init__('keyboard_control_node')

        # Create publishers
        self.action_pub = self.create_publisher(Int32, '/demo/action', 10)
        self.toggle_pub = self.create_publisher(Bool, '/demo/toggle', 10)

        self.get_logger().info("Keyboard Control Node initialized")
        self.get_logger().info("Controls:")
        self.get_logger().info("p: Toggle demonstration mode on/off")
        self.get_logger().info("w: Move forward")
        self.get_logger().info("a: Turn left")
        self.get_logger().info("s: Move backward")
        self.get_logger().info("d: Turn right")
        self.get_logger().info("x: Stop")
        self.get_logger().info("q: Quit")

        # Start keyboard input thread
        self.running = True
        self.thread = threading.Thread(target=self.read_keyboard_input)
        self.thread.daemon = True
        self.thread.start()

    def read_keyboard_input(self):
        """Thread for reading keyboard input"""
        # Check if stdin is a TTY
        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY, keyboard input disabled")
            return

        # Save terminal settings
        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().error(f"Error getting terminal settings: {e}")
            return

        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            while self.running:
                # Read a single character
                key = sys.stdin.read(1)
                # Restore terminal settings for logging
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                # Process the key
                self.process_key(key)

                # Set terminal back to raw mode for next input
                tty.setraw(sys.stdin.fileno())

                # Brief pause to prevent excessive CPU usage
                time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f"Error in keyboard thread: {e}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def process_key(self, key):
        """Process keyboard input and publish to topics"""
        if key == 'p':
            # Toggle demonstration mode
            msg = Bool()
            msg.data = True
            self.toggle_pub.publish(msg)
            self.get_logger().info("Toggling demonstration mode")
        elif key == 'w':
            # Forward
            self.publish_action(1)
            self.get_logger().info("Action: Forward")
        elif key == 's':
            # Backward
            self.publish_action(2)
            self.get_logger().info("Action: Backward")
        elif key == 'd':
            # Turn Right
            self.publish_action(3)
            self.get_logger().info("Action: Turn Right")
        elif key == 'a':
            # Turn Left
            self.publish_action(4)
            self.get_logger().info("Action: Turn Left")
        elif key == 'x':
            # Stop
            self.publish_action(0)
            self.get_logger().info("Action: Stop")
        elif key == 'q':
            # Quit
            self.get_logger().info("Exiting keyboard control")
            self.running = False
            self.destroy_node()
            rclpy.shutdown()

    def publish_action(self, action):
        """Publish an action to the action topic"""
        msg = Int32()
        msg.data = action
        self.action_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()