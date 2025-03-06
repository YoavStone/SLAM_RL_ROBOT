import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ControlMotorsPublisher(Node):
    def __init__(self):
        super().__init__('control_motors_pub')
        self.publisher_ = self.create_publisher(String, 'control_motors_topic', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, world! %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    print("make publisher:")
    rclpy.init(args=args)
    publisher_node = ControlMotorsPublisher()
    rclpy.spin(publisher_node)
    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()