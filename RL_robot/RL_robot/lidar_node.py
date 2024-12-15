import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class LaserScanSubscriber(Node):
    def __init__(self):

        super().__init__('lidar_scan_interpreter')

        self.publisher_ = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
            )

        self.subscription_ = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10
            )
        
        self.subscription_  # Prevent unused variable warning


    def listener_callback(self, msg):
        all_more = True
        i = 0
        print(msg)

        for range_value in msg.ranges:
            i += 1
            print("indx: ", i, " ranges: ", range_value)
            if range_value < 1.0 and (i < 360+60 and i > 360-60):
                all_more = False
                break

        if all_more:
            twist_msg = Twist()
            twist_msg.linear.x = 0.5
            twist_msg.angular.z = 0.0
        else:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.5

        self.publisher_.publish(twist_msg)


def main(args=None):

    rclpy.init(args=args)

    laser_scan_subscriber = LaserScanSubscriber()

    rclpy.spin(laser_scan_subscriber)

    # Destroy the node explicitly
    laser_scan_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()