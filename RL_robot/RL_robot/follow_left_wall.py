from time import sleep

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math


class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_ = self.create_subscription(LaserScan, '/scan', self.listener_callback, 10)
        self.subscription_  # prevent unused variable warning

        self.desired_distance = 0.6  # distance from the wall
        self.forward_speed = 0.3  # 0.3 for normal speed
        self.angular_speed = 0.6  # 0.6 for normal speed
        self.region_front = 1.0
        self.region_left = 1.0
        self.region_right = 1.0
        self.error_from_wall = 0.45
        self.wall_found = False
        self.dont_update_for_secs = 0.2  # sleep for more continues actions


    def listener_callback(self, msg):
        self.process_laser_scan(msg)
        self.follow_wall()

    def process_laser_scan(self, msg):
        ranges = msg.ranges
        num_ranges = len(ranges)

        print(num_ranges)

        # Define regions for front, left, and right
        # Front is now centered at index 0, spanning +/-35 degrees
        front_indices = list(range(num_ranges - 35, num_ranges)) + list(range(0, 35 + 1))
        right_indices = range(num_ranges - 35 - 60, num_ranges - 35)  # right 60/2 degrees
        left_indices = range(35 + 1, 35 + 60 + 1)  # left 60/2 degrees

        # Calculate minimum distances in each region
        self.region_front = min([ranges[i] for i in front_indices if not math.isinf(ranges[i])], default=float('inf'))
        self.region_left = min([ranges[i] for i in left_indices if not math.isinf(ranges[i])], default=float('inf'))
        self.region_right = min([ranges[i] for i in right_indices if not math.isinf(ranges[i])], default=float('inf'))

        print("front closest: ", self.region_front)
        print("left closest: ", self.region_left)
        print("right closest: ", self.region_right)

    def is_searching_wall(self):
        if self.region_front - self.desired_distance < self.error_from_wall:
            return False
        if self.region_left - self.desired_distance < self.error_from_wall:
            return False
        if self.region_right - self.desired_distance < self.error_from_wall:
            return False
        return True

    def init_wall_search(self):
        if self.is_searching_wall():
            return False
        self.wall_found = True
        return True

    def follow_wall(self):

        twist_msg = Twist()

        sleep(self.dont_update_for_secs)

        error_left = self.region_left - self.desired_distance

        if not self.init_wall_search() and not self.wall_found:
            print("SEARCH: ", self.region_front)
            twist_msg.linear.x = self.forward_speed
            twist_msg.angular.z = 0.0

        elif not self.is_searching_wall() and self.wall_found:
            if self.desired_distance+self.error_from_wall > self.region_front:  # if too close to front
                print("too front turn right: ", self.region_front)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = -self.angular_speed  # Turn right if front is too close
            elif error_left > self.error_from_wall/1.2: # if left wall is too far, turn left
                print("left wall too far, turn left: ", self.region_left)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = self.angular_speed
            elif error_left < self.error_from_wall/-1.2: # if left wall is too close, turn right
                print("left wall too close, turn right: ", self.region_left)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = -self.angular_speed
            else:
                print("GO: ", self.region_front)
                twist_msg.linear.x = self.forward_speed
                twist_msg.angular.z = 0.0

        else:
            print("WALL LEFT TURN LEFT: ", self.region_front)
            print("left wall too far, turn left: ", self.region_left)
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = self.angular_speed

        self.publisher_.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()