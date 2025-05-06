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

        self.desired_distance = 0.6  # distance from the wall
        self.forward_speed = 0.3  # 0.3 for normal speed
        self.angular_speed = 1.2  # 1.2 for normal speed
        self.dist_front_wall = 1.0
        self.dist_left_wall = 1.0
        self.dist_right_wall = 1.0
        self.allowed_dist_delta = 0.20
        self.wall_found = False
        self.turn_delay = 0.05  # sleep for more continues actions
        self.forward_delay = 0.25  # sleep for more continues actions

    def listener_callback(self, msg):
        self.process_laser_scan(msg)
        self.follow_wall()

    def process_laser_scan(self, msg):
        ranges = msg.ranges
        num_ranges = len(ranges)

        print(num_ranges)
        circle_size = int(num_ranges/360)

        # Define regions for front, left, and right
        # Front is now centered at index 0, spanning +/-35 degrees
        front_indices = list(range(num_ranges - 35*circle_size, num_ranges)) + list(range(0, 35*circle_size + 1))
        right_indices = range(num_ranges - 35*circle_size - 60*circle_size, num_ranges - 35*circle_size)  # right 60 degrees
        left_indices = range(35*circle_size + 1, 35*circle_size + 60*circle_size + 1)  # left 60 degrees

        # Calculate minimum distances in each region
        self.dist_front_wall = min([ranges[i] for i in front_indices if not math.isinf(ranges[i])], default=float('inf'))
        self.dist_left_wall = min([ranges[i] for i in left_indices if not math.isinf(ranges[i])], default=float('inf'))
        self.dist_right_wall = min([ranges[i] for i in right_indices if not math.isinf(ranges[i])], default=float('inf'))

        print("front closest: ", self.dist_front_wall)
        print("left closest: ", self.dist_left_wall)
        print("right closest: ", self.dist_right_wall)

    def is_searching_wall(self):
        if self.dist_front_wall - self.desired_distance < self.allowed_dist_delta + 0.15:
            return False
        if self.dist_left_wall - self.desired_distance < self.allowed_dist_delta + 0.15:
            return False
        if self.dist_right_wall - self.desired_distance < self.allowed_dist_delta + 0.15:
            return False
        return True

    def init_wall_search(self):
        if not self.is_searching_wall():
            self.wall_found = True

    def follow_wall(self):

        twist_msg = Twist()

        if not self.wall_found:
            self.init_wall_search()
            print("SEARCH: ", self.dist_front_wall)
            twist_msg.linear.x = self.forward_speed
            twist_msg.angular.z = 0.0
            self.publisher_.publish(twist_msg)
            sleep(self.forward_delay)

        elif not self.is_searching_wall():
            if self.dist_front_wall < self.desired_distance + self.allowed_dist_delta/1.5:  # if too close to front
                print("too close to front turn right: ", self.dist_front_wall)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = -self.angular_speed  # Turn right if front is too close
                self.publisher_.publish(twist_msg)
                sleep(self.turn_delay)
            elif self.dist_left_wall > self.desired_distance + self.allowed_dist_delta/2: # if left wall is too far, turn left
                print("left wall too far, turn left: ", self.dist_left_wall)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = self.angular_speed
                self.publisher_.publish(twist_msg)
                sleep(self.turn_delay)
            elif self.dist_left_wall < self.desired_distance - self.allowed_dist_delta/2: # if left wall is too close, turn right
                print("left wall too close, turn right: ", self.dist_left_wall)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = -self.angular_speed
                self.publisher_.publish(twist_msg)
                sleep(self.turn_delay)
            else:
                print("GO: ", self.dist_front_wall)
                twist_msg.linear.x = self.forward_speed
                twist_msg.angular.z = 0.0
                self.publisher_.publish(twist_msg)
                sleep(self.forward_delay)

        else:
            print("WALL LEFT TURN LEFT: ", self.dist_front_wall)
            print("left wall too far, turn left: ", self.dist_left_wall)
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = self.angular_speed
            self.publisher_.publish(twist_msg)
            sleep(self.turn_delay)


def main(args=None):
    rclpy.init(args=args)

    wall_follower = WallFollower()
    try:
        rclpy.spin(wall_follower)
    except KeyboardInterrupt:
        wall_follower.publisher_.publish(Twist())  # stop robot
        print("Node stopped cleanly")
        sleep(0.5)
    finally:
        if wall_follower is not None:
            wall_follower.destroy_node()
            print("Node destroyed.")
        if rclpy.ok():
            rclpy.shutdown()
            print("Node shut down.")

        return 0

if __name__ == '__main__':
    main()