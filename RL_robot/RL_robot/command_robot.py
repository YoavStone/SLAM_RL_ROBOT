import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import sys
import tty
import termios
import threading
import time


class BaseToRobot(Node):
    def __init__(self):
        super().__init__('base_talk_to_robot')

        # Create publisher for sending velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Create subscription for receiving odometry data
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)

        # Set default velocities
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.3  # rad/s

        # Store latest position and orientation
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0

        # Thresholds for printing odometry updates
        self.position_threshold = 0.001
        self.orientation_threshold = 0.0017

        # Start keyboard reading thread
        print('Use WASD keys to control the robot:')
        print('W: Move forward')
        print('A: Turn left')
        print('S: Move backward')
        print('D: Turn right')
        print('X: Stop')
        print('Q: Quit')

        self.running = True
        self.thread = threading.Thread(target=self.read_keyboard_input)
        self.thread.daemon = True
        self.thread.start()

        print('Teleop Node initialized')

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

        # Optional: Print position for debugging
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        new_theta = 2 * math.atan2(qz, qw)
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z

        # Check if position or orientation has changed significantly
        position_change = math.sqrt((new_x - self.current_x)**2 + (new_y - self.current_y)**2)
        orientation_change = abs(new_theta - self.current_theta)
        # Normalize orientation change to handle wrapping around +/- pi
        orientation_change = min(orientation_change, 2 * math.pi - orientation_change)

        if position_change >= self.position_threshold or orientation_change >= self.orientation_threshold:
            # Print updated odometry
            print(f'Position: ({new_x:.3f}, {new_y:.3f})')
            print(f'Orientation: {math.degrees(new_theta):.1f} degrees')
            print(f'vel: {linear_velocity:.3f} m/s, {angular_velocity:.3f} rad/s')

            # Update stored values
            self.current_x = new_x
            self.current_y = new_y
            self.current_theta = new_theta

    def read_keyboard_input(self):
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            while self.running:
                # Set terminal to raw mode
                tty.setraw(sys.stdin.fileno())
                # Read a single character
                key = sys.stdin.read(1)
                # Restore terminal settings for logging
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                # Process the key
                self.process_key(key)

                # Brief pause to prevent excessive CPU usage
                time.sleep(0.1)
        except Exception as e:
            print(f'Error reading keyboard input: {e}')
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def process_key(self, key):
        # Create Twist message
        twist = Twist()

        # Set linear and angular velocity based on key pressed
        if key == 'w':
            twist.linear.x = self.linear_speed
            print('Moving forward')
        elif key == 's':
            twist.linear.x = -self.linear_speed
            print('Moving backward')
        elif key == 'a':
            twist.angular.z = self.angular_speed
            print('Turning left')
        elif key == 'd':
            twist.angular.z = -self.angular_speed
            print('Turning right')
        elif key == 'x':
            # Stop the robot
            print('Stopping')
        elif key == 'q':
            # Quit the program
            print('Quitting')
            self.running = False
            self.cmd_vel_publisher.publish(twist)
            rclpy.shutdown()
            return

        # Publish the velocity command
        self.cmd_vel_publisher.publish(twist)

        # Restore terminal settings for next iteration
        tty.setraw(sys.stdin.fileno())

    def cleanup(self):
        # Stop the robot before shutting down
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)
        print('Stopping robot and shutting down')
        self.running = False


def main(args=None):
    rclpy.init(args=args)
    node = BaseToRobot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()