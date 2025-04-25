import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

from .MotorsSynchronizer import MotorsSynchronizer
from .MotorsController import MotorsController
from .RobotPositionCalculator import RobotPositionCalculator
from .VelocityToMotorsCmdConvertor import VelocityToMotorsCmdConvertor


class RobotControlNode(Node):
    def __init__(self):
        print("will initialized robot control node")
        super().__init__('robot_control_node')

        # Initialize motors synchronizer
        # Define GPIO pins
        in1 = 17
        in2 = 27
        in3 = 23
        in4 = 24

        pwmR = 22
        pwmL = 25

        ena1 = 19
        enb1 = 26
        ena2 = 13
        enb2 = 6

        FREQ = 1000

        self.motors_synchronizer = MotorsSynchronizer(in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2)
        self.motors_synchronizer.call_encoder_interrupt()

        # Initialize velocity subscriber
        self.velocity_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.velocity_callback,
            10
        )

        # Initialize position publisher
        self.odom_publisher = self.create_publisher(
            Odometry,
            'odom_robot',
            10
        )

        # Set up parameters for odometry calculation
        self.wheel_radius = 0.034  # meters
        self.wheel_separation = 0.34  # meters
        self.ticks_per_revolution = 170  # based on your encoder

        # Speed multiplier factor
        self.linear_speed_factor = 1.0
        self.turn_speed_factor = 1.0

        self.motor_controller = MotorsController(self.motors_synchronizer, self.ticks_per_revolution)
        self.closed_loop_speed_control_timer = self.create_timer(0.05, self.motor_controller.closed_loop_control_speed)  # 20Hz update rate

        # Time tracking
        self.last_time = time.time()

        self.robot_position_calculator = RobotPositionCalculator(self.motors_synchronizer, self.wheel_radius, self.wheel_separation, self.ticks_per_revolution)
        self.vel_to_motors_cmd_convertor = VelocityToMotorsCmdConvertor(self.motors_synchronizer, self.motor_controller, self.wheel_radius, self.wheel_separation)

        # Set up timer for regular position updates
        self.publisher_timer = self.create_timer(0.1, self.publish_position)  # 10Hz update rate

        print('Robot Control Node has been initialized')

    def velocity_callback(self, msg):
        # Convert received velocities to motor commands
        self.vel_to_motors_cmd_convertor.convert_vel_to_motor_dir(msg)

    def publish_position(self):
        # Get current motor positions
        odom = self.robot_position_calculator.create_odom_message(self.get_clock().now().to_msg())
        # Publish the message
        self.odom_publisher.publish(odom)