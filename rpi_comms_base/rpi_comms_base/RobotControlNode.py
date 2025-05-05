import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

from .ArduinoMotorsSynchronizer import ArduinoMotorsSynchronizer
from .MotorsController import MotorsController
from .RobotPositionCalculator import RobotPositionCalculator
from .VelocityToMotorsCmdConvertor import VelocityToMotorsCmdConvertor


class RobotControlNode(Node):
    def __init__(self):
        print("Initializing robot control node with Arduino interface")
        super().__init__('robot_control_node')

        # Define parameters
        self.declare_parameter('arduino_port', '/dev/ttyACM0')
        self.declare_parameter('arduino_baudrate', 115200)
        self.declare_parameter('wheel_radius', 0.034)  # meters
        self.declare_parameter('wheel_separation', 0.34)  # meters
        self.declare_parameter('ticks_per_revolution', 170*2)  # based on encoder

        # Get parameters
        arduino_port = self.get_parameter('arduino_port').value
        arduino_baudrate = self.get_parameter('arduino_baudrate').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.ticks_per_revolution = self.get_parameter('ticks_per_revolution').value

        # Initialize motors synchronizer with Arduino connection
        try:
            self.motors_synchronizer = ArduinoMotorsSynchronizer(
                port=arduino_port,
                baudrate=arduino_baudrate,
                timeout=1.0
            )
            print(f"Connected to Arduino on {arduino_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Arduino: {e}")
            raise

        # Speed multiplier factor
        self.linear_speed_factor = 1.0
        self.turn_speed_factor = 1.0

        # Initialize motor controller
        self.motor_controller = MotorsController(self.motors_synchronizer, self.ticks_per_revolution)
        self.closed_loop_speed_control_timer = self.create_timer(0.05, self.motor_controller.closed_loop_control_speed)  # 20Hz update rate

        # Initialize position calculator
        self.robot_position_calculator = RobotPositionCalculator(
            self.motors_synchronizer,
            self.wheel_radius,
            self.wheel_separation,
            self.ticks_per_revolution
        )

        # Initialize velocity converter
        self.vel_to_motors_cmd_convertor = VelocityToMotorsCmdConvertor(
            self.motors_synchronizer,
            self.motor_controller,
            self.wheel_separation,
            self.wheel_radius
        )

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
            'odom',
            10
        )

        # Time tracking
        self.last_time = time.time()

        # Set up timer for regular position updates
        self.publisher_timer = self.create_timer(0.1, self.publish_position)  # 10Hz update rate

        print('Robot Control Node with Arduino interface has been initialized')

    def velocity_callback(self, msg):
        # Convert received velocities to motor commands
        self.vel_to_motors_cmd_convertor.convert_vel_to_motor_dir(msg)

    def publish_position(self):
        # Get current motor positions and publish odometry
        odom = self.robot_position_calculator.create_odom_message(self.get_clock().now().to_msg())
        self.odom_publisher.publish(odom)

    def on_shutdown(self):
        """Clean up on node shutdown."""
        self.get_logger().info("Shutting down RobotControlNode")
        if hasattr(self, 'motors_synchronizer'):
            self.motors_synchronizer.stop()
            self.motors_synchronizer.close()