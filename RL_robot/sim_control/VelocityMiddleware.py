import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


ANGULAR_SPEED_FACTOR = 1.5


class VelocityMiddleware(Node):
    def __init__(self):
        super().__init__('velocity_middleware')

        # Declare parameters with default values
        self.declare_parameter('max_linear_accel', 1.5)
        self.declare_parameter('max_linear_decel', 0.5)
        self.declare_parameter('max_angular_accel', 3.0)
        self.declare_parameter('max_angular_decel', 1.0)
        self.declare_parameter('rate_hz', 50.0)

        # Get parameter values
        self.max_linear_accel = self.get_parameter('max_linear_accel').value
        self.max_linear_decel = self.get_parameter('max_linear_decel').value
        self.max_angular_accel = self.get_parameter('max_angular_accel').value
        self.max_angular_decel = self.get_parameter('max_angular_decel').value
        self.rate_hz = self.get_parameter('rate_hz').value

        # Calculate dt from rate
        self.dt = 1.0 / self.rate_hz

        # Current velocity state
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        # Target velocity (from commands)
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0

        # Last update time
        self.last_update_time = self.get_clock().now()

        # Publishers and Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',  # Subscribe to this topic (your main controller)
            self.cmd_vel_callback,
            10)

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel_output',  # Publish to this topic (DiffDrive's input)
            10)

        # Timer for publishing commands
        self.timer = self.create_timer(self.dt, self.update_velocity)

        self.get_logger().info("Asymmetric Velocity Controller started")
        self.get_logger().info(f"Linear accel: {self.max_linear_accel}, decel: {self.max_linear_decel}")
        self.get_logger().info(f"Angular accel: {self.max_angular_accel}, decel: {self.max_angular_decel}")

    def cmd_vel_callback(self, msg):
        # Store the requested target velocities
        self.target_linear_vel = msg.linear.x
        self.target_angular_vel = msg.angular.z * ANGULAR_SPEED_FACTOR

    def update_velocity(self):
        """Apply acceleration constraints and publish velocity"""
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now

        # Limit dt to avoid large jumps
        if dt > 0.1:
            dt = 0.1

        # Calculate linear velocity change
        linear_vel_diff = self.target_linear_vel - self.current_linear_vel

        # Determine which acceleration/deceleration to use
        if abs(self.target_linear_vel) < 0.01:
            # Coming to a stop - use deceleration
            max_linear_change = self.max_linear_decel * dt
        elif self.target_linear_vel * self.current_linear_vel < 0:
            # Changing direction - use acceleration
            max_linear_change = self.max_linear_accel * dt
        elif abs(self.target_linear_vel) > abs(self.current_linear_vel):
            # Increasing speed - use acceleration
            max_linear_change = self.max_linear_accel * dt
        else:
            # Decreasing speed but not stopping - use deceleration
            max_linear_change = self.max_linear_decel * dt

        # Apply limits to linear velocity change
        if linear_vel_diff > max_linear_change:
            linear_vel_diff = max_linear_change
        elif linear_vel_diff < -max_linear_change:
            linear_vel_diff = -max_linear_change

        # Update linear velocity
        self.current_linear_vel += linear_vel_diff

        # Repeat for angular velocity
        angular_vel_diff = self.target_angular_vel - self.current_angular_vel

        # Determine which angular acceleration/deceleration to use
        if abs(self.target_angular_vel) < 0.01:
            # Coming to a stop - use deceleration
            max_angular_change = self.max_angular_decel * dt
        elif self.target_angular_vel * self.current_angular_vel < 0:
            # Changing rotation direction - use acceleration
            max_angular_change = self.max_angular_accel * dt
        elif abs(self.target_angular_vel) > abs(self.current_angular_vel):
            # Increasing angular speed - use acceleration
            max_angular_change = self.max_angular_accel * dt
        else:
            # Decreasing angular speed but not stopping - use deceleration
            max_angular_change = self.max_angular_decel * dt

        # Apply limits to angular velocity change
        if angular_vel_diff > max_angular_change:
            angular_vel_diff = max_angular_change
        elif angular_vel_diff < -max_angular_change:
            angular_vel_diff = -max_angular_change

        # Update angular velocity
        self.current_angular_vel += angular_vel_diff

        # Create and publish new velocity command
        cmd = Twist()
        cmd.linear.x = self.current_linear_vel
        cmd.angular.z = self.current_angular_vel

        self.cmd_vel_pub.publish(cmd)
