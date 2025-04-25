import rclpy
from rclpy.node import Node
from std_msgs.msg import String # Import the String message type
from geometry_msgs.msg import Twist, Pose, Quaternion # Still needed for target pose representation
import math
import time
import subprocess # Required for the subprocess call


class TeleportServiceNode(Node):
    """
    A ROS 2 node that subscribes to a teleport command topic using a String message
    and uses a subprocess call to teleport the robot in Gazebo.
    NOTE: Using subprocess.communicate() for teleporting is a blocking operation
    and can make this node temporarily unresponsive.
    """

    def __init__(self):
        super().__init__('teleport_service_node')
        self.get_logger().info("Teleport Service Node initializing...")

        # Robot model name in Gazebo - ensure this matches your robot
        self.model_name = 'mapping_robot' # This will be used in the subprocess command

        # Create a subscriber to the teleport command topic using String message
        # Topic name: /teleport_command
        # Message type: std_msgs/msg/String
        self.teleport_subscriber = self.create_subscription(
            String,
            '/teleport_command',
            self.teleport_command_callback,
            10
        )
        self.get_logger().info("Subscribed to /teleport_command topic using String message")

        self.get_logger().info("Teleport Service Node initialized.")


    def teleport_command_callback(self, msg):
        """
        Callback function for receiving teleport command messages as a String.
        Uses a subprocess call to execute the gz service command.
        Expected message format: "x,y,yaw"
        """
        teleport_data_str = msg.data # Get the string data from the message
        self.get_logger().info(f"Received teleport command string: {teleport_data_str}")

        try:
            # Parse the string data (assuming "x,y,yaw" format)
            x_str, y_str, yaw_str = teleport_data_str.split(',')
            target_x = float(x_str.strip())
            target_y = float(y_str.strip())
            target_yaw = float(yaw_str.strip())

        except ValueError:
            self.get_logger().error(f"Failed to parse teleport command string: '{teleport_data_str}'. Expected format 'x,y,yaw'.")
            return # Exit the callback if parsing fails
        except Exception as e:
            self.get_logger().error(f"An error occurred while parsing teleport command string: {e}")
            return # Exit the callback on other parsing errors


        self.get_logger().info(f"Parsed target pose: x={target_x}, y={target_y}, yaw={target_yaw}")

        # Convert yaw to quaternion (assuming rotation around Z axis)
        qz = math.sin(target_yaw / 2.0)
        qw = math.cos(target_yaw / 2.0)

        # Format the command for Gazebo Harmonic
        # Assumes the world name is 'empty'
        cmd = [
            'gz', 'service', '-s', '/world/empty/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', # Timeout for the gz service call itself (milliseconds)
            '--req',
            f'name: "{self.model_name}", position: {{x: {target_x}, y: {target_y}, z: 0.0}}, orientation: {{w: {qw}, x: 0.0, y: 0.0, z: {qz}}}'
        ]

        self.get_logger().info(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            # --- Blocking call ---
            # This line will pause the current thread until the subprocess finishes
            # or the timeout (2.5 seconds) is reached. This can block this node's spin.
            stdout, stderr = process.communicate(timeout=2.5) # Timeout for communicate (seconds)
            output = stdout.decode() if stdout else ""
            error = stderr.decode() if stderr else ""

            if error:
                self.get_logger().warn(f"Teleport command error: {error}")
            if output:
                 self.get_logger().info(f"Teleport command output: {output.strip()}")

            # Check if the command returned success (based on the gz service output format)
            # This is a basic check, the actual output might vary
            if "data: true" in output:
                 self.get_logger().info("Robot teleport successful via subprocess.")
                 # You might want to publish a "teleport_complete" message here
                 # for other nodes (like the agent) to know the teleport is done.
            else:
                 self.get_logger().warn("Robot teleport failed via subprocess (output did not indicate success).")


        except subprocess.TimeoutExpired:
            process.kill()
            self.get_logger().warn("Teleport command timed out.")
        except FileNotFoundError:
             self.get_logger().error("Error: 'gz' command not found. Is Gazebo Sim installed and in your PATH?")
        except Exception as e:
            self.get_logger().error(f"Error executing teleport command: {e}")

        # Add a small delay after attempting teleportation for physics to settle
        time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = TeleportServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("TeleportServiceNode shutdown complete")


if __name__ == '__main__':
    main()
