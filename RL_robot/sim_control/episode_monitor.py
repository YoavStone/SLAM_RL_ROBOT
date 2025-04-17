# episode_monitor.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import subprocess
import os
import signal
import random
import sys # Import sys to access command line args if needed, though params are better

class EpisodeMonitor(Node):
    def __init__(self):
        super().__init__('episode_monitor')

        # Declare parameters with default values
        self.declare_parameter('launch_dqn', True)
        self.declare_parameter('learning_mode', True)
        self.declare_parameter('spawn_location', '') # Default: empty string means random
        self.declare_parameter('nn_path', '')       # Default: empty string means no specific NN path

        # Get parameter values
        self.launch_dqn = self.get_parameter('launch_dqn').get_parameter_value().bool_value
        self.learning_mode = self.get_parameter('learning_mode').get_parameter_value().bool_value
        self.spawn_location_str = self.get_parameter('spawn_location').get_parameter_value().string_value
        self.nn_path = self.get_parameter('nn_path').get_parameter_value().string_value

        self.get_logger().info(f"Received parameters: launch_dqn={self.launch_dqn}, learning_mode={self.learning_mode}, "
                             f"spawn_location='{self.spawn_location_str}', nn_path='{self.nn_path}'")

        self.subscription = self.create_subscription(
            Empty,
            'episode_end',
            self.episode_callback,
            10
        )
        self.pkg = 'RL_robot' # Replace with your actual package name if different
        self.process = None
        self.launch_file = 'gazebo_model.launch.py' # The launch file for Gazebo and the model

        # Predefined possible random positions (only used if spawn_location param is empty)
        self.positions = [
            (0.0, 0.0),
            (6.3, 0.0),
            (-6.3, 0.0),
            (0.0, 6.3),
            (0.0, -6.3)
        ]

        # Initial launch
        self.launch_system()

    def get_random_pose_args(self):
        """Generates random pose arguments from the predefined list."""
        x, y = random.choice(self.positions)
        self.get_logger().info(f"Using random spawn location: x={x}, y={y}")
        return [f'robot_spawn_x:={x}', f'robot_spawn_y:={y}']

    def parse_spawn_location(self):
        """Parses the spawn_location parameter string 'x,y'."""
        try:
            parts = self.spawn_location_str.split(',')
            if len(parts) == 2:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                self.get_logger().info(f"Using specified spawn location: x={x}, y={y}")
                return [f'robot_spawn_x:={x}', f'robot_spawn_y:={y}']
            else:
                self.get_logger().warn(f"Invalid format for spawn_location parameter: '{self.spawn_location_str}'. Expected 'x,y'. Falling back to random.")
                return None
        except ValueError:
            self.get_logger().warn(f"Could not parse spawn_location parameter: '{self.spawn_location_str}' into floats. Falling back to random.")
            return None
        except Exception as e:
             self.get_logger().error(f"Error parsing spawn_location parameter: {e}. Falling back to random.")
             return None


    def launch_system(self):
        """Launches the gazebo_model.launch.py with appropriate arguments."""
        if self.process:
            self.get_logger().warn("Launch called while a process might still be running. Ensure proper cleanup.")
            # Optional: add forceful termination here if needed, though restart_system handles it
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            self.process = None

        # --- Determine Pose Arguments ---
        pose_args = []
        if self.spawn_location_str: # If a specific location is provided
            parsed_args = self.parse_spawn_location()
            if parsed_args:
                pose_args = parsed_args
            else: # Fallback to random if parsing failed
                pose_args = self.get_random_pose_args()
        else: # No specific location provided, use random
            pose_args = self.get_random_pose_args()

        # --- Determine Base and Optional Arguments ---
        base_args = [
            f'launch_dqn:={self.launch_dqn}',
            f'learning_mode:={self.learning_mode}'
        ]

        optional_args = []
        if self.nn_path: # If a neural network path is provided
             # IMPORTANT: gazebo_model.launch.py must be modified to accept and use 'nn_path'
            optional_args.append(f'nn_path:={self.nn_path}')
            self.get_logger().info(f"Passing nn_path: {self.nn_path}")
        else:
             self.get_logger().info("No specific nn_path provided.")


        # --- Combine Arguments ---
        full_args = base_args + pose_args + optional_args

        self.get_logger().info(f"🔄 Launching '{self.launch_file}' with args: {full_args}")

        # --- Launch Subprocess ---
        try:
            cmd = ['ros2', 'launch', self.pkg, self.launch_file] + full_args
            self.process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid # Use process group to kill all child processes
            )
            self.get_logger().info(f"🚀 Launched process with PID: {self.process.pid}")
        except FileNotFoundError:
             self.get_logger().error(f"Error: 'ros2' command not found. Is ROS 2 sourced?")
             rclpy.shutdown()
             sys.exit(1)
        except Exception as e:
            self.get_logger().error(f"Failed to launch subprocess: {e}")
            rclpy.shutdown()
            sys.exit(1)


    def episode_callback(self, msg):
        self.get_logger().info("📩 Episode ended signal received — restarting system.")
        self.restart_system()

    def restart_system(self):
        """Terminates the current simulation process and launches a new one."""
        if self.process and self.process.poll() is None: # Check if process exists and is running
            self.get_logger().info(f"Terminating process group {os.getpgid(self.process.pid)}...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM) # Send SIGTERM to the whole process group
                self.process.wait(timeout=5) # Wait for graceful termination
                self.get_logger().info("Process terminated.")
            except ProcessLookupError:
                 self.get_logger().warn("Process group already terminated.")
            except subprocess.TimeoutExpired:
                 self.get_logger().warn("Process did not terminate gracefully after 5s. Sending SIGKILL.")
                 try:
                     os.killpg(os.getpgid(self.process.pid), signal.SIGKILL) # Force kill
                     self.process.wait(timeout=2)
                 except Exception as e:
                     self.get_logger().error(f"Error during SIGKILL: {e}")
            except Exception as e:
                self.get_logger().error(f"Error terminating process: {e}")
            finally:
                self.process = None # Ensure process handle is cleared
        elif self.process:
             self.get_logger().info("Process was already terminated.")
             self.process = None # Clear handle if process finished but handle wasn't cleared
        else:
            self.get_logger().info("No process was running.")

        # Launch a new instance
        self.launch_system()

    def shutdown_hook(self):
        """Cleanly shut down the subprocess when the node is terminated."""
        self.get_logger().info("Shutting down EpisodeMonitor node.")
        if self.process and self.process.poll() is None:
            self.get_logger().info(f"Terminating subprocess group {os.getpgid(self.process.pid)} during shutdown.")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception as e:
                self.get_logger().warn(f"Error during shutdown termination: {e}. Trying SIGKILL.")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=2)
                except Exception as kill_e:
                     self.get_logger().error(f"Error during shutdown SIGKILL: {kill_e}")
            finally:
                self.process = None


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down.')
    finally:
        # Ensure subprocess is killed before destroying the node
        node.shutdown_hook()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("EpisodeMonitor shutdown complete.")

if __name__ == '__main__':
    main()