import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv
import subprocess
import os
import time
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
import threading


class ResetHandler(Node):
    """Node that handles external resets of SLAM and robot position"""

    def __init__(self):
        super().__init__('reset_handler')

        # Path to SLAM parameters file
        self.package_name = 'RL_robot'
        self.slam_params_path = os.path.join(
            get_package_share_directory(self.package_name),
            'parameters/mapper_params_online_async.yaml'
        )

        # Create a subscriber for reset requests
        self.reset_sub = self.create_subscription(
            Empty,
            '/environment_reset_request',
            self.reset_callback,
            10
        )

        # Publisher to send clear_map service request using topic
        self.clear_map_pub = self.create_publisher(Empty, '/slam_toolbox/clear_map', 10)

        # Create clients for the SLAM services
        self.clear_map_client = self.create_client(EmptySrv, '/slam_toolbox/clear_map')
        self.pause_mapping_client = self.create_client(EmptySrv, '/slam_toolbox/pause_mapping')
        self.resume_mapping_client = self.create_client(EmptySrv, '/slam_toolbox/resume_mapping')

        # Publisher for empty map
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 10)

        # Track latest map data
        self.latest_map = None
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        # Limit how often we can restart SLAM
        self.last_restart_time = 0
        self.min_restart_interval = 5.0  # seconds

        # Flag to track if reset is in progress
        self.reset_in_progress = False
        self.reset_lock = threading.Lock()

        self.get_logger().info("Reset handler node initialized, listening for reset requests")
        self.get_logger().info(f"Using SLAM parameters from: {self.slam_params_path}")

    def map_callback(self, msg):
        """Store latest map for potential reset"""
        self.latest_map = msg

    def call_slam_service(self, client, service_name):
        """Call a SLAM toolbox service safely"""
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"{service_name} service not available")
            return False

        try:
            request = EmptySrv.Request()
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is not None:
                self.get_logger().info(f"{service_name} service call succeeded")
                return True
            else:
                self.get_logger().warn(f"{service_name} service call failed")
                return False
        except Exception as e:
            self.get_logger().error(f"Error calling {service_name} service: {e}")
            return False

    def reset_callback(self, msg):
        """Handle environment reset request with multiple approaches"""
        # Prevent multiple resets from running simultaneously
        with self.reset_lock:
            if self.reset_in_progress:
                self.get_logger().info("Reset already in progress, skipping")
                return
            self.reset_in_progress = True

        try:
            self.get_logger().info("Received reset request, attempting to reset SLAM")

            # Approach 1: Try using ros2 service call directly
            try:
                self.get_logger().info("Attempting direct service call via command line")
                subprocess.run(['ros2', 'service', 'call', '/slam_toolbox/clear_map', 'std_srvs/srv/Empty'],
                               timeout=2.0)
                time.sleep(0.5)
                self.get_logger().info("Direct service call completed")
            except Exception as e:
                self.get_logger().warn(f"Direct service call failed: {e}")

            # Approach 2: Try using service clients
            self.call_slam_service(self.pause_mapping_client, "pause_mapping")
            time.sleep(0.2)
            success = self.call_slam_service(self.clear_map_client, "clear_map")
            time.sleep(0.2)
            self.call_slam_service(self.resume_mapping_client, "resume_mapping")

            if success:
                self.get_logger().info("Successfully cleared map using services")
            else:
                # Approach 3: Try using the topic instead
                for _ in range(10):
                    self.clear_map_pub.publish(Empty())
                    time.sleep(0.1)
                self.get_logger().info("Published clear_map messages via topic")

            # Approach 4: Try publishing an empty map if we have a template
            if self.latest_map is not None:
                try:
                    empty_map = OccupancyGrid()
                    empty_map.header = self.latest_map.header
                    empty_map.header.stamp = self.get_clock().now().to_msg()
                    empty_map.info = self.latest_map.info

                    # Fill with -1 (unknown)
                    size = self.latest_map.info.width * self.latest_map.info.height
                    empty_map.data = [-1] * size

                    # Publish empty map multiple times
                    for _ in range(5):
                        self.map_pub.publish(empty_map)
                        time.sleep(0.1)
                    self.get_logger().info("Published empty map")
                except Exception as e:
                    self.get_logger().error(f"Error publishing empty map: {e}")

            # Approach 5: If enough time has passed, try restarting SLAM
            current_time = time.time()
            if current_time - self.last_restart_time >= self.min_restart_interval:
                self.last_restart_time = current_time
                self.restart_slam_toolbox()
            else:
                self.get_logger().info(f"Skipping SLAM restart - too soon (interval: {self.min_restart_interval}s)")

        finally:
            # Reset the in-progress flag
            with self.reset_lock:
                self.reset_in_progress = False

    def restart_slam_toolbox(self):
        """Restart the SLAM toolbox using ROS2 CLI commands"""
        self.get_logger().info("Attempting to restart SLAM toolbox")

        try:
            # Try to kill any existing slam_toolbox nodes
            subprocess.run(['ros2', 'node', 'kill', '/slam_toolbox'], timeout=2.0)
            time.sleep(1.0)

            # Launch a new slam_toolbox instance
            launch_cmd = [
                'ros2', 'launch', 'slam_toolbox', 'online_async_launch.py',
                f'slam_params_file:={self.slam_params_path}',
                'use_sim_time:=true'
            ]

            # Run in background
            subprocess.Popen(
                launch_cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            self.get_logger().info("SLAM toolbox restart initiated")
        except Exception as e:
            self.get_logger().error(f"Error restarting SLAM toolbox: {e}")


def main(args=None):
    rclpy.init(args=args)

    reset_handler = ResetHandler()

    try:
        rclpy.spin(reset_handler)
    except KeyboardInterrupt:
        pass
    finally:
        reset_handler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()