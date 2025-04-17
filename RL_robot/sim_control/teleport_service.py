import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import subprocess
import time
import math


class HarmonicDiffDriveTeleporter(Node):
    """Teleporter specifically for differential drive robots in Gazebo Harmonic"""

    def __init__(self):
        super().__init__('harmonic_teleporter')

        # Robot name
        self.robot_name = 'mapping_robot'
        self.declare_parameter('robot_name', 'mapping_robot')
        self.robot_name = self.get_parameter('robot_name').value

        # Create subscription for teleport requests
        self.teleport_sub = self.create_subscription(
            Pose,
            '/teleport_robot',
            self.teleport_callback,
            10
        )

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Get entity ID for the robot
        self.entity_id = self.get_entity_id()

        self.get_logger().info("Harmonic DiffDrive teleporter initialized")

    def get_entity_id(self):
        """Get the entity ID for the robot in Gazebo Harmonic"""
        try:
            # List all entities in the world
            result = subprocess.run(['gz', 'sim', '-e', '--entity-name', self.robot_name],
                                    capture_output=True, text=True, timeout=2.0)

            if result.returncode == 0 and result.stdout.strip():
                # Parse entity ID from output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if self.robot_name in line:
                        parts = line.split()
                        if len(parts) >= 1:
                            entity_id = parts[0]
                            self.get_logger().info(f"Found robot entity ID: {entity_id}")
                            return entity_id

            # Fallback approach - try model list
            model_result = subprocess.run(['gz', 'model', '-l'],
                                          capture_output=True, text=True, timeout=2.0)

            if model_result.returncode == 0:
                models = model_result.stdout.strip().split('\n')
                for i, model in enumerate(models):
                    if self.robot_name in model:
                        self.get_logger().info(f"Found robot with model ID: {i}")
                        return str(i)

            self.get_logger().warn(f"Could not find entity ID for {self.robot_name}, using default '1'")
            return "1"  # Default entity ID if not found

        except Exception as e:
            self.get_logger().error(f"Error getting entity ID: {e}")
            return "1"  # Default entity ID if not found

    def teleport_callback(self, msg):
        """Handle teleport request with Gazebo Harmonic ECS approach"""
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z if msg.position.z > 0 else 0.05

        # Calculate yaw from quaternion
        qz = msg.orientation.z
        qw = msg.orientation.w

        if abs(qw) < 0.001:
            yaw = math.copysign(math.pi / 2, qz)
        else:
            yaw = 2.0 * math.atan2(qz, qw)

        self.get_logger().info(f"Teleporting robot to x={x}, y={y}, z={z}, yaw={yaw}")

        # Stop the robot first
        stop_cmd = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.05)

        # Try multiple teleportation methods:

        # Method 1: Use the gz entity pose command with entity ID
        try:
            if self.entity_id:
                entity_cmd = ['gz', 'sim', '-e', self.entity_id, '--pose', f'{x},{y},{z},{0},{0},{yaw}']

                self.get_logger().info(f"Running entity pose command: {' '.join(entity_cmd)}")

                entity_result = subprocess.run(entity_cmd, capture_output=True, text=True, timeout=5.0)

                if entity_result.returncode == 0:
                    self.get_logger().info("Entity pose command successful")
                else:
                    self.get_logger().error(f"Entity pose command failed: {entity_result.stderr}")
        except Exception as e:
            self.get_logger().error(f"Error running entity pose command: {e}")

        # Method 2: Try direct link teleportation for all robot links
        try:
            # Get all links for the robot
            links_cmd = ['gz', 'model', '-m', self.robot_name, '--link']

            links_result = subprocess.run(links_cmd, capture_output=True, text=True, timeout=2.0)

            if links_result.returncode == 0 and links_result.stdout.strip():
                links = [line.strip() for line in links_result.stdout.strip().split('\n')
                         if line.strip() and not line.startswith('-')]

                self.get_logger().info(f"Found robot links: {links}")

                # Try to teleport each link
                for link in links:
                    link_cmd = ['gz', 'model', '-m', self.robot_name,
                                '--link', link,
                                '--pose', f'{x} {y} {z} 0 0 {yaw}']

                    self.get_logger().info(f"Teleporting link {link}: {' '.join(link_cmd)}")

                    link_result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=2.0)

                    if link_result.returncode == 0:
                        self.get_logger().info(f"Link {link} teleport successful")
                    else:
                        self.get_logger().warn(f"Link {link} teleport failed: {link_result.stderr}")
            else:
                self.get_logger().warn("Could not get robot links")

        except Exception as e:
            self.get_logger().error(f"Error teleporting links: {e}")

        # Method 3: Try direct world pose command
        try:
            world_cmd = ['gz', 'sim', '-w', '--pose-model',
                         f'{self.robot_name}@{x},{y},{z},{0},{0},{yaw}']

            self.get_logger().info(f"Running world pose command: {' '.join(world_cmd)}")

            world_result = subprocess.run(world_cmd, capture_output=True, text=True, timeout=5.0)

            if world_result.returncode == 0:
                self.get_logger().info("World pose command successful")
            else:
                self.get_logger().error(f"World pose command failed: {world_result.stderr}")

        except Exception as e:
            self.get_logger().error(f"Error running world pose command: {e}")

        # Force the robot to stop again
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.05)

        # Final check
        try:
            # Get current robot pose
            pose_cmd = ['gz', 'model', '-m', self.robot_name, '-p']
            pose_result = subprocess.run(pose_cmd, capture_output=True, text=True, timeout=2.0)

            if pose_result.returncode == 0:
                current_pose = pose_result.stdout.strip()
                self.get_logger().info(f"Final robot pose: {current_pose}")

                # Check if position changed
                if not current_pose or '-0.000' in current_pose.split('\n')[1]:
                    self.get_logger().warn("Robot position unchanged - teleportation failed")

                    # Provide instructions for SDF modification
                    self.get_logger().info("To fix this, try modifying your robot's model SDF to disable dynamics:")
                    self.get_logger().info("1. Find your robot's SDF file")
                    self.get_logger().info("2. For each link, add <static>true</static> or disable gravity")
                    self.get_logger().info("3. Or implement virtual teleportation in your RL algorithm")
                else:
                    self.get_logger().info("Robot position appears to have changed - teleportation successful!")

            else:
                self.get_logger().warn(f"Could not verify final position: {pose_result.stderr}")

        except Exception as e:
            self.get_logger().error(f"Error checking final position: {e}")


def main(args=None):
    rclpy.init(args=args)
    teleporter = HarmonicDiffDriveTeleporter()

    try:
        rclpy.spin(teleporter)
    except KeyboardInterrupt:
        pass
    finally:
        teleporter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()