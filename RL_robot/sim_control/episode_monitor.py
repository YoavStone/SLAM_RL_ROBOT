import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import subprocess
import os
import signal


class EpisodeMonitor(Node):
    def __init__(self):
        super().__init__('episode_monitor')
        self.subscription = self.create_subscription(
            Empty,
            'episode_end',
            self.episode_callback,
            10
        )
        self.pkg = 'RL_robot'
        self.process = None
        self.launch_file = 'gazebo_model.launch.py'
        self.launch_args = ['launch_dqn:=true', 'learning_mode:=true']
        self.launch_system()

    def launch_system(self):
        self.get_logger().info("🔄 Launching system...")
        self.process = subprocess.Popen(
            ['ros2', 'launch', self.pkg, self.launch_file] + self.launch_args,
            preexec_fn=os.setsid
        )

    def episode_callback(self, msg):
        self.get_logger().info("📩 Episode ended — restarting system.")
        self.restart_system()

    def restart_system(self):
        if self.process:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
        self.launch_system()


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.process:
            os.killpg(os.getpgid(node.process.pid), signal.SIGTERM)
        node.destroy_node()
        rclpy.shutdown()
