import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import subprocess
import os
import signal
import random


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
        self.positions = [
            (0.0, 0.0),
            (6.3, 0.0),
            (-6.3, 0.0),
            (0.0, 6.3),
            (0.0, -6.3)
        ]
        self.launch_system()

    def get_random_pose_args(self):
        x, y = random.choice(self.positions)
        return [f'robot_spawn_x:={x}', f'robot_spawn_y:={y}']

    def launch_system(self):
        pose_args = self.get_random_pose_args()
        full_args = self.get_launch_args() + pose_args
        print(f"🔄 Launching system with pose args: {pose_args}")
        print("launch args: ", full_args)
        self.process = subprocess.Popen(
            ['ros2', 'launch', self.pkg, self.launch_file] + full_args,
            preexec_fn=os.setsid
        )

    def get_launch_args(self):
        return self.launch_args

    def episode_callback(self, msg):
        print("📩 Episode ended — restarting system.")
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
