from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
import os


def generate_launch_description():
    # Get the path to your virtual environment's Python interpreter
    venv_path = os.path.expanduser("~/new_pytorch_env/bin")

    # Set environment variables to use your virtual environment
    env = {
        'PATH': venv_path + ':' + os.environ.get('PATH', ''),
        'PYTHONPATH': os.path.expanduser("~/new_pytorch_env/lib/python3.12/site-packages") + ':' + os.environ.get(
            'PYTHONPATH', '')
    }

    return LaunchDescription([
        Node(
            package='RL_robot',
            executable='dqn_agent',
            name='dqn_agent_node',
            output='screen',
            emulate_tty=True,
            env=env,
        )
    ])