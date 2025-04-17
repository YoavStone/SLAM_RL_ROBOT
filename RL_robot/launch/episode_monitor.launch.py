from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='RL_robot',
            executable='episode_monitor',
            name='episode_monitor',
            output='screen',
            parameters=[
                {'launch_dqn': True},
                {'learning_mode': True}
            ]
        )
    ])
