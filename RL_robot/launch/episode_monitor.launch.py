from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments for optional parameters
    spawn_location_arg = DeclareLaunchArgument(
        'spawn_location',
        default_value='',
        description="Optional spawn location for the robot in 'x,y' format. Empty means random."
    )

    return LaunchDescription([
        spawn_location_arg,

        Node(
            package='RL_robot',
            executable='episode_monitor',
            name='episode_monitor',
            output='screen',
            parameters=[
                {'spawn_location': LaunchConfiguration('spawn_location')}
            ]
        )
    ])