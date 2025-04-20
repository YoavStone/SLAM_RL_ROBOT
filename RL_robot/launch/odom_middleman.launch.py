from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Launch arguments
    spawn_location = LaunchConfiguration('spawn_location')
    spawn_location_arg = DeclareLaunchArgument(
        'spawn_location',
        default_value='',
        description='Initial spawn location for the robot in x,y format'
    )

    # Odometry middleman node
    odom_middleman_node = Node(
        package='RL_robot',
        executable='odom_middleman',
        name='odom_middleman',
        output='screen',
        parameters=[
            {'publish_rate': 30.0}
        ]
    )

    # Episode monitor node
    episode_monitor_node = Node(
        package='RL_robot',
        executable='episode_monitor',
        name='episode_monitor',
        output='screen',
        parameters=[
            {'spawn_location': spawn_location}
        ]
    )

    # Create and return the launch description
    return LaunchDescription([
        spawn_location_arg,
        odom_middleman_node,
        episode_monitor_node
    ])