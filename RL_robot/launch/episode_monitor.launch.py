# episode_monitor.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Declare launch arguments for optional parameters
    # Default value is an empty string, matching the node's expectation for 'use default'
    spawn_location_arg = DeclareLaunchArgument(
        'spawn_location',
        default_value='',
        description="Optional spawn location for the robot in 'x,y' format. Empty means random."
    )

    nn_path_arg = DeclareLaunchArgument(
        'nn_path',
        default_value='',
        description="Optional path to the neural network file. Empty means none specified."
    )

    # You can also make launch_dqn and learning_mode launch arguments if desired
    # launch_dqn_arg = DeclareLaunchArgument('launch_dqn', default_value='true')
    # learning_mode_arg = DeclareLaunchArgument('learning_mode', default_value='true')

    return LaunchDescription([
        spawn_location_arg,
        nn_path_arg,
        # launch_dqn_arg,  # Uncomment if you make these launch args
        # learning_mode_arg, # Uncomment if you make these launch args

        Node(
            package='RL_robot', # Replace with your actual package name
            executable='episode_monitor', # Make sure to include .py if it's a script
            name='episode_monitor',
            output='screen',
            parameters=[
                # Pass parameters to the Python node.
                # These names must match exactly what's used in declare_parameter() in the script.
                {'launch_dqn': True}, # Or use LaunchConfiguration('launch_dqn') if made into a launch arg
                {'learning_mode': True}, # Or use LaunchConfiguration('learning_mode') if made into a launch arg
                {'spawn_location': LaunchConfiguration('spawn_location')},
                {'nn_path': LaunchConfiguration('nn_path')}
            ]
        )
    ])