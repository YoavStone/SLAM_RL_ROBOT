# combined_launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os


def generate_launch_description():

    namePackage = 'RL_robot'

    # launch arguments
    learning_mode = LaunchConfiguration('learning_mode')
    model_path = LaunchConfiguration('model_path')
    spawn_location = LaunchConfiguration('spawn_location')

    learning_mode_arg = DeclareLaunchArgument(
        'learning_mode',
        default_value='true',
        description='Whether the agent is in learning mode (true) or execution mode (false)'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to a saved model to load. If empty, starts with a fresh model.'
    )

    spawn_location_arg = DeclareLaunchArgument(
        'spawn_location',
        default_value='6.3,0',
        description='Initial spawn location for the robot in x,y format'
    )

    # Launch the DQN agent separately (this will keep running)
    dqn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'slam_dqn_agent_launch.py')
        ]),
        launch_arguments={
            'learning_mode': learning_mode,
            'model_path': model_path
        }.items()
    )

    # Launch the episode monitor which will manage the simulation restarts
    episode_monitor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'episode_monitor.launch.py')
        ]),
        launch_arguments={
            'spawn_location': spawn_location,
        }.items()
    )

    return LaunchDescription([
        learning_mode_arg,
        model_path_arg,
        spawn_location_arg,
        episode_monitor_launch,
        dqn_launch,
    ])