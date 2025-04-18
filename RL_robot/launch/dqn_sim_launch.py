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

    # Launch arguments
    learning_mode = LaunchConfiguration('learning_mode')
    model_path = LaunchConfiguration('model_path')
    spawn_location = LaunchConfiguration('spawn_location')
    robot_spawn_x = LaunchConfiguration('robot_spawn_x')
    robot_spawn_y = LaunchConfiguration('robot_spawn_y')

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
        default_value='',
        description='Initial spawn location for the robot in x,y format'
    )

    # Create separate arguments for x and y
    robot_spawn_x_arg = DeclareLaunchArgument(
        'robot_spawn_x',
        default_value='0.0',
        description='Robot spawn X position'
    )

    robot_spawn_y_arg = DeclareLaunchArgument(
        'robot_spawn_y',
        default_value='0.0',
        description='Robot spawn Y position'
    )

    # Launch Gazebo and the robot
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'gazebo_model.launch.py')
        ]),
        launch_arguments={
            'robot_spawn_x': robot_spawn_x,
            'robot_spawn_y': robot_spawn_y
        }.items()
    )

    # Launch the DQN agent
    dqn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'slam_dqn_agent_launch.py')
        ]),
        launch_arguments={
            'learning_mode': learning_mode,
            'model_path': model_path
        }.items()
    )

    # Launch the episode monitor
    episode_monitor_node = Node(
        package='RL_robot',
        executable='episode_monitor',
        name='episode_monitor',
        output='screen',
        parameters=[
            {'spawn_location': spawn_location}
        ]
    )

    return LaunchDescription([
        learning_mode_arg,
        model_path_arg,
        spawn_location_arg,
        robot_spawn_x_arg,
        robot_spawn_y_arg,
        gazebo_launch,  # Launch Gazebo first
        dqn_launch,  # Then launch the DQN agent
        episode_monitor_node  # Finally launch the episode monitor
    ])