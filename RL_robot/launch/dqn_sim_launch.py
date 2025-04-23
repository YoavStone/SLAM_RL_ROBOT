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
    spawn_location = LaunchConfiguration('spawn_location')
    robot_spawn_x = LaunchConfiguration('robot_spawn_x')
    robot_spawn_y = LaunchConfiguration('robot_spawn_y')
    # dql parameters
    learning_mode = LaunchConfiguration('learning_mode')
    model_path = LaunchConfiguration('model_path')
    epsilon_start = LaunchConfiguration('epsilon_start')
    epsilon_end = LaunchConfiguration('epsilon_end')
    epsilon_decay = LaunchConfiguration('epsilon_decay')

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

    # Epsilon launch arguments with defaults
    epsilon_start_arg = DeclareLaunchArgument(
        'epsilon_start',
        default_value='1.0',
        description='Initial exploration rate (epsilon) for the agent'
    )

    epsilon_end_arg = DeclareLaunchArgument(
        'epsilon_end',
        default_value='0.02',
        description='Final exploration rate (epsilon) for the agent'
    )

    epsilon_decay_arg = DeclareLaunchArgument(
        'epsilon_decay',
        default_value='45000',
        description='Number of steps over which epsilon decays from start to end value'
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

    # Launch the odometry middleman and episode monitor
    odom_middleman_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'odom_middleman.launch.py')
        ]),
        launch_arguments={
            'spawn_location': spawn_location
        }.items()
    )

    # Launch the DQN agent with epsilon parameters
    dqn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'slam_dqn_agent_launch.py')
        ]),
        launch_arguments={
            'learning_mode': learning_mode,
            'model_path': model_path,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay
        }.items()
    )

    return LaunchDescription([
        epsilon_start_arg,
        epsilon_end_arg,
        epsilon_decay_arg,
        learning_mode_arg,
        model_path_arg,
        spawn_location_arg,
        robot_spawn_x_arg,
        robot_spawn_y_arg,
        gazebo_launch,      # Launch Gazebo first
        odom_middleman_launch,  # Then launch the odom middleman and episode monitor
        dqn_launch          # Finally launch the DQN agent
    ])