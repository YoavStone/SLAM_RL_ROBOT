from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('RL_robot')
    config_file = os.path.join(pkg_dir, 'DQLRobotSLAM', 'dql_params.yaml')

    # Launch arguments
    learning_mode = LaunchConfiguration('learning_mode')
    model_path = LaunchConfiguration('model_path')

    # Declare launch arguments
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

    # Get the path to your virtual environment
    venv_base = os.path.expanduser("~/new_pytorch_env")
    venv_bin = os.path.join(venv_base, "bin")
    venv_lib = os.path.join(venv_base, "lib")
    venv_python_pkgs = os.path.join(venv_lib, "python3.12/site-packages")

    # ROS paths
    ros_base = "/opt/ros/jazzy"
    ros_lib = os.path.join(ros_base, "lib")
    ros_python_pkgs = os.path.join(ros_lib, "python3.12/site-packages")

    # Home directory for ROS logs
    home_dir = os.path.expanduser("~")
    ros_home = os.path.join(home_dir, ".ros")

    # Combine environment variables
    env = {
        'PATH': f"{venv_bin}:{os.environ.get('PATH', '')}",
        'PYTHONPATH': f"{venv_python_pkgs}:{ros_python_pkgs}:{os.environ.get('PYTHONPATH', '')}",
        'LD_LIBRARY_PATH': f"{venv_lib}:{ros_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        'VIRTUAL_ENV': venv_base,
        # ROS-specific environment variables
        'ROS_HOME': ros_home,
        'ROS_LOG_DIR': os.path.join(ros_home, "log"),
        'HOME': home_dir,  # Ensure HOME is properly set
        'USER': os.environ.get('USER', ''),  # Pass through user name
        'ROS_DOMAIN_ID': os.environ.get('ROS_DOMAIN_ID', '0'),
        'RMW_IMPLEMENTATION': os.environ.get('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    }

    agent_node = Node(
        package='RL_robot',
        executable='slam_robot_dqn_agent',
        name='dqn_agent_node',
        output='screen',
        emulate_tty=True,
        env=env,
        parameters=[
            {'learning_mode': learning_mode},
            {'model_path': model_path},
            {'best_model_name': 'gazebo_dql_model.pth'}
        ],
    )

    return LaunchDescription([
        learning_mode_arg,
        model_path_arg,
        agent_node
    ])