from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    namePackage = 'RL_robot'

    # Only keep model path parameter - other simulation/learning params removed
    model_path = LaunchConfiguration('model_path')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to a saved model to load.'
    )

    tf_broadcaster_node = Node(
        package=namePackage,
        executable='tf_broadcaster',  # This matches the entry_point in setup.py
        name='tf_broadcaster_node',
        output='screen'
    )

    # Create node to run pre-trained model (without simulation components)
    dqn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'slam_dqn_agent_launch.py')
        ]),
        launch_arguments={
            'learning_mode': 'False',
            'model_path': model_path,
            'epsilon_start': '0.0',
            'epsilon_end': '0.0',
            'epsilon_decay': '1',
            'is_sim': 'False'
        }.items()
    )

    # Async toolbox SLAM parameters file
    absPathParamSLAM = os.path.join(get_package_share_directory(namePackage),
                                    'parameters/mapper_params_online_async.yaml')

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')]),
        launch_arguments={'use_sim_time': 'true', 'slam_params_file': absPathParamSLAM}.items()
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory(namePackage), 'rviz', 'physical_robot.rviz')],
    )

    return LaunchDescription([
        tf_broadcaster_node,
        slam_toolbox_launch,
        rviz,
        model_path_arg,
        dqn_launch
    ])