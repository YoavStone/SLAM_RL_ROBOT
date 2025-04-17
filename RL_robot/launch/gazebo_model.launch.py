import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

from launch_ros.actions import Node
import xacro


def generate_launch_description():
    # Add launch argument for DQN agent
    launch_dqn = LaunchConfiguration('launch_dqn')
    launch_dqn_arg = DeclareLaunchArgument(
        'launch_dqn',
        default_value='false',
        description='Whether to launch the DQN agent (true) or not (false)'
    )

    # DQN-related launch arguments
    learning_mode = LaunchConfiguration('learning_mode')
    model_path = LaunchConfiguration('model_path')

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

    # name in xacro file
    robotXacroName = 'mapping_robot'

    # pkg name to define the paths
    namePackage = 'RL_robot'

    # path to xacro file
    modelFileRelativePath = 'model/robot.xacro'

    # absolute path to model
    pathModelFile = os.path.join(get_package_share_directory(namePackage), modelFileRelativePath)

    # get robot description from xacro model file && combine with .gazebo
    robotDescription = xacro.process_file(pathModelFile).toxml()

    # launch file from gazebo_ros pkg
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'))

    gz_model_path = os.path.join(get_package_share_directory(namePackage), 'models')
    set_gz_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gz_model_path
    )
    # Path to world.sdf file
    worldFileRelativePath = 'worlds/AI_Training_World.sdf'

    # absolute path to world.sdf
    pathWorldFile = os.path.join(get_package_share_directory(namePackage), worldFileRelativePath)

    # Gazebo
    gazeboLaunch = IncludeLaunchDescription(gazebo_rosPackageLaunch, launch_arguments=
    {'gz_args': [f'-r -v -v4 {pathWorldFile}'], 'on_exit_shutdown': 'true'}.items())

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
        arguments=['-d', os.path.join(get_package_share_directory(namePackage), 'rviz', 'gpu_lidar_bridge.rviz')],
    )

    robot_x_location = DeclareLaunchArgument('robot_spawn_x', default_value='0.0')
    robot_y_location = DeclareLaunchArgument('robot_spawn_y', default_value='0.0')

    # Spawn model GZ node
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robotXacroName,
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('robot_spawn_x'),
            '-y', LaunchConfiguration('robot_spawn_y'),
            '-z', '0.01',
        ],
        output='screen',
    )

    # Robot State Publisher node
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robotDescription,
                     'use_sim_time': True}]
    )

    # to control robot from ROS2
    bridge_params = os.path.join(
        get_package_share_directory(namePackage),
        'parameters',
        'bridge_parameters.yaml'
    )

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen',
    )

    sim_vel_control_node = Node(
        package='RL_robot',
        executable='asymmetric_velocity_controller',
        name='asymmetric_velocity_controller',
        output='screen',
        # Optional: Add parameters if you want to configure values from launch file
        parameters=[{
            'max_linear_accel': 1.5,
            'max_linear_decel': 0.5,
            'max_angular_accel': 3.0,
            'max_angular_decel': 1.0,
            'rate_hz': 50.0,
        }],
    )

    # Include the DQN agent launch file conditionally
    dqn_agent_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(namePackage), 'launch', 'slam_dqn_agent_launch.py')
        ]),
        condition=IfCondition(launch_dqn),
        launch_arguments={
            'learning_mode': learning_mode,
            'model_path': model_path
        }.items()
    )

    # empty launch description object
    launchDescriptionObject = LaunchDescription()

    # Add the launch arguments
    launchDescriptionObject.add_action(launch_dqn_arg)
    launchDescriptionObject.add_action(robot_x_location)
    launchDescriptionObject.add_action(robot_y_location)
    launchDescriptionObject.add_action(learning_mode_arg)
    launchDescriptionObject.add_action(model_path_arg)

    # add gazeboLaunch
    launchDescriptionObject.add_action(sim_vel_control_node)
    launchDescriptionObject.add_action(set_gz_model_path)
    launchDescriptionObject.add_action(gazeboLaunch)

    # add the nodes
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)

    launchDescriptionObject.add_action(rviz)

    launchDescriptionObject.add_action(slam_toolbox_launch)

    # Add the DQN agent launch file (conditional)
    launchDescriptionObject.add_action(dqn_agent_launch)

    return launchDescriptionObject