#############################################################
#     ROS2 && Gazebo Launch File of the diff drive robot    #
#############################################################

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro


def generate_launch_description():

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
    


    # Path to world.sdf file
    worldFileRelativePath = 'worlds/world_try.sdf'

    # absolute path to world.sdf
    pathWorldFile = os.path.join(get_package_share_directory(namePackage), worldFileRelativePath)

    # Gazebo
    gazeboLaunch = IncludeLaunchDescription(gazebo_rosPackageLaunch, launch_arguments=
        {'gz_args': [f'-r -v -v4 {pathWorldFile}'], 'on_exit_shutdown': 'true'}.items())



    # Async toolbox SLAM parameters file
    absPathParamSLAM = os.path.join(get_package_share_directory(namePackage), 'robot_controller/mapper_params_online_async.yaml')

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')]),
        launch_arguments={'use_sim_time': 'true', 'slam_params_file': absPathParamSLAM}.items()
    )
    

    # RViz
    rviz = Node(
       package='rviz2',
       executable='rviz2',
       arguments=['-d', os.path.join(get_package_share_directory(namePackage), 'rviz', 'gpu_lidar_bridge.rviz')],
    )


    # Spawn model GZ node
    spawnModelNodeGazebo = Node(
        package = 'ros_gz_sim',
        executable = 'create',
        arguments = [
            '-name', robotXacroName,
            '-topic', 'robot_description'
        ],
        output = 'screen',
    )

    # Robot State Publisher node
    nodeRobotStatePublisher = Node(
        package = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output = 'screen',
        parameters = [{'robot_description': robotDescription,
        'use_sim_time': True}]
    )

    # to control robot from ROS2
    bridge_params = os.path.join(
        get_package_share_directory(namePackage),
        'parameters',
        'bridge_parameters.yaml'
    )

    start_gazebo_ros_bridge_cmd = Node(
        package = 'ros_gz_bridge',
        executable = 'parameter_bridge',
        arguments = [
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output = 'screen',
    )

    # empty launch description object
    launchDescriptionObject = LaunchDescription()

    # add gazeboLaunch
    launchDescriptionObject.add_action(gazeboLaunch)

    # add the nodes
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)

    launchDescriptionObject.add_action(rviz)

    launchDescriptionObject.add_action(slam_toolbox_launch)

    return launchDescriptionObject