#############################################################
#     ROS2 && Robot Launch File of the diff drive robot     #
#############################################################

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # pkg name to define the paths
    namePackage = 'RL_robot'

    # path to xacro file
    modelFileRelativePath = 'model/robot.xacro'

    # absolute path to model
    pathModelFile = os.path.join(get_package_share_directory(namePackage), modelFileRelativePath)

    # get robot description from xacro model file && combine with .gazebo
    robotDescription = xacro.process_file(pathModelFile).toxml()

    # Async toolbox SLAM parameters file
    absPathParamSLAM = os.path.join(get_package_share_directory(namePackage),
                                    'robot_controller/mapper_params_online_async.yaml')

    # Robot State Publisher node
    nodeRobotStatePublisher = Node(
        package = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output = 'screen',
        parameters = [{'robot_description': robotDescription,
        'use_sim_time': True}]
    )

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


    # empty launch description object
    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(nodeRobotStatePublisher)

    launchDescriptionObject.add_action(rviz)

    launchDescriptionObject.add_action(slam_toolbox_launch)

    return launchDescriptionObject