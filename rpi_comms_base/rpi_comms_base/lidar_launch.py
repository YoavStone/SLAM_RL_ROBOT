import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',  # Provide a default
        description='Path to the serial port for the RPLIDAR'
    )

    serial_port = LaunchConfiguration('serial_port')

    rplidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        output='screen',
        parameters=[{
            'serial_port': serial_port,
            'frame_id': 'base_footprint',
            'angle_compensate': True,
            'scan_mode': 'Standard'
        }]
    )

    return LaunchDescription([
        serial_port_arg,
        rplidar_node
    ])