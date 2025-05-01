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
            'frame_id': 'lidar_link',
            'angle_compensate': True,
            'scan_mode': 'Standard',
            'angle_min': 0.0,            # Start angle in radians (0 degrees)
            'angle_max': 6.28318531,     # End angle in radians (360 degrees or 2π)
            'scan_frequency': 5.5,      # Optional: Set scan frequency in Hz
            'range_min': 0.15,           # Optional: Minimum detection range (meters)
            'range_max': 12.0            # Optional: Maximum detection range (meters)
        }]
    )

    base_to_body_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_body_transform',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'body_link']
    )

    # relative to body link (with 180 degree rotation)
    body_to_lidar_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_lidar_transform',
        arguments=['-0.12', '0', '0.312', '3.1415926', '0', '0', 'body_link', 'lidar_link']
    )

    tf_broadcaster = Node(
        package='rpi_comms_base',
        executable='TFBroadCasterRobot',
        name='tf_broadcaster_node',
        output='screen',
    )

    return LaunchDescription([
        serial_port_arg,
        base_to_body_transform,
        body_to_lidar_transform,
        tf_broadcaster,
        rplidar_node
    ])