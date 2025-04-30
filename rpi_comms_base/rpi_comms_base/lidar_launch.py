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
            'scan_mode': 'Standard',
            'angle_min': 0.0,            # Start angle in radians (0 degrees)
            'angle_max': 6.28318531,     # End angle in radians (360 degrees or 2π)
            'angle_offset': 3.14159265,  # Add 180 degrees (π radians) rotation offset
            'scan_frequency': 5.5,      # Optional: Set scan frequency in Hz
            'range_min': 0.15,           # Optional: Minimum detection range (meters)
            'range_max': 12.0            # Optional: Maximum detection range (meters)
        }]
    )

    return LaunchDescription([
        serial_port_arg,
        rplidar_node
    ])