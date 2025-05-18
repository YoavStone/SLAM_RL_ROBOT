import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
import numpy as np
import time
import math

from visualizers.MapVisualizationNode import MapVisualizationNode


class DataFilter:
    """
    this class exists to filter the map, lidar, odom, vel, orientation data so that the Network gets the correct data
    """
    def __init__(self, vis_cropped_map=False):

        # visualize the cropped map usually for debug
        print("Creating visualization node...")
        self.vis_node = MapVisualizationNode(publish=vis_cropped_map)

        # data to filter
        self.map_processed = []  # Processed map data for DQL input
        self.velocities = None  # [vx, va]
        self.slam_pose = None  # Store the latest SLAM pose [orientation, x, y]
        self.grid_position = None  # stores position on grid [sin(x), cos(x), x, y]
        self.measured_distance_to_walls = [10.0] * 16  # distances in sixteenths of circle

        self.map_raw = None
        self.center_cell_x = None
        self.center_cell_y = None

        # Data ready flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.slam_pose_ready = False
        self.should_update_center = True


    def reset(self):
        # Reset flags
        self.scan_ready = False
        self.odom_ready = False
        self.map_ready = False
        self.slam_pose_ready = False
        self.should_update_center = True

        self.map_processed = None
        self.slam_pose = None
        self.grid_position = None
        self.velocities = None

        self.map_raw = None
        self.center_cell_x = None
        self.center_cell_y = None
