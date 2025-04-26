import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


class MapVisualizationNode(Node):
    def __init__(self, name='map_visualization_node', publish=False):
        super().__init__(name)

        # Add log to confirm initialization
        self.get_logger().info("MapVisualizationNode initializing...")

        self.publish = publish

        # QoS setup and publisher creation as before
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            'cropped_map',
            qos_profile=map_qos
        )

        # Map data
        self.map_processed = None
        self.resolution = 0.15

        # Log timer creation
        self.get_logger().info("Visualization timer created")

    def set_map(self, cropped_map, resolution):
        """Store the map data exactly as provided from the agent"""
        if self.publish:
            if cropped_map is None:
                return

            # Store as-is without any reprocessing
            self.map_processed = cropped_map.copy()
            self.resolution = resolution

            # Print debug info
            map_size = int(len(cropped_map) ** 0.5)
            # self.get_logger().info(f"Stored cropped map: {map_size}x{map_size}, res={resolution}")

            # Publish immediately
            self.publish_map()

    def publish_map(self):
        """Publish the map with diagnostics"""
        if self.publish:
            if self.map_processed is None:
                self.get_logger().warn("No map data to publish")
                return

            try:
                # Create message
                msg = OccupancyGrid()

                # Set header
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'

                # Set metadata
                size = int(len(self.map_processed) ** 0.5)

                # Check if size calculation is valid
                if size * size != len(self.map_processed):
                    self.get_logger().error(f"Map data length {len(self.map_processed)} is not a perfect square")
                    return

                msg.info.width = size
                msg.info.height = size
                msg.info.resolution = self.resolution

                # Set origin
                offset = size * self.resolution / 2.0
                msg.info.origin.position.x = -offset
                msg.info.origin.position.y = -offset
                msg.info.origin.orientation.w = 1.0

                # Calculate data statistics for diagnostics
                unknown_count = 0
                free_count = 0
                obstacle_count = 0

                data = []
                for y in range(size):  # Going row by row
                    for x in range(size):  # Then column by column
                        idx = y * size + x

                        # Make sure we're within bounds
                        if idx < len(self.map_processed):
                            val = self.map_processed[idx]

                            if val == -1.0:
                                data.append(-1)  # Unknown
                            elif val < 0.2:
                                data.append(0)  # Free space
                            else:
                                data.append(100)  # Obstacle
                        else:
                            data.append(-1)  # Default to unknown if out of bounds

                msg.data = data

                # Log diagnostics
                # self.get_logger().info(
                #     f"Publishing map {size}x{size}, res={self.resolution}: "
                #     f"unknown={unknown_count}, free={free_count}, obstacle={obstacle_count}"
                # )

                # Publish
                self.map_pub.publish(msg)

            except Exception as e:
                self.get_logger().error(f"Error publishing map: {e}")