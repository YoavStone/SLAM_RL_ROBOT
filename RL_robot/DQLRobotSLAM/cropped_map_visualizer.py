import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


class MapVisualizationNode(Node):
    def __init__(self, name='map_visualization_node'):
        super().__init__(name)

        # Add log to confirm initialization
        self.get_logger().info("MapVisualizationNode initializing...")

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
        if cropped_map is None:
            self.get_logger().warn("[MapVis] Received None map data")
            return

        # Store as-is without any reprocessing
        self.map_processed = cropped_map.copy()
        self.resolution = resolution

        # Print debug info
        map_size = int(len(cropped_map) ** 0.5)
        self.get_logger().info(f"[MapVis] Received cropped map: {map_size}x{map_size}, res={resolution}")
        self.get_logger().info(f"[MapVis] First 100 cells: {cropped_map[:100]}")
        self.get_logger().info(f"[MapVis] Last 100 cells: {cropped_map[-100:]}")

        # Count unknown cells in first row
        if map_size > 0:
            first_row = cropped_map[:map_size]
            unknown_count = sum(1 for cell in first_row if cell == -1.0)
            self.get_logger().info(f"[MapVis] First row: {unknown_count}/{map_size} cells are unknown")

        # Publish immediately
        self.publish_map()

    def publish_map(self):
        """Publish the map with diagnostics"""
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

            # Log received map details
            self.get_logger().info(
                f"[MapVis] Processing map with dimensions: {size}x{size}, resolution: {self.resolution}")
            self.get_logger().info(f"[MapVis] First 100 cells: {self.map_processed[:100]}")
            self.get_logger().info(f"[MapVis] Last 100 cells: {self.map_processed[-100:]}")

            # Set origin
            offset = size * self.resolution / 2.0
            msg.info.origin.position.x = -offset
            msg.info.origin.position.y = -offset
            msg.info.origin.orientation.w = 1.0

            # Calculate data statistics for diagnostics
            unknown_count = 0
            free_count = 0
            obstacle_count = 0

            # For debugging, track the first row specifically
            first_row_data = []

            data = []
            for y in range(size):
                row_data = []  # For debugging
                for x in range(size):
                    idx = y * size + x

                    # Make sure we're within bounds
                    if idx < len(self.map_processed):
                        val = self.map_processed[idx]

                        # Convert to occupancy grid values
                        if val == -1.0:
                            grid_val = -1  # Unknown
                            unknown_count += 1
                        elif val < 0.2:
                            grid_val = 0  # Free space
                            free_count += 1
                        else:
                            grid_val = 100  # Obstacle
                            obstacle_count += 1
                    else:
                        grid_val = -1  # Default to unknown if out of bounds
                        unknown_count += 1

                    data.append(grid_val)
                    row_data.append(grid_val)

                    # Save first row for debugging
                    if y == 0:
                        first_row_data.append(grid_val)

                # Debug first 3 and last 3 rows
                if y < 3 or y >= size - 3:
                    self.get_logger().info(f"[MapVis] Row {y} sample: {row_data[:5]}{'...' if size > 5 else ''}")

            msg.data = data

            # Log first row statistics
            unknown_in_first_row = first_row_data.count(-1)
            self.get_logger().info(f"[MapVis] First row: {unknown_in_first_row}/{size} cells are unknown")

            # Log diagnostics
            self.get_logger().info(
                f"[MapVis] Publishing map {size}x{size}, res={self.resolution}: "
                f"unknown={unknown_count}, free={free_count}, obstacle={obstacle_count}"
            )

            # Publish
            self.map_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"[MapVis] Error publishing map: {e}")