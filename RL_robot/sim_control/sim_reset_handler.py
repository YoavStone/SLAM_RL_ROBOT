import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
import time


class SimulationResetHandler:
    """Handles simulation reset functionality for the DQLAgent"""

    def __init__(self, agent_node, env):
        """
        Initialize the reset handler.

        Args:
            agent_node: The ROS node (DQLAgent) this handler is associated with
            env: The environment object that needs to be reset
        """
        self.node = agent_node
        self.env = env
        self.logger = agent_node.get_logger()
        self.is_resetting = False

        # Subscribe to simulation reset notifications
        self.sim_reset_sub = agent_node.create_subscription(
            Empty,
            'simulation_reset',
            self.simulation_reset_callback,
            10
        )

        # Service to handle reset requests
        self.reset_service = agent_node.create_service(
            Trigger,
            'dqn_agent_reset',
            self.reset_service_callback
        )

        self.logger.info("Simulation reset handler initialized")

    def simulation_reset_callback(self, msg):
        """Called when a simulation reset notification is received"""
        self.logger.info("Simulation reset notification received")
        self.handle_reset()

    def reset_service_callback(self, request, response):
        """Service callback to handle reset requests"""
        self.logger.info("Reset service called")
        success = self.handle_reset()
        response.success = success
        response.message = "Reset handled successfully" if success else "Reset failed"
        return response

    def handle_reset(self):
        """Handle environment reset without losing training progress"""
        self.logger.info("Handling environment reset...")

        # Set flag to prevent training during reset
        self.is_resetting = True

        try:
            # Reset the environment to get new initial state
            current_obs, _ = self.env.reset()
            if current_obs is None:
                self.logger.error("Failed to get observation after environment reset")
                return False

            # Update agent's current observation
            self.node.current_obs = current_obs

            # Reset episode-specific variables
            self.node.episode_reward = 0.0

            self.logger.info("Environment reset handled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error handling environment reset: {e}")
            return False
        finally:
            # Clear reset flag
            self.is_resetting = False

    def is_reset_in_progress(self):
        """Check if a reset is currently in progress"""
        return self.is_resetting