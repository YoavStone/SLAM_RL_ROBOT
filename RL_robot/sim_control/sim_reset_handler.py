import time
import rclpy
from std_msgs.msg import Empty

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

        # Subscribe to reset complete notifications
        self.reset_complete_sub = agent_node.create_subscription(
            Empty,
            'reset_complete',
            self.reset_complete_callback,
            10
        )

        # flags for synchronization
        self.reset_requested = False
        self.reset_completed = False
        self.reset_timeout = None
        self.max_reset_wait = 60.0  # Maximum seconds to wait for a reset to complete

        # Add a timer to check reset status
        self.reset_check_timer = agent_node.create_timer(0.5, self.check_reset_status)

        self.logger.info("Simulation reset handler initialized")

    def simulation_reset_callback(self, msg):
        """Called when a simulation reset notification is received"""
        self.logger.info("Simulation reset notification received")
        self.reset_requested = True
        self.reset_completed = False
        self.reset_timeout = time.time() + self.max_reset_wait
        self.handle_reset()

    def reset_complete_callback(self, msg):
        """Called when reset complete notification is received"""
        self.logger.info("Reset complete notification received")
        self.reset_completed = True
        self.is_resetting = False

    def check_reset_status(self):
        """Periodically check if reset is in progress"""
        if self.reset_requested and not self.reset_completed:
            # Check for timeout
            if self.reset_timeout and time.time() > self.reset_timeout:
                self.logger.warn(f"Reset timed out after {self.max_reset_wait} seconds. Forcing resume.")
                self.is_resetting = False
                self.reset_requested = False
                self.reset_completed = False
                self.reset_timeout = None
                # Force a reset to get a clean state
                obs, _ = self.env.reset()
                if obs is not None:
                    self.node.current_obs = obs
            else:
                # Still waiting for reset
                self.logger.info("Waiting for reset to complete...", throttle_duration_sec=5.0)
                self.is_resetting = True
        elif self.reset_requested and self.reset_completed:
            self.logger.info("Reset completed, resuming operation")
            self.is_resetting = False
            self.reset_requested = False
            self.reset_completed = False
            self.reset_timeout = None

    def handle_reset(self):
        """Handle environment reset without losing training progress"""
        self.logger.info("Handling environment reset...")

        # Set flag to prevent training during reset
        self.is_resetting = True

        try:
            # Force a longer waiting period to ensure map reset completes
            self.logger.info("Waiting for environment to reset (5 seconds)...")
            time.sleep(5.0)  # Longer wait time to ensure SLAM map reset

            # Reset the environment to get new initial state
            self.logger.info("Getting new observation after reset...")
            current_obs, _ = self.env.reset()

            if current_obs is None:
                self.logger.error("Failed to get observation after environment reset")
                # Try again after another delay
                time.sleep(2.0)
                current_obs, _ = self.env.reset()
                if current_obs is None:
                    self.logger.error("Still failed to get observation after retry")
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

    def is_reset_in_progress(self):
        """Check if a reset is currently in progress"""
        return self.is_resetting