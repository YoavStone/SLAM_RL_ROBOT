import time
import rclpy
from std_msgs.msg import Empty


class EnvironmentResetHandler:
    """Handles reset operations for the DQLEnv class, including observation space management"""

    def __init__(self, env):
        """Initialize the handler with a reference to the environment"""
        self.env = env
        self.is_resetting = False
        self._saved_obs_space = None

        # Get the ROS node from the gazebo_env
        self.node = self.env.gazebo_env
        self.logger = self.node.get_logger()

        # Subscribe to reset signals
        self.sim_reset_sub = self.node.create_subscription(
            Empty,
            'simulation_reset',
            self.simulation_reset_callback,
            10
        )

        self.reset_complete_sub = self.node.create_subscription(
            Empty,
            'reset_complete',
            self.reset_complete_callback,
            10
        )

        # Subscribe to episode end signal to detect when resets are requested
        self.episode_end_sub = self.node.create_subscription(
            Empty,
            'episode_end',
            self.episode_end_callback,
            10
        )

        # Flags for synchronization
        self.reset_requested = False
        self.reset_completed = False
        self.reset_timeout = None
        self.max_reset_wait = 60.0  # Maximum seconds to wait for reset

        # Add a timer to check reset status
        self.reset_check_timer = self.node.create_timer(0.5, self.check_reset_status)

        # Flag exposed directly to GazeboEnv
        # This allows the environment to check the reset state directly
        if hasattr(self.env, 'gazebo_env'):
            self.env.gazebo_env.is_resetting = False

        self.logger.info("Environment reset handler initialized")

    def simulation_reset_callback(self, msg):
        """Called when a simulation reset notification is received"""
        self.logger.info("Environment received simulation reset notification")
        self.reset_requested = True
        self.reset_completed = False
        self.reset_timeout = time.time() + self.max_reset_wait

        # Invalidate observation space during reset
        if hasattr(self.env, 'observation_space') and self.env.observation_space is not None:
            self.logger.info("Environment: Setting observation_space to None during reset")
            self._saved_obs_space = self.env.observation_space
            self.env.observation_space = None

        # Set reset flag on both the handler and the underlying environment
        self.is_resetting = True
        if hasattr(self.env, 'gazebo_env'):
            self.env.gazebo_env.is_resetting = True

    def episode_end_callback(self, msg):
        """Called when episode end signal is received"""
        self.logger.info("Environment detected episode end signal - anticipating reset")
        # Mark as resetting even before getting the simulation_reset notification
        self.is_resetting = True
        if hasattr(self.env, 'gazebo_env'):
            self.env.gazebo_env.is_resetting = True

        # Invalidate observation space during reset
        if hasattr(self.env,
                   'observation_space') and self.env.observation_space is not None and self._saved_obs_space is None:
            self.logger.info("Environment: Setting observation_space to None due to episode end")
            self._saved_obs_space = self.env.observation_space
            self.env.observation_space = None

        # Start timeout
        self.reset_timeout = time.time() + self.max_reset_wait
        self.reset_requested = True
        self.reset_completed = False

    def reset_complete_callback(self, msg):
        """Called when reset complete notification is received"""
        self.logger.info("Reset complete notification received")
        self.reset_completed = True

        # IMPORTANT: Immediately restore the observation space
        # This fixes the "environment not ready" issue after reset
        self.restore_observation_space()

    def restore_observation_space(self):
        """Immediately restore the observation space and clear reset flags"""
        if self._saved_obs_space is not None:
            self.logger.info("Environment: Restoring observation_space immediately after reset")
            self.env.observation_space = self._saved_obs_space
            self._saved_obs_space = None

        # Clear reset flags
        self.is_resetting = False
        if hasattr(self.env, 'gazebo_env'):
            self.env.gazebo_env.is_resetting = False

        self.reset_requested = False

    def check_reset_status(self):
        """Periodically check if reset is in progress"""
        if self.reset_requested and not self.reset_completed:
            # Check for timeout
            if self.reset_timeout and time.time() > self.reset_timeout:
                self.logger.warn(f"Reset timeout after {self.max_reset_wait} seconds. Forcing resume.")

                # Clear reset flags and restore observation space
                self.restore_observation_space()
                self.reset_timeout = None
            else:
                # Still waiting for reset
                self.logger.info("Environment: Waiting for reset to complete...", throttle_duration_sec=5.0)

                # Ensure reset flags are set
                self.is_resetting = True
                if hasattr(self.env, 'gazebo_env'):
                    self.env.gazebo_env.is_resetting = True

    def is_reset_in_progress(self):
        """Check if a reset is currently in progress"""
        return self.is_resetting