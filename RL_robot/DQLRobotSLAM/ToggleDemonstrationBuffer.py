import random
from collections import deque
import numpy as np
import sys
import termios
import tty
import select
import time


class ToggleDemonstrationBuffer:
    """
    Buffer for storing and using human demonstrations for improved DQN learning.
    This class allows toggling between AI control and human demonstration recording.
    Features:
    - Toggle demonstration mode with 'p' key
    - Control robot with WASD keys during demonstration mode
    - Auto-timeout for demonstration mode
    - Adaptive ratio of demonstrations based on buffer sizes
    """

    def __init__(self, max_demos=50000, demo_batch_ratio=0.3, auto_timeout=300):  # 300 seconds = 5 minutes
        """
        Initialize the demonstration buffer.

        Args:
            max_demos: Maximum number of demonstration transitions to store
            demo_batch_ratio: Maximum ratio of demonstration samples in training batches
            auto_timeout: Automatically exit demo mode after this many seconds
        """
        self.max_demos = max_demos
        self.demo_batch_ratio = demo_batch_ratio
        self.auto_timeout = auto_timeout

        # Buffer to store demonstration transitions
        self.demo_buffer = deque(maxlen=max_demos)

        # Recording state
        self.is_recording = False
        self.current_demo_states = []
        self.current_demo_actions = []
        self.current_demo_rewards = []
        self.current_demo_dones = []
        self.current_demo_next_states = []
        self.episode_reward = 0.0

        # Timer for auto-timeout
        self.demo_start_time = None

        # Terminal settings for keyboard input
        self.old_settings = None
        self.pending_action = None

        # For logging
        self.logger = None

        # Current state for toggling
        self.current_state = None

    def set_logger(self, logger):
        """Set a logger for messages"""
        self.logger = logger

    def log(self, message):
        """Log a message using the provided logger or print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def check_for_toggle(self):
        """
        Check if the 'p' key is pressed to toggle demonstration mode or timeout is reached.
        Returns True if in demo mode, False otherwise.
        """
        key = self.get_key()

        # Check for timeout if recording
        if self.is_recording and self.demo_start_time is not None:
            current_time = time.time()
            if current_time - self.demo_start_time > self.auto_timeout:
                self.log(f"Demonstration mode auto-timeout after {self.auto_timeout} seconds")
                self.stop_recording()
                return False

        # Check for toggle key
        if key == 'p':
            if self.is_recording:
                self.stop_recording()
                return False  # Stop recording
            else:
                self.start_recording()
                return True  # Start recording

        return self.is_recording  # Return current recording state

    def get_key(self):
        """Get a single keypress from the terminal (non-blocking)"""
        # Save terminal settings if not already saved
        if not hasattr(self, 'old_settings') or self.old_settings is None:
            # Only try to get terminal settings if stdin is a tty
            if sys.stdin.isatty():
                try:
                    self.old_settings = termios.tcgetattr(sys.stdin)
                except Exception as e:
                    self.log(f"Warning: Failed to get terminal settings: {e}")
                    return None
            else:
                return None

        try:
            # Set terminal to raw mode
            if sys.stdin.isatty():
                try:
                    tty.setraw(sys.stdin.fileno())
                except Exception as e:
                    self.log(f"Warning: Failed to set raw terminal mode: {e}")
                    return None

                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key = sys.stdin.read(1)
                else:
                    key = None
            else:
                key = None

        finally:
            # Restore terminal settings
            if self.old_settings and sys.stdin.isatty():
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except Exception as e:
                    self.log(f"Warning: Failed to restore terminal settings: {e}")

        return key

    def get_action(self):
        """
        Get action from keyboard input for the current state.
        If no new input, use the previous action.
        """
        # Process keyboard input
        key = self.get_key()

        # Map keys to actions
        action = None

        if key == 'w':
            action = 1  # Forward
            self.log('Moving forward')
        elif key == 's':
            action = 2  # Backward
            self.log('Moving backward')
        elif key == 'd':
            action = 3  # Turn right
            self.log('Turning right')
        elif key == 'a':
            action = 4  # Turn left
            self.log('Turning left')
        elif key == 'x':
            action = 0  # Stop
            self.log('Stopping')
        elif key == 'p':
            # 'p' is handled by check_for_toggle
            pass

        # If new action provided, store it
        if action is not None:
            self.pending_action = action
            return action

        # If no new action, use previous action or default to stop
        if self.pending_action is not None:
            return self.pending_action
        else:
            return 0  # Default to stop

    def start_recording(self):
        """Start recording a new demonstration"""
        self.is_recording = True
        self.current_demo_states = []
        self.current_demo_actions = []
        self.current_demo_rewards = []
        self.current_demo_dones = []
        self.current_demo_next_states = []
        self.episode_reward = 0.0

        # Start the auto-timeout timer
        self.demo_start_time = time.time()

        self.log(f"Started demonstration recording. Auto-timeout in {self.auto_timeout} seconds.")
        self.log("Use WASD to control, X to stop, P to toggle back to AI control.")
        self.log("Current reward: 0.0")

        # Reset pending action
        self.pending_action = None

    def set_current_state(self, state):
        """Set the current state - called by the agent during normal operation"""
        self.current_state = state

    def add_transition(self, state, action, reward, next_state, done):
        """Add a single transition to the demonstration buffer"""
        if not self.is_recording:
            return

        # Store the transition
        self.current_demo_states.append(state)
        self.current_demo_actions.append(action)
        self.current_demo_rewards.append(reward)
        self.current_demo_dones.append(done)
        self.current_demo_next_states.append(next_state)

        # Update episode reward
        self.episode_reward += reward

        # Log current reward
        self.log(f"Action: {action}, Reward: {reward}, Total: {self.episode_reward:.2f}")

        # If episode ended, reset the stats
        if done:
            self.log(f"Episode complete with reward: {self.episode_reward:.2f}")
            self.episode_reward = 0.0

    def stop_recording(self):
        """Stop recording and add the demonstration to the buffer"""
        if not self.is_recording or len(self.current_demo_states) == 0:
            return

        self.is_recording = False

        # Add all transitions to the demonstration buffer
        for i in range(len(self.current_demo_states)):
            # Add to buffer
            self.demo_buffer.append((
                self.current_demo_states[i],
                self.current_demo_actions[i],
                self.current_demo_rewards[i],
                self.current_demo_dones[i],
                self.current_demo_next_states[i]
            ))

        self.log(f"Demonstration recording complete. Added {len(self.current_demo_states)} transitions.")
        self.log(f"Total demonstrations in buffer: {len(self.demo_buffer)}")

    def get_mixed_batch(self, replay_buffer, batch_size):
        """
        Get a mixed batch of transitions with adaptive ratio based on relative buffer sizes.

        Args:
            replay_buffer: The regular replay buffer to sample from
            batch_size: Total batch size

        Returns:
            Tuple of (obs_batch, act_batch, rew_batch, done_batch, next_obs_batch)
        """
        if len(self.demo_buffer) == 0 or len(replay_buffer) < batch_size // 2:
            # If no demonstrations or not enough replay data, use regular sampling
            return None

        # Calculate adaptive ratio based on relative buffer sizes
        # As replay buffer grows, we'll use fewer demonstrations
        total_experiences = len(self.demo_buffer) + len(replay_buffer)
        adaptive_ratio = min(
            self.demo_batch_ratio,  # Cap at the configured max ratio
            max(0.05, len(self.demo_buffer) / total_experiences)  # Ensure at least 5%
        )

        # Calculate how many samples to take from demonstrations vs replay buffer
        demo_batch_size = min(int(batch_size * adaptive_ratio), len(self.demo_buffer))
        replay_batch_size = batch_size - demo_batch_size

        # Ensure we have enough samples from each buffer
        if len(replay_buffer) < replay_batch_size:
            replay_batch_size = len(replay_buffer)
            demo_batch_size = min(batch_size - replay_batch_size, len(self.demo_buffer))

        # Log the current ratio (occasionally)
        if random.random() < 0.01:  # Log approximately once every 100 calls
            if self.logger:
                self.logger.info(f"Adaptive demo ratio: {adaptive_ratio:.2f}, "
                                 f"Demo batch: {demo_batch_size}, Replay batch: {replay_batch_size}")

        # Sample from replay buffer
        replay_transitions = random.sample(replay_buffer, replay_batch_size)

        # Sample from demonstration buffer
        demo_transitions = random.sample(self.demo_buffer, demo_batch_size)

        # Combine transitions
        combined_transitions = replay_transitions + demo_transitions

        # Convert to numpy arrays
        obs_batch = np.array([t[0] for t in combined_transitions], dtype=np.float32)
        act_batch = np.array([t[1] for t in combined_transitions], dtype=np.int64)
        rew_batch = np.array([t[2] for t in combined_transitions], dtype=np.float32)
        done_batch = np.array([t[3] for t in combined_transitions], dtype=np.float32)
        next_obs_batch = np.array([t[4] for t in combined_transitions], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, done_batch, next_obs_batch

    def close(self):
        """Clean up resources"""
        # Restore terminal settings
        if hasattr(self, 'old_settings') and self.old_settings:
            if sys.stdin.isatty():
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except Exception as e:
                    self.log(f"Warning: Failed to restore terminal settings when closing: {e}")