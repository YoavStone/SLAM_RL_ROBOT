import os
import pickle
import random
from collections import deque
import numpy as np
import time
from std_msgs.msg import Int32, Bool


class ToggleDemonstrationBuffer:
    """
    Buffer for storing and using human demonstrations for improved DQN learning.
    Uses ROS topics for control instead of direct keyboard input.
    """

    def __init__(self, max_demos=50000, demo_batch_ratio=0.3, auto_timeout=300, save_path="src/RL_robot/saved_networks/saved_demonstrations/demo_buffer.pkl"):
        self.max_demos = max_demos
        self.demo_batch_ratio = demo_batch_ratio
        self.auto_timeout = auto_timeout
        self.save_path = save_path

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

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

        # Pending action
        self.pending_action = None

        # For logging
        self.logger = None
        self.current_state = None

        self.load_demonstrations()

    def load_demonstrations(self, custom_path=None):
        """Load demonstration buffer from file if it exists

        Args:
            custom_path: Optional path to load from. If None, uses default save_path.

        Returns:
            bool: True if demonstrations were successfully loaded, False otherwise
        """
        path_to_use = custom_path if custom_path else self.save_path

        try:
            if os.path.exists(path_to_use):
                with open(path_to_use, 'rb') as f:
                    loaded_buffer = pickle.load(f)

                    # Convert to a deque with maxlen if it's not already
                    if isinstance(loaded_buffer, deque) and hasattr(loaded_buffer,
                                                                    'maxlen') and loaded_buffer.maxlen == self.max_demos:
                        self.demo_buffer = loaded_buffer
                    else:
                        # In case the loaded buffer is not a deque or has different maxlen
                        self.demo_buffer = deque(loaded_buffer, maxlen=self.max_demos)

                    self.log(f"Loaded {len(self.demo_buffer)} demonstrations from {path_to_use}")
                    return True
            else:
                self.log(f"No demonstration file found at {path_to_use}. Starting with empty buffer.")
                return False
        except Exception as e:
            self.log(f"Failed to load demonstrations: {e}. Starting with empty buffer.")
            return False

    def save_demonstrations(self, custom_path=None):
        """Save current demonstration buffer to file

        Args:
            custom_path: Optional path to save to. If None, uses default save_path.

        Returns:
            bool: True if demonstrations were successfully saved, False otherwise
        """
        if len(self.demo_buffer) == 0:
            self.log("No demonstrations to save.")
            return False

        path_to_use = custom_path if custom_path else self.save_path

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path_to_use), exist_ok=True)

            with open(path_to_use, 'wb') as f:
                pickle.dump(self.demo_buffer, f)

            self.log(f"Saved {len(self.demo_buffer)} demonstrations to {path_to_use}")
            return True
        except Exception as e:
            self.log(f"Failed to save demonstrations: {e}")
            return False

    def toggle_callback(self, msg):
        """Handle toggle messages from keyboard node"""
        if msg.data:
            if self.is_recording:
                self.log("Stopping demonstration recording")
                self.stop_recording()
                # Save demonstrations when stopping recording
                self.save_demonstrations()
            else:
                self.log("Starting demonstration recording")
                self.start_recording()

    def check_for_toggle(self):
        """Check if timeout has been reached"""
        # Check for timeout if recording
        if self.is_recording and self.demo_start_time is not None:
            current_time = time.time()
            if current_time - self.demo_start_time > self.auto_timeout:
                self.log(f"Demonstration mode auto-timeout after {self.auto_timeout} seconds")
                self.stop_recording()
                # Save demonstrations when timeout occurs
                self.save_demonstrations()  # Added this line

        return self.is_recording

    def action_callback(self, msg):
        """Handle action messages from keyboard node"""
        action = msg.data
        if action >= 0 and action <= 4 and self.is_recording:
            # action_names = ["Stop", "Forward", "Backward", "Turn right", "Turn left"]
            # self.log(f'Demo action: {action_names[action]}')
            self.pending_action = action

    def set_logger(self, logger):
        """Set a logger for messages"""
        self.logger = logger

    def log(self, message):
        """Log a message using the provided logger or print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def get_action(self):
        """Get the current action (or default to stop)"""
        if self.pending_action is None:
            return 0
        return self.pending_action

    def set_current_state(self, state):
        """Set the current state"""
        self.current_state = state

    def start_recording(self):
        """Start recording demonstration"""
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
        self.log("Use WASD keys to control from the keyboard node")

        # Reset pending action
        self.pending_action = None

    def add_transition(self, state, action, reward, next_state, done):
        """Add a transition to the current demonstration"""
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

        # # Log current reward
        # self.log(f"Action: {action}, Reward: {reward}, Total: {self.episode_reward:.2f}")

        # If episode ended, reset the stats
        if done:
            self.episode_reward = 0.0

    def stop_recording(self):
        """Stop recording and save demonstration to buffer"""
        if not self.is_recording or len(self.current_demo_states) == 0:
            return

        self.is_recording = False

        # Add all transitions to the demonstration buffer
        for i in range(len(self.current_demo_states)):
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
        """Get a mixed batch from replay buffer and demonstrations"""
        if len(self.demo_buffer) == 0 or len(replay_buffer) < batch_size // 2:
            return None

        # Calculate adaptive ratio based on relative buffer sizes
        total_experiences = len(self.demo_buffer) + len(replay_buffer)
        adaptive_ratio = min(
            self.demo_batch_ratio,  # Cap at the configured max ratio
            max(0.05, len(self.demo_buffer) / total_experiences)  # Ensure at least 5%
        )

        # Calculate how many samples to take from each buffer
        demo_batch_size = min(int(batch_size * adaptive_ratio), len(self.demo_buffer))
        replay_batch_size = batch_size - demo_batch_size

        # Ensure we have enough samples from each buffer
        if len(replay_buffer) < replay_batch_size:
            replay_batch_size = len(replay_buffer)
            demo_batch_size = min(batch_size - replay_batch_size, len(self.demo_buffer))

        # Log the current ratio (occasionally)
        if random.random() < 0.01:  # Log approximately once every 100 calls
            self.log(f"Demo ratio: {adaptive_ratio:.2f}, Demo: {demo_batch_size}, Replay: {replay_batch_size}")

        # Sample from both buffers
        replay_transitions = random.sample(replay_buffer, replay_batch_size)
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