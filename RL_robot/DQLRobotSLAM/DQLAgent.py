import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from .DQNetwork import DQNetwork
from .DuelingDQN import DuelingDQN
from .DQLEnv import DQLEnv

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE_START = 2.5e-4
LEARNING_RATE_END = 1e-4
LEARNING_RATE_DECAY = 150000
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000  # Minimum experiences in buffer before learning starts
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 150000  # Steps over which epsilon decays
TARGET_UPDATE_FREQ = 1000  # Steps between updating the target network

SAVE_VIDEO_STEP_COUNT_THRESHOLD = 100


class DQLAgent(Node):
    def __init__(self, learning_mode=True, model_path='', best_model_name="best_dqn_gazebo_model"):
        super().__init__('dql_agent')

        # --- Parameters ---
        self.declare_parameter('learning_mode', learning_mode)
        self.declare_parameter('model_path', model_path)  # Path to load a pre-trained model (optional)
        self.declare_parameter('best_model_name', best_model_name)  # Filename for the best performing model
        # Add new epsilon parameters
        self.declare_parameter('epsilon_start', EPSILON_START)  # Starting epsilon value
        self.declare_parameter('epsilon_end', EPSILON_END)  # Final epsilon value
        self.declare_parameter('epsilon_decay', EPSILON_DECAY)  # Steps for decay
        # Add spawn_location parameter for reset handler
        self.declare_parameter('spawn_location', '')  # For reset handler

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.best_model_name = self.get_parameter('best_model_name').value
        # Get epsilon parameters with defaults
        self.epsilon_start = self.get_parameter('epsilon_start').value
        self.epsilon_end = self.get_parameter('epsilon_end').value
        self.epsilon_decay = self.get_parameter('epsilon_decay').value
        # Get spawn location parameter for reset handler
        self.spawn_location = self.get_parameter('spawn_location').value

        self.get_logger().info(f"--- DQL Agent Configuration ---")
        self.get_logger().info(f"Learning Mode: {self.learning_mode}")
        self.get_logger().info(f"Load Model Path: '{self.model_path}'")
        self.get_logger().info(f"Best Model Filename: '{self.best_model_name}'")
        self.get_logger().info(f"Epsilon Start: {self.epsilon_start}")
        self.get_logger().info(f"Epsilon End: {self.epsilon_end}")
        self.get_logger().info(f"Epsilon Decay Steps: {self.epsilon_decay}")
        self.get_logger().info(f"Spawn Location: '{self.spawn_location}'")
        self.get_logger().info(f"-------------------------------")

        # --- Environment Setup ---
        self.env = DQLEnv()

        # Wait for observation space initialization
        while self.env.observation_space is None:
            if self.env.update_observation_space():
                self.get_logger().info("Observation space initialized successfully!")
                break
            else:
                # Allow ROS callbacks in the environment node to process
                rclpy.spin_once(self.env.gazebo_env, timeout_sec=0.1)
                self.get_logger().info("Waiting for observation space to be initialized...", throttle_duration_sec=2.0)
                time.sleep(0.5)

        self.get_logger().info(f"Observation space shape: {self.env.observation_space.shape}")
        self.get_logger().info(f"Action space size: {self.env.action_space.n}")

        # # --- Network Initialization --- for when using standard dqn
        # self.q_network = DQNetwork(self.env.observation_space.shape, self.env.action_space.n)
        # self.target_net = DQNetwork(self.env.observation_space.shape, self.env.action_space.n)

        # --- Network Initialization ---
        self.q_network = DuelingDQN(self.env.observation_space.shape, self.env.action_space.n)
        self.target_net = DuelingDQN(self.env.observation_space.shape, self.env.action_space.n)

        # --- Load Pre-trained Model (if specified) ---
        load_path = self.model_path if self.model_path else self.best_model_name  # Try loading best if no specific path given
        # Construct full path relative to package
        potential_best_path = os.path.join("src/RL_robot/saved_networks/network_params/", load_path)

        effective_load_path = None
        if self.model_path and os.path.exists(self.model_path):
            effective_load_path = self.model_path  # Use specific path if it exists
        elif os.path.exists(potential_best_path) and not self.model_path:
            effective_load_path = potential_best_path  # Use default best path if it exists and no specific path given

        if effective_load_path:
            try:
                self.q_network.load_state_dict(torch.load(effective_load_path))
                self.target_net.load_state_dict(self.q_network.state_dict())  # Sync target net
                self.get_logger().info(f"✅ Successfully loaded model from {effective_load_path}")
            except Exception as e:
                self.get_logger().error(f"!!**!! Failed to load model from {effective_load_path}: {e}. Starting fresh.")
                # Ensure target net is synced with the randomly initialized q_network
                self.target_net.load_state_dict(self.q_network.state_dict())
        else:
            self.target_net.load_state_dict(self.q_network.state_dict())  # Sync target net
            if self.model_path:
                self.get_logger().warn(f"!!**!! Specified model path '{self.model_path}' not found.")
            elif load_path == self.best_model_name:
                self.get_logger().info(f"!!**!! Default best model '{potential_best_path}' not found. Starting fresh.")
            else:
                self.get_logger().info("No model path specified or found. Starting fresh.")

        # Initialize optimizer with starting learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE_START)
        # Create Exponential decay (smooth)
        decay_rate = np.log(LEARNING_RATE_END/LEARNING_RATE_START) / LEARNING_RATE_DECAY
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=np.exp(decay_rate)
        )

        # Log the learning rate configuration
        self.get_logger().info(f"Learning rate scheduler configured:")
        self.get_logger().info(f"  Initial rate: {LEARNING_RATE_START}")
        self.get_logger().info(f"  Final rate: {LEARNING_RATE_END}")
        self.get_logger().info(f"  Decay steps: {LEARNING_RATE_DECAY}")

        # --- Replay Buffer & Reward Tracking ---
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=1000)  # For logging avg reward
        self.episode_reward = 0.0

        # --- Saving Logic ---
        self.best_episode_reward = -float('inf')  # Track the best single episode reward
        self.best_model_dir = os.path.join("src/RL_robot/saved_networks/network_params/")
        self.episode_model_dir = os.path.join("src/RL_robot/saved_networks/episode_network_params/")
        self.best_model_path = os.path.join(self.best_model_dir, self.best_model_name)

        # Create directories if they don't exist
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.episode_model_dir, exist_ok=True)
        self.get_logger().info(f"Models will be saved to:")
        self.get_logger().info(f"  Best: {self.best_model_dir}")
        self.get_logger().info(f"  Episodes: {self.episode_model_dir}")

        # --- State for Training/Execution Loop ---
        self.timer = self.create_timer(0.01, self.timer_callback)  # Timer for the main loop
        self.training_initialized = False
        self.current_obs = None
        self.steps = 0
        self.episode_count = 0  # Start counting episodes from 0

        self.get_logger().info(f"DQL Agent initialized successfully in {'LEARNING' if self.learning_mode else 'EXECUTION'} mode.")
        if self.learning_mode and len(self.replay_buffer) < MIN_REPLAY_SIZE:
            self.get_logger().info("Initializing replay buffer...")
            # Start buffer initialization immediately if in learning mode
            self.initialize_replay_buffer()

    def initialize_replay_buffer(self):
        """Fills the replay buffer with initial random experiences."""
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to get initial observation from environment!")
                return False  # Indicate failure

        init_steps = 0
        while len(self.replay_buffer) < MIN_REPLAY_SIZE:
            epsilon = np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

            if random.random() < epsilon:
                action = self.env.action_space.sample()  # Explore
            else:
                action = self.q_network.act(self.current_obs)  # Exploit

            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated

            if new_obs is None:
                self.get_logger().warn("Received None observation during buffer initialization. Skipping step.")
                # Attempt reset if needed
                if is_done:
                    obs_reset, _ = self.env.reset()
                    if obs_reset is None:
                        self.get_logger().error("Failed to reset environment during buffer init!")
                        return False
                    self.current_obs = obs_reset
                continue  # Skip adding this transition

            # Ensure observations are numpy arrays before adding
            current_obs_np = np.array(self.current_obs, dtype=np.float32)
            new_obs_np = np.array(new_obs, dtype=np.float32)

            self.replay_buffer.append((current_obs_np, action, reward, is_done, new_obs_np))
            init_steps += 1

            if is_done:
                self.current_obs, _ = self.env.reset()
                if self.current_obs is None:
                    self.get_logger().error("Failed to reset environment during buffer init!")
                    return False
            else:
                self.current_obs = new_obs

            if init_steps % 20 == 0:
                self.get_logger().info(f"Replay buffer filling: {len(self.replay_buffer)}/{MIN_REPLAY_SIZE}")

        self.get_logger().info("++--++ Replay buffer initialized.")
        self.training_initialized = True
        return True

    def timer_callback(self):
        """Main loop called by the ROS timer."""
        if self.learning_mode:
            if not self.training_initialized:
                # Try to initialize buffer if not done yet
                if not self.initialize_replay_buffer():
                    self.get_logger().warn("Replay buffer initialization pending or failed, skipping training step.")
                    return  # Wait for next timer call
            # Proceed with training step only if buffer is initialized
            self.train_step()
        else:
            self.execute_step()

    def train_step(self):
        """Executes one step of interaction and learning."""
        # If current_obs is None, we need to reset or initialize
        if self.current_obs is None:
            self.get_logger().info("Initializing/resetting observation for training")
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to get observation after reset in train_step")
                return
            self.episode_reward = 0.0

        # --- Action Selection (Epsilon-Greedy) ---
        epsilon = np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        if random.random() < epsilon:
            action = self.env.action_space.sample()  # Explore
        else:
            action = self.q_network.act(self.current_obs)  # Exploit

        # --- Environment Step ---
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        # Check if we received a valid observation
        if new_obs is None:
            self.get_logger().warn("Received None observation during training step. Attempting reset.")
            self.current_obs, _ = self.env.reset()
            return  # Skip this step

        # --- Store Experience ---
        # Ensure observations are numpy arrays
        current_obs_np = np.array(self.current_obs, dtype=np.float32)
        new_obs_np = np.array(new_obs, dtype=np.float32)
        self.replay_buffer.append((current_obs_np, action, reward, is_done, new_obs_np))

        self.current_obs = new_obs
        self.episode_reward += reward

        # --- Episode End Handling ---
        if is_done:
            self.handle_episode_end()
        else:
            # Learning step
            self.learn_step()

            # Update Target Network
            if self.steps % TARGET_UPDATE_FREQ == 0 and self.steps > 0:
                self.target_net.load_state_dict(self.q_network.state_dict())
                self.get_logger().info(f"**--** Updated target network at step {self.steps}")

        self.steps += 1

    def execute_step(self):
        """Executes the learned policy without exploration or learning."""
        # Initialize observation if needed
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to get initial observation in execution mode!")
                return
            self.episode_reward = 0.0

        # Choose action greedily (no exploration)
        action = self.q_network.act(self.current_obs)

        # Execute action
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        # Handle invalid observation
        if new_obs is None:
            self.get_logger().warn("Received None observation during execution step.")
            self.current_obs, _ = self.env.reset()
            return

        self.current_obs = new_obs
        self.episode_reward += reward

        # Handle episode end
        if is_done:
            self.get_logger().info(f"Execution episode complete. Final reward: {self.episode_reward:.2f}")
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0.0

    def handle_episode_end(self):
        """Handle the end of a training episode"""
        self.episode_count += 1
        self.reward_buffer.append(self.episode_reward)  # Add final reward to buffer for avg calculation
        mean_reward_100 = np.mean(self.reward_buffer)

        # log episode details
        current_lr = self.optimizer.param_groups[0]['lr']
        self.get_logger().info(
            f"--- Episode {self.episode_count} Finished --- \n"
            f"Reward: {self.episode_reward:.2f} | Avg Reward (Last 100): {mean_reward_100:.2f} \n"
            f"Steps: {self.steps} | Epsilon: {np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end]):.3f} \n"
            f"Current learning rate: {current_lr:.6f} at step {self.steps}"
            f"-------------------------------------"
        )

        # Save models
        self.save_models()

        # Reset for next episode
        self.current_obs, _ = self.env.reset()
        if self.current_obs is None:
            self.get_logger().error("Failed to reset environment after episode end!")
            return
        self.episode_reward = 0.0  # Reset episode reward

    def learn_step(self):
        """Performs a gradient descent step based on a batch of experiences."""
        # Only learn if buffer has enough samples and we are in learning mode
        if len(self.replay_buffer) < BATCH_SIZE or not self.learning_mode:
            return

        # Sample batch
        transitions = random.sample(self.replay_buffer, BATCH_SIZE)
        # Unpack and convert transitions using list comprehensions for clarity
        obs_batch = np.array([t[0] for t in transitions], dtype=np.float32)
        act_batch = np.array([t[1] for t in transitions], dtype=np.int64)
        rew_batch = np.array([t[2] for t in transitions], dtype=np.float32)
        done_batch = np.array([t[3] for t in transitions], dtype=np.float32)  # Use float for multiplication later
        next_obs_batch = np.array([t[4] for t in transitions], dtype=np.float32)

        obs_t = torch.as_tensor(obs_batch)
        acts_t = torch.as_tensor(act_batch).unsqueeze(-1)  # Shape: [batch_size, 1]
        rews_t = torch.as_tensor(rew_batch).unsqueeze(-1)  # Shape: [batch_size, 1]
        dones_t = torch.as_tensor(done_batch).unsqueeze(-1)  # Shape: [batch_size, 1]
        next_obs_t = torch.as_tensor(next_obs_batch)

        # --- Compute Target Q Values ---
        with torch.no_grad():  # No gradients needed for target computation
            # Get Q values for next states from target network
            target_q_values = self.target_net(next_obs_t)
            # Select best action Q value according to target network: max(Q_target(s', a'))
            max_target_q = target_q_values.max(dim=1, keepdim=True)[0]
            # Calculate TD target: r + gamma * max(Q_target(s', a')) * (1 - done)
            targets = rews_t + GAMMA * max_target_q * (1.0 - dones_t)

        # --- Compute Current Q Values ---
        # Get Q values for current states from main Q network
        q_values = self.q_network(obs_t)
        # Get the Q value corresponding to the action actually taken: Q(s, a)
        action_q_values = torch.gather(q_values, dim=1, index=acts_t)

        # --- Compute Loss ---
        # Using Smooth L1 loss (Huber loss) which is less sensitive to outliers than MSE
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # --- Gradient Descent ---
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network weights

        self.scheduler.step()  # learning rate decay

    def save_models(self):
        """Save current episode model and update best model if applicable"""
        # Only save models after a certain number of episodes
        if self.episode_count <= SAVE_VIDEO_STEP_COUNT_THRESHOLD:
            return

        if self.episode_count % 10 == 0:
            try:
                # Save Current Episode Model
                episode_model_name = f"episode_{self.episode_count}_reward_{self.episode_reward:.2f}_dqn_model.pth"
                episode_model_path = os.path.join(self.episode_model_dir, episode_model_name)
                torch.save(self.q_network.state_dict(), episode_model_path)
                self.get_logger().info(f"@@**@@ Saved episode model to {episode_model_path}")
            except Exception as e:
                self.get_logger().error(f"!!**!! Failed to save episode model: {e}")

        # Save Best Model if Current Episode is Better
        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            try:
                # Save with episode number for history tracking
                numbered_best_path = f"{self.best_model_path}_ep_{self.episode_count}_reward_{self.episode_reward}.pth"
                torch.save(self.q_network.state_dict(), numbered_best_path)

                # Also save as the standard best model file
                torch.save(self.q_network.state_dict(), f"{self.best_model_path}.pth")

                self.get_logger().info(
                    f"@@@**--**@@@ Saved NEW BEST model! Episode: {self.episode_count}, Reward: {self.best_episode_reward:.2f}"
                )
            except Exception as e:
                self.get_logger().error(f"🔥 Failed to save new best model: {e}")