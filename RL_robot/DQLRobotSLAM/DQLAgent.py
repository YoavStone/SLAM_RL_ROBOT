# dql_agent.py (modified)

import itertools
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import imageio.v2 as imageio # Not used in the provided snippet, can be removed if not needed elsewhere
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from .DQN import DQN  # Assuming DQN.py is in the same directory
from .DQL_ENV import DQLEnv # Assuming DQL_ENV.py is in the same directory

# Hyperparameters (Keep relevant ones)
GAMMA = 0.99
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 100 # Minimum experiences in buffer before learning starts
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000 # Steps over which epsilon decays
TARGET_UPDATE_FREQ = 1000 # Steps between updating the target network


class DQLAgent(Node):
    def __init__(self, learning_mode=True, model_path='', best_model_name="best_dqn_gazebo_model.pth"):
        super().__init__('dql_agent')

        self.episode_end_pub = self.create_publisher(Empty, 'episode_end', 10)

        # --- Parameters ---
        self.declare_parameter('learning_mode', learning_mode)
        self.declare_parameter('model_path', model_path) # Path to load a pre-trained model (optional)
        self.declare_parameter('best_model_name', best_model_name) # Filename for the best performing model

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.best_model_name = self.get_parameter('best_model_name').value

        self.get_logger().info(f"--- DQL Agent Configuration ---")
        self.get_logger().info(f"Learning Mode: {self.learning_mode}")
        self.get_logger().info(f"Load Model Path: '{self.model_path}'")
        self.get_logger().info(f"Best Model Filename: '{self.best_model_name}'")
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

        # --- Network Initialization ---
        self.q_network = DQN(self.env)
        self.target_net = DQN(self.env)

        # --- Load Pre-trained Model (if specified) ---
        load_path = self.model_path if self.model_path else self.best_model_name # Try loading best if no specific path given
        # Construct full path relative to package - ADJUST IF NEEDED
        # Assuming execution from workspace root or script location allows this relative path
        potential_best_path = os.path.join("src/RL_robot/saved_networks/network_params/", load_path)

        effective_load_path = None
        if self.model_path and os.path.exists(self.model_path):
             effective_load_path = self.model_path # Use specific path if it exists
        elif os.path.exists(potential_best_path) and not self.model_path:
             effective_load_path = potential_best_path # Use default best path if it exists and no specific path given

        if effective_load_path:
            try:
                self.q_network.load_state_dict(torch.load(effective_load_path))
                self.target_net.load_state_dict(self.q_network.state_dict()) # Sync target net
                self.get_logger().info(f"✅ Successfully loaded model from {effective_load_path}")
            except Exception as e:
                self.get_logger().error(f"⚠️ Failed to load model from {effective_load_path}: {e}. Starting fresh.")
                # Ensure target net is synced with the randomly initialized q_network
                self.target_net.load_state_dict(self.q_network.state_dict())
        else:
            self.target_net.load_state_dict(self.q_network.state_dict()) # Sync target net
            if self.model_path:
                self.get_logger().warn(f"⚠️ Specified model path '{self.model_path}' not found.")
            elif load_path == self.best_model_name:
                 self.get_logger().info(f"⚠️ Default best model '{potential_best_path}' not found. Starting fresh.")
            else:
                 self.get_logger().info("No model path specified or found. Starting fresh.")


        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        # --- Replay Buffer & Reward Tracking ---
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=100) # Still useful for logging avg reward
        self.episode_reward = 0.0

        # --- Saving Logic ---
        self.best_episode_reward = -float('inf') # Track the best single episode reward
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
        self.timer = self.create_timer(0.01, self.timer_callback) # Timer for the main loop
        self.training_initialized = False
        self.current_obs = None
        self.steps = 0
        self.episode_count = 0 # Start counting episodes from 0

        self.get_logger().info(f"🚀 DQL Agent initialized successfully in {'LEARNING' if self.learning_mode else 'EXECUTION'} mode.")
        if self.learning_mode and len(self.replay_buffer) < MIN_REPLAY_SIZE :
             self.get_logger().info("Initializing replay buffer...")
             # Start buffer initialization immediately if in learning mode
             self.initialize_replay_buffer()


    def initialize_replay_buffer(self):
        """Fills the replay buffer with initial random experiences."""
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to get initial observation from environment!")
                # Consider shutting down or retrying
                return False # Indicate failure

        init_steps = 0
        while len(self.replay_buffer) < MIN_REPLAY_SIZE:
            action = self.env.action_space.sample()
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated

            if new_obs is None:
                 self.get_logger().warn("Received None observation during buffer initialization. Skipping step.")
                 # Attempt reset if stuck or handle error appropriately
                 if is_done:
                    obs_reset, _ = self.env.reset()
                    if obs_reset is None:
                         self.get_logger().error("Failed to reset environment during buffer init!")
                         return False
                    self.current_obs = obs_reset
                 continue # Skip adding this transition

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

            if init_steps % 100 == 0:
                self.get_logger().info(f"Replay buffer filling: {len(self.replay_buffer)}/{MIN_REPLAY_SIZE}")

        self.get_logger().info("✅ Replay buffer initialized.")
        self.training_initialized = True
        return True


    def timer_callback(self):
        """Main loop called by the ROS timer."""
        if self.learning_mode:
            if not self.training_initialized:
                 # Try to initialize buffer if not done yet
                 if not self.initialize_replay_buffer():
                      self.get_logger().warn("Replay buffer initialization pending or failed, skipping training step.")
                      return # Wait for next timer call
            # Proceed with training step only if buffer is initialized
            self.train_step()
        else:
            self.execute_step()


    def train_step(self):
        """Executes one step of interaction and learning."""
        if self.current_obs is None:
             self.get_logger().warn("Current observation is None at start of train_step. Attempting reset.")
             self.current_obs, _ = self.env.reset()
             if self.current_obs is None:
                  self.get_logger().error("Failed to reset environment in train_step. Stopping training.")
                  # Potentially stop the timer or node here
                  self.timer.cancel()
                  return
             self.episode_reward = 0.0 # Reset reward for safety


        # --- Action Selection (Epsilon-Greedy) ---
        epsilon = np.interp(self.steps, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        if random.random() < epsilon:
            action = self.env.action_space.sample() # Explore
        else:
            action = self.q_network.act(self.current_obs) # Exploit

        # --- Environment Step ---
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        if new_obs is None:
             self.get_logger().warn("Received None observation during training step. Skipping step.")
             # Decide how to handle this - maybe end episode prematurely?
             # For now, just log and potentially reset if 'done' flag was somehow set.
             if is_done:
                 obs_reset, _ = self.env.reset()
                 if obs_reset is None:
                     self.get_logger().error("Failed to reset environment after receiving None observation!")
                     self.timer.cancel() # Stop training if env becomes unresponsive
                     return
                 self.current_obs = obs_reset
                 self.episode_reward = 0.0
             return # Skip learning and buffer addition for this step


        # --- Store Experience ---
        # Ensure observations are numpy arrays
        current_obs_np = np.array(self.current_obs, dtype=np.float32)
        new_obs_np = np.array(new_obs, dtype=np.float32)
        self.replay_buffer.append((current_obs_np, action, reward, is_done, new_obs_np))

        self.current_obs = new_obs
        self.episode_reward += reward

        # --- Episode End Handling ---
        if is_done:
            self.episode_count += 1
            self.reward_buffer.append(self.episode_reward) # Add final reward to buffer for avg calculation
            mean_reward_100 = np.mean(self.reward_buffer)

            self.get_logger().info(
                f"--- Episode {self.episode_count} Finished --- \n"
                f"Reward: {self.episode_reward:.2f} | Avg Reward (Last 100): {mean_reward_100:.2f} \n"
                f"Steps: {self.steps} | Epsilon: {epsilon:.3f}"
                f"-------------------------------------"
            )

            # --- Save Current Episode Model ---
            self.save_models()

            # Reset for next episode
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to reset environment after episode end! Stopping training.")
                self.timer.cancel()
                return
            self.episode_reward = 0.0 # Reset episode reward

            # Publish episode end signal for external listeners (like episode_monitor)
            self.episode_end_pub.publish(Empty())


        # --- Learning Step ---
        self.learn_step()

        # --- Update Target Network ---
        if self.steps % TARGET_UPDATE_FREQ == 0 and self.steps > 0:
            self.target_net.load_state_dict(self.q_network.state_dict())
            self.get_logger().info(f"🔄 Updated target network at step {self.steps}")

        self.steps += 1


    def execute_step(self):
        """Executes the learned policy without exploration or learning."""
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                 self.get_logger().error("Failed to get initial observation in execution mode!")
                 # Consider stopping or retrying
                 return
            self.episode_reward = 0.0
            self.episode_count += 1 # Count episodes in execution mode too
            self.get_logger().info(f"Starting Execution Episode {self.episode_count}")

        # Choose action greedily (no exploration)
        action = self.q_network.act(self.current_obs)

        # Execute action
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        if new_obs is None:
             self.get_logger().warn("Received None observation during execution step.")
             # Handle appropriately, maybe end episode
             if is_done:
                 obs_reset, _ = self.env.reset()
                 self.current_obs = obs_reset # Prepare for next potential step/episode
                 self.get_logger().info(f"Execution Episode {self.episode_count} ended early due to None observation. Reward: {self.episode_reward:.2f}")
                 self.episode_reward = 0.0
                 self.episode_end_pub.publish(Empty()) # Signal end
             return


        self.current_obs = new_obs
        self.episode_reward += reward

        if is_done:
            self.get_logger().info(f"Execution Episode {self.episode_count} complete. Final reward: {self.episode_reward:.2f}")
            # Reset for the next potential episode start
            self.current_obs, _ = self.env.reset() # Reset immediately
            if self.current_obs is None:
                 self.get_logger().error("Failed to reset environment after execution episode!")
                 # Consider stopping
                 return
            self.episode_reward = 0.0
            # Don't increment episode count here, it's done at the start of the next call

            # Publish episode end signal
            self.episode_end_pub.publish(Empty())


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
        done_batch = np.array([t[3] for t in transitions], dtype=np.float32) # Use float for multiplication later
        next_obs_batch = np.array([t[4] for t in transitions], dtype=np.float32)


        obs_t = torch.as_tensor(obs_batch)
        acts_t = torch.as_tensor(act_batch).unsqueeze(-1) # Shape: [batch_size, 1]
        rews_t = torch.as_tensor(rew_batch).unsqueeze(-1) # Shape: [batch_size, 1]
        dones_t = torch.as_tensor(done_batch).unsqueeze(-1) # Shape: [batch_size, 1]
        next_obs_t = torch.as_tensor(next_obs_batch)

        # --- Compute Target Q Values ---
        with torch.no_grad(): # No gradients needed for target computation
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
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        # Optional: Gradient clipping (can help stability)
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step() # Update network weights

    def save_models(self):
        """Save current episode model and update best model if applicable"""
        # Save Current Episode Model
        try:
            episode_model_name = f"episode_{self.episode_count}_reward_{self.episode_reward:.2f}_dqn_model.pth"
            episode_model_path = os.path.join(self.episode_model_dir, episode_model_name)
            torch.save(self.q_network.state_dict(), episode_model_path)
            # self.get_logger().info(f"💾 Saved episode model to {episode_model_path}")  # Optional logging
        except Exception as e:
            self.get_logger().error(f"🔥 Failed to save episode model {episode_model_name}: {e}")

        # Save Best Model if Current Episode is Better
        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            try:
                torch.save(self.q_network.state_dict(), self.best_model_path)
                self.get_logger().info(
                    f"🏆 Saved NEW BEST model! Episode: {self.episode_count}, Reward: {self.best_episode_reward:.2f}. Path: {self.best_model_path}"
                )
            except Exception as e:
                self.get_logger().error(f"🔥 Failed to save new best model: {e}")
