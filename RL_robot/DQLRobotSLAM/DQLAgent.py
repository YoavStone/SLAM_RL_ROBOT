import itertools
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

from .DQN import DQN
from .DQL_ENV import DQLEnv
from sim_control.sim_reset_handler import SimulationResetHandler
from .initialize_replay_buffer import initialize_replay_buffer

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000  # Minimum experiences in buffer before learning starts
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 30000  # Steps over which epsilon decays
TARGET_UPDATE_FREQ = 1000  # Steps between updating the target network


class DQLAgent(Node):
    def __init__(self, learning_mode=True, model_path='', best_model_name="best_dqn_gazebo_model"):
        super().__init__('dql_agent')

        # Add buffer initialization function
        self.initialize_replay_buffer = initialize_replay_buffer.__get__(self)

        # Set hyperparameters as instance variables for access in initialize_replay_buffer
        self.min_replay_size = MIN_REPLAY_SIZE

        self.episode_end_pub = self.create_publisher(Empty, 'episode_end', 10)

        # --- Parameters ---
        self.declare_parameter('learning_mode', learning_mode)
        self.declare_parameter('model_path', model_path)  # Path to load a pre-trained model (optional)
        self.declare_parameter('best_model_name', best_model_name)  # Filename for the best performing model
        # Add new epsilon parameters
        self.declare_parameter('epsilon_start', EPSILON_START)  # Starting epsilon value
        self.declare_parameter('epsilon_end', EPSILON_END)  # Final epsilon value
        self.declare_parameter('epsilon_decay', EPSILON_DECAY)  # Steps for decay

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.best_model_name = self.get_parameter('best_model_name').value
        # Get epsilon parameters with defaults
        self.epsilon_start = self.get_parameter('epsilon_start').value
        self.epsilon_end = self.get_parameter('epsilon_end').value
        self.epsilon_decay = self.get_parameter('epsilon_decay').value

        self.get_logger().info(f"--- DQL Agent Configuration ---")
        self.get_logger().info(f"Learning Mode: {self.learning_mode}")
        self.get_logger().info(f"Load Model Path: '{self.model_path}'")
        self.get_logger().info(f"Best Model Filename: '{self.best_model_name}'")
        self.get_logger().info(f"Epsilon Start: {self.epsilon_start}")
        self.get_logger().info(f"Epsilon End: {self.epsilon_end}")
        self.get_logger().info(f"Epsilon Decay Steps: {self.epsilon_decay}")
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

        self.reset_handler = SimulationResetHandler(self, self.env)

        self.get_logger().info(f"Observation space shape: {self.env.observation_space.shape}")
        self.get_logger().info(f"Action space size: {self.env.action_space.n}")

        # --- Network Initialization ---
        self.q_network = DQN(self.env)
        self.target_net = DQN(self.env)

        # --- Load Pre-trained Model (if specified) ---
        load_path = self.model_path if self.model_path else self.best_model_name  # Try loading best if no specific path given
        # Construct full path relative to package - ADJUST IF NEEDED
        # Assuming execution from workspace root or script location allows this relative path
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
                self.get_logger().error(f"⚠️ Failed to load model from {effective_load_path}: {e}. Starting fresh.")
                # Ensure target net is synced with the randomly initialized q_network
                self.target_net.load_state_dict(self.q_network.state_dict())
        else:
            self.target_net.load_state_dict(self.q_network.state_dict())  # Sync target net
            if self.model_path:
                self.get_logger().warn(f"⚠️ Specified model path '{self.model_path}' not found.")
            elif load_path == self.best_model_name:
                self.get_logger().info(f"⚠️ Default best model '{potential_best_path}' not found. Starting fresh.")
            else:
                self.get_logger().info("No model path specified or found. Starting fresh.")

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        # --- Replay Buffer & Reward Tracking ---
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=100)  # Still useful for logging avg reward
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

        self.get_logger().info(
            f"🚀 DQL Agent initialized successfully in {'LEARNING' if self.learning_mode else 'EXECUTION'} mode.")
        if self.learning_mode and len(self.replay_buffer) < MIN_REPLAY_SIZE:
            self.get_logger().info("Initializing replay buffer...")
            # Start buffer initialization immediately if in learning mode
            self.initialize_replay_buffer()

    def timer_callback(self):
        """Main loop called by the ROS timer."""
        if self.learning_mode:
            if not self.training_initialized:
                # Try to initialize buffer if not done yet
                if not self.initialize_replay_buffer():
                    self.get_logger().warn("Replay buffer initialization pending or failed, skipping training step.")
                    return  # Wait for next timer call

            # Check if environment is ready for training
            if self.env.observation_space is None or self.env.reset_handler.is_reset_in_progress():
                self.get_logger().info(
                    "Environment not ready (reset in progress or invalid observation space), skipping training step.",
                    throttle_duration_sec=5.0)
                time.sleep(0.5)  # Small delay
                return

            # Proceed with training step only if buffer is initialized and environment is ready
            self.train_step()
        else:
            # In execution mode, also check if environment is ready
            if self.env.observation_space is None or self.env.reset_handler.is_reset_in_progress():
                self.get_logger().info(
                    "Environment not ready (reset in progress or invalid observation space), skipping execution step.",
                    throttle_duration_sec=5.0)
                time.sleep(0.5)  # Small delay
                return

            self.execute_step()

    def train_step(self):
        """Executes one step of interaction and learning."""
        # Skip execution during reset
        if self.reset_handler.is_reset_in_progress():
            self.get_logger().debug("Skipping training step during reset")
            time.sleep(0.5)  # Reduced wait time to check more frequently
            return

        if self.current_obs is None:
            self.get_logger().warn("Current observation is None at start of train_step. Attempting reset.")
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to reset environment in train_step. Stopping training.")
                # Potentially stop the timer or node here
                return
            self.episode_reward = 0.0  # Reset reward for safety

        # --- Action Selection (Epsilon-Greedy) ---
        epsilon = np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        if random.random() < epsilon:
            action = self.env.action_space.sample()  # Explore
        else:
            action = self.q_network.act(self.current_obs)  # Exploit

        # --- Environment Step ---
        new_obs, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated

        # Check if step was during reset or returned None
        if new_obs is None or (info and info.get("reset_in_progress", False)):
            self.get_logger().warn("Received None observation or reset in progress during training step. Skipping.")
            # Try to reset if we're done or the observation is None
            if is_done or new_obs is None:
                self.episode_end_pub.publish(Empty())  # Trigger reset
                time.sleep(0.5)  # Small delay
            return  # Skip learning for this step

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
            self.reward_buffer.append(self.episode_reward)  # Add final reward to buffer for avg calculation
            mean_reward_100 = np.mean(self.reward_buffer)

            self.get_logger().info(
                f"--- Episode {self.episode_count} Finished --- \n"
                f"Reward: {self.episode_reward:.2f} | Avg Reward (Last 100): {mean_reward_100:.2f} \n"
                f"Steps: {self.steps} | Epsilon: {epsilon:.3f} \n"
                f"-------------------------------------"
            )

            # --- Save Current Episode Model ---
            self.save_models()

            # Reset for next episode
            self.current_obs = None  # Set to None so we'll reset on next call
            self.episode_reward = 0.0  # Reset episode reward

            # Publish episode end signal for external listeners (like episode_monitor)
            self.episode_end_pub.publish(Empty())
            # Give time for reset to complete
            time.sleep(5.0)  # Reduced wait time, will continue to wait as needed in next steps

        # --- Learning Step ---
        self.learn_step()

        # --- Update Target Network ---
        if self.steps % TARGET_UPDATE_FREQ == 0 and self.steps > 0:
            self.target_net.load_state_dict(self.q_network.state_dict())
            self.get_logger().info(f"🔄 Updated target network at step {self.steps}")

        self.steps += 1

    def execute_step(self):
        """Executes the learned policy without exploration or learning."""
        # Skip execution during reset
        if self.reset_handler.is_reset_in_progress():
            self.get_logger().debug("Skipping execution step during reset")
            time.sleep(0.5)  # Reduced wait time
            return

        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to get initial observation in execution mode!")
                # Consider stopping or retrying
                return
            self.episode_reward = 0.0
            self.episode_count += 1  # Count episodes in execution mode too
            self.get_logger().info(f"Starting Execution Episode {self.episode_count}")

        # Choose action greedily (no exploration)
        action = self.q_network.act(self.current_obs)

        # Execute action
        new_obs, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated

        # Check if step was during reset or returned None
        if new_obs is None or (info and info.get("reset_in_progress", False)):
            self.get_logger().warn("Received None observation or reset in progress during execution step.")
            # Try to reset if we're done or the observation is None
            if is_done or new_obs is None:
                self.episode_end_pub.publish(Empty())  # Trigger reset
                time.sleep(0.5)  # Small delay
                self.current_obs = None  # Prepare for next call to reset
                self.get_logger().info(
                    f"Execution Episode {self.episode_count} ended early. Reward: {self.episode_reward:.2f}")
                self.episode_reward = 0.0
            return

        self.current_obs = new_obs
        self.episode_reward += reward

        if is_done:
            self.get_logger().info(
                f"Execution Episode {self.episode_count} complete. Final reward: {self.episode_reward:.2f}")
            # Reset for the next potential episode start
            self.current_obs = None  # Set to None so we'll reset on next call
            self.episode_reward = 0.0

            # Publish episode end signal
            self.episode_end_pub.publish(Empty())
            time.sleep(5.0)  # Wait for reset to complete

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
        # Optional: Gradient clipping (can help stability)
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()  # Update network weights

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
                torch.save(self.q_network.state_dict(), self.best_model_path + str(self.episode_count) + '.pth')
                self.get_logger().info(
                    f"🏆 Saved NEW BEST model! Episode: {self.episode_count}, Reward: {self.best_episode_reward:.2f}. Path: {self.best_model_path, self.episode_count}.pth"
                )
                torch.save(self.q_network.state_dict(), self.best_model_path +'.pth')
                self.get_logger().info(
                    f"🏆 Saved NEW BEST model! Episode: {self.episode_count}, Reward: {self.best_episode_reward:.2f}. Path: {self.best_model_path}.pth"
                )
            except Exception as e:
                self.get_logger().error(f"🔥 Failed to save new best model: {e}")


def main(args=None):
    rclpy.init(args=args)
    agent = DQLAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("Keyboard interrupt received, shutting down...")
    finally:
        # Clean up
        agent.get_logger().info("Shutting down DQL Agent")
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()