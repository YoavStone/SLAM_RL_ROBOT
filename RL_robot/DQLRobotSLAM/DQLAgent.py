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
from std_msgs.msg import Int32, Bool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from .DQNetwork import DQNetwork
from .DuelingDQN import DuelingDQN
from .DuelingDQN_CNN import DuelingDQN_CNN
from .DQLEnv import DQLEnv
from .ToggleDemonstrationBuffer import ToggleDemonstrationBuffer

# Constants for learning
from constants.constants import (
    GAMMA,
    LEARNING_RATE_START,
    LEARNING_RATE_END,
    LEARNING_RATE_DECAY,
    BATCH_SIZE,
    BUFFER_SIZE,
    MIN_REPLAY_SIZE,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    TARGET_UPDATE_FREQ,
    SAVE_NETWORK_STEP_COUNT_THRESHOLD
)


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
        # parameter for if running a simulation or running the robot
        self.declare_parameter('is_sim', True)  # For Env

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.best_model_name = self.get_parameter('best_model_name').value
        # Get epsilon parameters with defaults
        self.epsilon_start = self.get_parameter('epsilon_start').value
        self.epsilon_end = self.get_parameter('epsilon_end').value
        self.epsilon_decay = self.get_parameter('epsilon_decay').value
        # Get spawn location parameter for reset handler
        self.spawn_location = self.get_parameter('spawn_location').value
        self.is_sim = self.get_parameter('is_sim').value

        self.get_logger().info(f"--- DQL Agent Configuration ---")
        self.get_logger().info(f"Learning Mode: {self.learning_mode}")
        self.get_logger().info(f"Load Model Path: '{self.model_path}'")
        self.get_logger().info(f"Best Model Filename: '{self.best_model_name}'")
        self.get_logger().info(f"Epsilon Start: {self.epsilon_start}")
        self.get_logger().info(f"Epsilon End: {self.epsilon_end}")
        self.get_logger().info(f"Epsilon Decay Steps: {self.epsilon_decay}")
        self.get_logger().info(f"Spawn Location: '{self.spawn_location}'")
        self.get_logger().info(f"Is Simulation: '{self.is_sim}'")
        self.get_logger().info(f"-------------------------------")

        # --- Environment Setup ---
        self.env = DQLEnv(is_sim=self.is_sim)

        # Wait for observation space initialization
        while self.env.observation_space is None:
            if self.env.update_observation_space():
                self.get_logger().info("Observation space initialized successfully!")
                break
            else:
                # Allow ROS callbacks in the environment node to process
                rclpy.spin_once(self.env.sensors_processor, timeout_sec=0.1)
                self.get_logger().info("Waiting for observation space to be initialized...", throttle_duration_sec=2.0)
                time.sleep(0.5)

        self.get_logger().info(f"Observation space shape: {self.env.observation_space.shape}")
        self.get_logger().info(f"Action space size: {self.env.action_space.n}")

        # # --- Network Initialization --- for when using standard dqn
        # self.q_network = DQNetwork(self.env.observation_space.shape, self.env.action_space.n)
        # self.target_net = DQNetwork(self.env.observation_space.shape, self.env.action_space.n)

        # # --- Network Initialization --- for when using normal dueling dqn structure
        # self.q_network = DuelingDQN(self.env.observation_space.shape, self.env.action_space.n)
        # self.target_net = DuelingDQN(self.env.observation_space.shape, self.env.action_space.n)

        # --- Network Initialization ---
        self.q_network = DuelingDQN_CNN(self.env.observation_space.shape, self.env.action_space.n)
        self.target_net = DuelingDQN_CNN(self.env.observation_space.shape, self.env.action_space.n)

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
                self.get_logger().info(f"**--** Successfully loaded model from {effective_load_path}")
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

        # Log the learning rate configuration
        self.get_logger().info(f"Learning rate scheduler configured:")
        self.get_logger().info(f"Initial rate: {LEARNING_RATE_START}")
        self.get_logger().info(f"Final rate: {LEARNING_RATE_END}")
        self.get_logger().info(f"Decay steps: {LEARNING_RATE_DECAY}")

        # --- Replay Buffer & Reward Tracking ---
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=1000)  # For logging avg reward
        self.episode_reward = 0.0

        # --- for epsilon greedy - better random action picking ---
        self.current_random_action = None
        self.random_action_counter = 0

        # --- Saving Logic ---
        self.best_episode_reward = -float('inf')  # Track the best single episode reward
        self.best_model_dir = os.path.join("src/RL_robot/saved_networks/network_params/")
        self.episode_model_dir = os.path.join("src/RL_robot/saved_networks/episode_network_params/")
        self.best_model_path = os.path.join(self.best_model_dir, self.best_model_name)

        # Create directories if they don't exist
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.episode_model_dir, exist_ok=True)
        self.get_logger().info(f"Models will be saved to:")
        self.get_logger().info(f"Best: {self.best_model_dir}")
        self.get_logger().info(f"Episodes: {self.episode_model_dir}")

        # so that timer callback doesnt disregard the action and toggle subscriptions
        self.subscription_callback_group = ReentrantCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()

        # buffer for demonstrations of good human state action reward...
        self.demo_buffer = ToggleDemonstrationBuffer(
            max_demos=50000,
            demo_batch_ratio=0.3,
            auto_timeout=300,  # 5 minutes auto-timeout
            save_path="src/RL_robot/saved_networks/saved_demonstrations/demo_buffer.pkl"
        )
        # Set up logger and ROS node connection
        self.demo_buffer.set_logger(self.get_logger())
        # Create subscribers for demo buffer
        self.action_sub = self.create_subscription(
            Int32,
            '/demo/action',
            self.action_callback,
            10,
            callback_group=self.subscription_callback_group
        )

        self.toggle_sub = self.create_subscription(
            Bool,
            '/demo/toggle',
            self.toggle_callback,
            10,
            callback_group=self.subscription_callback_group
        )

        self.get_logger().info("Demo buffer initialized with ROS subscribers")

        # Print instructions about toggle demo recording
        self.get_logger().info("Press 'p' to toggle demonstration recording mode")
        self.get_logger().info("When in demo mode, use 'w/a/s/d' to control the robot, 'x' to stop, 'p' to exit demo mode")
        self.get_logger().info(f"Auto-timeout for demonstration mode: 5 minutes")

        # --- State for Training/Execution Loop ---
        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=self.timer_callback_group)  # Timer for the main loop
        self.training_initialized = False
        self.current_obs = None
        self.steps = 0
        self.episode_count = 0  # Start counting episodes from 0

        # Initialize optimizer with starting learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE_START)
        # Create LR decay (smooth according to lr_lambda func)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.lr_lambda
        )

        self.get_logger().info(f"DQL Agent initialized successfully in {'LEARNING' if self.learning_mode else 'EXECUTION'} mode.")
        if self.learning_mode and len(self.replay_buffer) < MIN_REPLAY_SIZE:
            self.get_logger().info("Initializing replay buffer...")
            # Start buffer initialization immediately if in learning mode
            self.initialize_replay_buffer()

    def action_callback(self, msg):
        self.demo_buffer.action_callback(msg)

    def toggle_callback(self, msg):
        self.demo_buffer.toggle_callback(msg)

    # Lambda function that decays to exactly the target rate and then stays there
    def lr_lambda(self, epoch):
        if self.steps >= LEARNING_RATE_DECAY:
            # After reaching decay steps, maintain final rate
            return LEARNING_RATE_END / LEARNING_RATE_START
        else:
            # Linear decay from start to end rate
            decay_factor = 1.0 - self.steps / LEARNING_RATE_DECAY
            normalized_lr = LEARNING_RATE_START * decay_factor + LEARNING_RATE_END * (1 - decay_factor)
            return normalized_lr / LEARNING_RATE_START

    def random_action_picking(self, epsilon):
        """
        Picks a random action and decides how many times to repeat it based on epsilon.

        Args:
            epsilon (float): Current exploration rate between 0 and 1

        Returns:
            int: Action to execute
        """
        # If we still have repetitions of the previous action to execute
        if self.random_action_counter > 0:
            self.random_action_counter -= 1
            return self.current_random_action

        # Pick a new random action
        action = self.env.action_space.sample()

        # between 1-5 repetitions, fewer as epsilon decreases
        max_repetitions = max(1, int(5 * epsilon))
        repetitions = random.randint(1, max_repetitions)

        # Store the action and set the counter
        self.current_random_action = action
        self.random_action_counter = repetitions - 1  # -1 because we'll execute once now

        return action

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

            if random.random() < epsilon or self.random_action_counter > 0:
                action = self.random_action_picking(epsilon)  # Explore with repetition
            else:
                action = self.q_network.act(self.current_obs)  # Exploit

            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated

            if new_obs is None:
                self.get_logger().warn("Received None observation during buffer initialization. Skipping step.")
                # Attempt reset if needed
                if is_done:
                    obs_reset, _ = self.env.reset()
                    self.current_random_action = None
                    self.random_action_counter = 0
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
                self.current_random_action = None
                self.random_action_counter = 0
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
        # Ensure correct modes for learning, in training activate weight dropout to prevent over fitting
        self.q_network.train()  # dropout active
        self.target_net.eval()  # No dropout

        # If current_obs is None, we need to reset or initialize
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            self.current_random_action = None
            self.random_action_counter = 0
            if self.current_obs is None:
                self.get_logger().error("Failed to get observation after reset in train_step")
                return
            self.episode_reward = 0.0

        # Update the current state in the demo buffer
        self.demo_buffer.set_current_state(self.current_obs)

        # Check if user has toggled demonstration mode
        is_demo_mode = self.demo_buffer.check_for_toggle()

        # Action selection based on mode
        if is_demo_mode:
            # Human demonstration mode - get action from keyboard
            action = self.demo_buffer.get_action()
        else:
            # AI control mode - epsilon-greedy selection
            epsilon = np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
            if random.random() < epsilon or self.random_action_counter > 0:
                action = self.random_action_picking(epsilon)  # Explore with repetition
            else:
                action = self.q_network.act(self.current_obs)  # Exploit

        # --- Environment Step ---
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        # Check if we received a valid observation
        if new_obs is None:
            self.get_logger().warn("Received None observation during training step. Attempting reset.")
            self.current_obs, _ = self.env.reset()
            self.current_random_action = None
            self.random_action_counter = 0
            return  # Skip this step

        # Process the transition based on mode
        current_obs_np = np.array(self.current_obs, dtype=np.float32)
        new_obs_np = np.array(new_obs, dtype=np.float32)

        if is_demo_mode:
            # In demo mode, record the transition in the demo buffer
            self.demo_buffer.add_transition(current_obs_np, action, reward, new_obs_np, is_done)
        else:
            # In normal mode, store in replay buffer
            self.replay_buffer.append((current_obs_np, action, reward, is_done, new_obs_np))

        # Update current state and reward
        self.current_obs = new_obs
        self.episode_reward += reward

        # --- Episode End Handling ---
        if is_done:
            self.handle_episode_end()
        else:
            # Learning step (only in normal mode)
            if not is_demo_mode:
                self.learn_step()

                # Update Target Network
                if self.steps % TARGET_UPDATE_FREQ == 0 and self.steps > 0:
                    self.target_net.load_state_dict(self.q_network.state_dict())
                    self.get_logger().info(f"**--** Updated target network at step {self.steps}")

        # Only increment steps in normal mode
        if not is_demo_mode:
            self.steps += 1

    def execute_step(self):
        """Executes the learned policy without exploration or learning."""
        # Ensure correct modes for execution, no dropout for max performance
        self.q_network.eval()  # No dropout
        self.target_net.eval()  # No dropout

        # Initialize observation if needed
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            self.current_random_action = None
            self.random_action_counter = 0
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
            self.current_random_action = None
            self.random_action_counter = 0
            return

        self.current_obs = new_obs
        self.episode_reward += reward

        # Handle episode end
        if is_done:
            self.get_logger().info(f"Execution episode complete. Final reward: {self.episode_reward:.2f}")
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0.0
            self.current_random_action = None
            self.random_action_counter = 0

    def handle_episode_end(self):
        """Handle the end of a training episode"""
        self.current_random_action = None
        self.random_action_counter = 0

        mean_reward_100 = 0.0  # Initialize with a default value

        if not self.demo_buffer.check_for_toggle():
            self.episode_count += 1
            if self.reward_buffer is not None:  # Add null check
                self.reward_buffer.append(self.episode_reward)
                if len(self.reward_buffer) > 0:  # Add size check
                    mean_reward_100 = np.mean(self.reward_buffer)

        # Log episode details
        current_lr = 0.0
        if self.optimizer and hasattr(self.optimizer, 'param_groups') and len(self.optimizer.param_groups) > 0:
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
        try:
            self.current_obs, _ = self.env.reset()
            if self.current_obs is None:
                self.get_logger().error("Failed to reset environment after episode end!")
                return
            self.episode_reward = 0.0  # Reset episode reward
        except Exception as e:
            self.get_logger().error(f"Error during environment reset: {e}")

    def learn_step(self):
        """Performs a gradient descent step based on a batch of experiences."""
        # Only learn if buffer has enough samples
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Check if we should use demonstration data
        mixed_batch = self.demo_buffer.get_mixed_batch(self.replay_buffer, BATCH_SIZE)

        if mixed_batch is not None:
            # Use mixed batch from demonstrations and replay buffer
            obs_batch, act_batch, rew_batch, done_batch, next_obs_batch = mixed_batch
        else:
            # Sample from replay buffer only
            transitions = random.sample(self.replay_buffer, BATCH_SIZE)
            # Unpack and convert transitions
            obs_batch = np.array([t[0] for t in transitions], dtype=np.float32)
            act_batch = np.array([t[1] for t in transitions], dtype=np.int64)
            rew_batch = np.array([t[2] for t in transitions], dtype=np.float32)
            done_batch = np.array([t[3] for t in transitions], dtype=np.float32)
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
        if self.episode_count <= SAVE_NETWORK_STEP_COUNT_THRESHOLD:
            return

        if self.demo_buffer.check_for_toggle():
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
                numbered_best_path = f"{self.best_model_path}_ep_{self.episode_count}_reward_{self.episode_reward:.2f}.pth"
                torch.save(self.q_network.state_dict(), numbered_best_path)

                # Also save as the standard best model file
                torch.save(self.q_network.state_dict(), f"{self.best_model_path}.pth")

                self.get_logger().info(
                    f"@@@**--**@@@ Saved NEW BEST model! Episode: {self.episode_count}, Reward: {self.best_episode_reward:.2f}"
                )
            except Exception as e:
                self.get_logger().error(f"🔥 Failed to save new best model: {e}")

    def cleanup(self):
        """Clean up resources when node is destroyed"""
        if self.env is not None:
            self.env.close()