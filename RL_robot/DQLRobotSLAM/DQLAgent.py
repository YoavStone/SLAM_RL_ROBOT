import itertools
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imageio.v2 as imageio
import os
import time
import rclpy
from rclpy.node import Node

from .DQN import DQN
from .DQL_ENV import DQLEnv

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 100
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
SAVE_THRESHOLD = 50  # Lower threshold for Gazebo environment


class DQLAgent(Node):
    def __init__(self, learning_mode=True, model_path='', best_model_name="best_dqn_gazebo_model.pth"):
        super().__init__('dql_agent')

        self.declare_parameter('learning_mode', learning_mode)
        self.declare_parameter('model_path', model_path)
        self.declare_parameter('best_model_name', best_model_name)

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.best_model_name = self.get_parameter('best_model_name').value

        # Setup environment
        self.env = DQLEnv()

        # Make sure observation space is initialized
        while self.env.observation_space is None:
            if self.env.update_observation_space():
                print("Observation space initialized successfully!")
                break
            else:
                rclpy.spin_once(self.env.gazebo_env, timeout_sec=0.1)
                print("Waiting for observation space to be initialized...")
                time.sleep(0.5)

        print(f"Observation space shape: {self.env.observation_space.shape}")

        # Initialize networks
        self.q_network = DQN(self.env)
        self.target_net = DQN(self.env)

        # Load model if path is provided
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.q_network.load_state_dict(torch.load(self.model_path))
                self.get_logger().info(f"🔄 Loaded model from {self.model_path}")
                if not self.learning_mode:
                    self.target_net.load_state_dict(self.q_network.state_dict())
            except Exception as e:
                self.get_logger().error(f"⚠️ Failed to load model from {self.model_path}: {e}")
        else:
            self.target_net.load_state_dict(self.q_network.state_dict())
            if self.model_path:
                self.get_logger().warn(f"⚠️ Model path {self.model_path} not found, starting fresh")

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        # Replay and reward tracking
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=100)
        self.episode_reward = 0.0

        # For saving the best model
        self.best_mean_reward = -float('inf')

        # Where to save models
        self.best_model_path = os.path.join("src/RL_robot/saved_networks/network_params/", self.best_model_name)
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

        # Timer for training/execution loop
        self.create_timer(0.01, self.timer_callback)

        # State for training loop
        self.training_initialized = False
        self.current_obs = None
        self.steps = 0
        self.episode_count = 0

        self.get_logger().info(f"DQL Agent initialized in {'learning' if self.learning_mode else 'execution'} mode")

    def timer_callback(self):
        """Main loop for training or execution"""
        if self.learning_mode:
            if not self.training_initialized:
                self.initialize_training()
            else:
                self.train_step()
        else:
            self.execute_step()

    def initialize_training(self):
        """Initialize replay buffer before starting training"""
        self.get_logger().info("Initializing replay buffer...")

        self.current_obs, _ = self.env.reset()

        # Fill replay buffer with random actions
        while len(self.replay_buffer) < MIN_REPLAY_SIZE:
            action = self.env.action_space.sample()
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated

            self.replay_buffer.append((self.current_obs, action, reward, is_done, new_obs))

            if is_done:
                self.current_obs, _ = self.env.reset()
            else:
                self.current_obs = new_obs

            if len(self.replay_buffer) % 100 == 0:
                print(f"Replay buffer: {len(self.replay_buffer)}/{MIN_REPLAY_SIZE}")

        print("Replay buffer filled, starting training")
        self.training_initialized = True

    def train_step(self):
        """Execute one step of training"""
        # Calculate epsilon based on steps
        epsilon = np.interp(self.steps, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.q_network.act(self.current_obs)

        # Execute action
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        # Store transition
        self.replay_buffer.append((self.current_obs, action, reward, is_done, new_obs))

        # Update current observation
        self.current_obs = new_obs

        # Update episode reward
        self.episode_reward += reward

        # Reset if episode is done
        if is_done:
            self.current_obs, _ = self.env.reset()
            self.reward_buffer.append(self.episode_reward)
            self.episode_count += 1

            print(f"Episode {self.episode_count}: Reward={self.episode_reward:.2f}, Avg={np.mean(self.reward_buffer):.2f}")
            self.episode_reward = 0.0

        # Learn from batch
        self.learn_step()

        # Update target network periodically
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_network.state_dict())
            print(f"Updated target network at step {self.steps}")

        # Save model if performance improves
        if len(self.reward_buffer) >= 5 and np.mean(self.reward_buffer) >= SAVE_THRESHOLD and np.mean(
                self.reward_buffer) > self.best_mean_reward:
            self.best_mean_reward = np.mean(self.reward_buffer)
            self.save_model()

        # Log progress
        if self.steps % 1000 == 0:
            print(f"Step: {self.steps}, Epsilon: {epsilon:.3f}, Avg reward: {np.mean(self.reward_buffer):.2f}")

        self.steps += 1

    def execute_step(self):
        """Execute the trained policy without learning"""
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0.0
            print("Starting new episode in execution mode")

        # Choose action using trained policy
        action = self.q_network.act(self.current_obs)

        # Execute action
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        # Update current observation and reward
        self.current_obs = new_obs
        self.episode_reward += reward

        # Reset if episode is done
        if is_done:
            print(f"Episode complete, reward: {self.episode_reward:.2f}")
            self.current_obs, _ = self.env.reset()
            self.episode_count += 1
            self.episode_reward = 0.0

    def learn_step(self):
        """Learn from a batch of experiences"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Sample batch of experiences
        transitions = random.sample(self.replay_buffer, BATCH_SIZE)
        obs_batch = np.asarray([t[0] for t in transitions])
        act_batch = np.asarray([t[1] for t in transitions])
        rew_batch = np.asarray([t[2] for t in transitions])
        done_batch = np.asarray([t[3] for t in transitions])
        next_obs_batch = np.asarray([t[4] for t in transitions])

        # Convert to tensors
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        acts_t = torch.as_tensor(act_batch, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rew_batch, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(done_batch, dtype=torch.float32).unsqueeze(-1)
        next_obs_t = torch.as_tensor(next_obs_batch, dtype=torch.float32)

        # Compute target Q values
        with torch.no_grad():
            target_q = self.target_net(next_obs_t)
            max_target_q = target_q.max(dim=1, keepdim=True)[0]
            targets = rews_t + GAMMA * (1 - dones_t) * max_target_q

        # Compute current Q values
        q_vals = self.q_network(obs_t)
        action_q_vals = torch.gather(q_vals, dim=1, index=acts_t)

        # Compute loss and update
        loss = nn.functional.smooth_l1_loss(action_q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        """Save the current model"""
        torch.save(self.q_network.state_dict(), self.best_model_path)
        self.get_logger().info(
            f"🏆 Saved best model with avg reward {self.best_mean_reward:.2f} at {self.best_model_path}")