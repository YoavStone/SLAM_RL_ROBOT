import itertools
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
import rclpy
from rclpy.node import Node
import os

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
SAVE_THRESHOLD = 400  # Threshold for saving the best model


class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, observation):
        observation_t = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(observation_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        return max_q_index.detach().item()


class DQLAgentNode(Node):
    def __init__(self):
        super().__init__('dql_agent_node')

        # ROS2 parameters
        self.declare_parameter('learning_mode', True)
        self.declare_parameter('model_path', '')

        self.learning_mode = self.get_parameter('learning_mode').value
        self.model_path = self.get_parameter('model_path').value

        print(f"Learning mode: {self.learning_mode}")
        print(f"Model path: {self.model_path}")

        # Initialize agent
        self.agent = DQLAgent(learning_mode=self.learning_mode, model_path=self.model_path)

        # Start the agent
        if self.learning_mode:
            print("Starting agent in learning mode")
            self.agent.init_replay_buffer()
            self.agent.train()
        else:
            print("Starting agent in execution mode")
            self.agent.run()


class DQLAgent:
    def __init__(self, env_name="CartPole-v1", learning_mode=True, model_path=''):
        self.env = gym.make(env_name)
        self.q_network = DQN(self.env)
        self.target_net = DQN(self.env)
        self.learning_mode = learning_mode

        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            try:
                self.q_network.load_state_dict(torch.load(model_path))
                print(f"🔄 Loaded model from {model_path}")
                # If we're in execution mode, we should also load the model to target network
                if not learning_mode:
                    self.target_net.load_state_dict(self.q_network.state_dict())
            except Exception as e:
                print(f"⚠️ Failed to load model from {model_path}: {e}")
        else:
            self.target_net.load_state_dict(self.q_network.state_dict())
            if model_path:
                print(f"⚠️ Model path {model_path} not found, starting with a fresh model")

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=100)
        self.epsilon_reward = 0.0

        # For saving the best model
        self.best_mean_reward = -float('inf')  # Store the best mean reward

        # Where to save best model
        best_model_name = "best_dqn_model.pth"
        best_model_path = "src/RL_robot/saved_networks/network_params/"
        self.best_model_path = best_model_path + best_model_name
        # Where to save model video
        vid_name = "dqn_cartpole_vid"
        vid_path = "src/RL_robot/saved_networks/video/"
        self.video_path = vid_path + vid_name

    def init_replay_buffer(self):
        if not self.learning_mode:
            return

        obs, _ = self.env.reset()
        for _ in range(MIN_REPLAY_SIZE):
            action = self.env.action_space.sample()
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated
            self.replay_buffer.append((obs, action, reward, is_done, new_obs))
            obs = new_obs
            if is_done:
                obs, _ = self.env.reset()

    def train(self):
        if not self.learning_mode:
            print("Agent is not in learning mode. Call run() instead.")
            return

        obs, _ = self.env.reset()
        for step in itertools.count():
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            action = self.env.action_space.sample() if random.random() < epsilon else self.q_network.act(obs)

            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated
            self.replay_buffer.append((obs, action, reward, is_done, new_obs))
            obs = new_obs
            self.epsilon_reward += reward

            if is_done:
                obs, _ = self.env.reset()
                self.reward_buffer.append(self.epsilon_reward)
                self.epsilon_reward = 0.0

            self.learn_step()

            if step % TARGET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.q_network.state_dict())

            # Save the model if the mean reward exceeds the threshold and it's the best so far
            if np.mean(self.reward_buffer) >= SAVE_THRESHOLD and np.mean(self.reward_buffer) > self.best_mean_reward:
                self.best_mean_reward = np.mean(self.reward_buffer)
                self.save_model()
                self.save_video(
                    path=f'{self.video_path}{self.best_mean_reward}.gif')  # save vid of best performing model

            if step % 1000 == 0:
                print(f"\nStep: {step}")
                print(f"Avg reward: {np.mean(self.reward_buffer)}")
                print(f"Epsilon: {epsilon}")

    def run(self):
        """Run the agent in execution mode (no learning)"""
        print("Running agent in execution mode (no learning)")
        steps = 0
        episodes = 0
        total_reward = 0

        try:
            while True:  # Run continuously
                obs, _ = self.env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = self.q_network.act(obs)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1

                    if steps % 100 == 0:
                        print(f"Step: {steps}, Episode: {episodes}, Current episode reward: {episode_reward}")

                episodes += 1
                total_reward += episode_reward
                avg_reward = total_reward / episodes

                print(f"\nEpisode {episodes} complete")
                print(f"Reward: {episode_reward}")
                print(f"Average reward: {avg_reward}")

                # Optional: save video periodically
                if episodes % 10 == 0:
                    self.save_video(path=f'{self.video_path}_execution_{episodes}.gif')
        except KeyboardInterrupt:
            print("\nStopping execution mode")
            return

    def learn_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        transitions = random.sample(self.replay_buffer, BATCH_SIZE)
        obs_batch = np.asarray([t[0] for t in transitions])
        act_batch = np.asarray([t[1] for t in transitions])
        rew_batch = np.asarray([t[2] for t in transitions])
        done_batch = np.asarray([t[3] for t in transitions])
        next_obs_batch = np.asarray([t[4] for t in transitions])

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        acts_t = torch.as_tensor(act_batch, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rew_batch, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(done_batch, dtype=torch.float32).unsqueeze(-1)
        next_obs_t = torch.as_tensor(next_obs_batch, dtype=torch.float32)

        with torch.no_grad():
            target_q = self.target_net(next_obs_t)
            max_target_q = target_q.max(dim=1, keepdim=True)[0]
            targets = rews_t + GAMMA * (1 - dones_t) * max_target_q

        q_vals = self.q_network(obs_t)
        action_q_vals = torch.gather(q_vals, dim=1, index=acts_t)

        loss = nn.functional.smooth_l1_loss(action_q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_video(self, path="dqn_cartpole.gif", steps=500):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        env_render = gym.make("CartPole-v1", render_mode="rgb_array")
        obs, _ = env_render.reset()
        frames = []
        for _ in range(steps):
            frames.append(env_render.render())
            action = self.q_network.act(obs)
            obs, _, term, trunc, _ = env_render.step(action)
            if term or trunc:
                break
        imageio.mimsave(path, frames, fps=30)
        print(f"🎥 Saved performance video to {path}")

    def save_model(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

        torch.save(self.q_network.state_dict(), self.best_model_path)
        print(f"🏆 Saved best model at {self.best_model_path}")


def main(args=None):
    rclpy.init(args=args)
    node = DQLAgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped cleanly")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()