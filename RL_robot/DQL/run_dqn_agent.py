import itertools
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import imageio.v2 as imageio

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


class DQLAgent:
    def __init__(self, env_name="CartPole-v1"):
        self.env = gym.make(env_name)
        self.q_network = DQN(self.env)
        self.target_net = DQN(self.env)
        self.target_net.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.reward_buffer = deque([0.0], maxlen=100)
        self.epsilon_reward = 0.0

        # For saving the best model
        self.best_mean_reward = -float('inf')  # Store the best mean reward
        self.best_model_path = "best_dqn_model.pth"

    def init_replay_buffer(self):
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
                self.save_video(path=f'dqn_cartpole{self.best_mean_reward}.gif') # save vid of best preforming model

            if step % 1000 == 0:
                print(f"\nStep: {step}")
                print(f"Avg reward: {np.mean(self.reward_buffer)}")
                print(f"Epsilon: {epsilon}")

    def learn_step(self):
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
        torch.save(self.q_network.state_dict(), self.best_model_path)
        print(f"🏆 Saved best model at {self.best_model_path}")


def main():
    agent = DQLAgent()
    agent.init_replay_buffer()
    agent.train()


if __name__ == "__main__":
    main()
