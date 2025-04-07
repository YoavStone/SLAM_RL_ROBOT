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