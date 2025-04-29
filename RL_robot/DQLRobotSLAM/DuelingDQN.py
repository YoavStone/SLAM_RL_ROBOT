import torch
import torch.nn as nn
import numpy as np


class DuelingDQN(nn.Module):
    """
    DQN of size input -> 512 -> 512 -> 256 -> (splits into two networks)
    -> (Dueling dqn) -> 128 -> state value
    -> (dqn) -> 128 -> action values
    """
    def __init__(self, observation_space_shape, action_space_n):
        super().__init__()
        in_features = int(np.prod(observation_space_shape))

        # Shared feature network - enlarged
        self.feature_network = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_n)
        )

    def forward(self, x):
        features = self.feature_network(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

    def act(self, observation):
        observation_t = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(observation_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        return max_q_index.detach().item()