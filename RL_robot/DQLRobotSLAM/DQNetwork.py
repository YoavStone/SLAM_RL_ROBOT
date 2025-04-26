import torch
import torch.nn as nn
import numpy as np


class DQNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_n):
        super().__init__()
        in_features = int(np.prod(observation_space_shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),  # input=>128=>64=>output
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, observation):
        observation_t = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(observation_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        return max_q_index.detach().item()