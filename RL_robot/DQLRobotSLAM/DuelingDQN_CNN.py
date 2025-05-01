import torch
import torch.nn as nn
import numpy as np
import math


class DuelingDQN_CNN(nn.Module):
    """
    Dueling DQN with CNN for spatial processing of map data:
    - Separates input into map data (processed with CNN) and non-map features
    - Processes map data through convolutional layers
    - Processes non-map features through fully connected layers
    - Combines features before the dueling architecture split
    - Uses dueling architecture (value and advantage streams)

    DQN "chart":
    map input -> cnn -> 1600
    non map input -> 64 -> 64
    cnn output + non map output = combined network -> 512 -> 256 -> (splits into two networks)
    -> (Dueling dqn) -> 128 -> state value
    -> (dqn) -> 128 -> action values
    """

    def __init__(self, observation_space_shape, action_space_n):
        super().__init__()

        # Calculate the total input features
        total_in_features = int(np.prod(observation_space_shape))

        # Constants for map processing
        self.map_size = 40  # 40x40 cells (6m x 6m area with 0.15m resolution)

        # Calculate number of non-map features (position, velocity, distance sensors)
        self.non_map_features = 22  # 4 (grid_position) + 2 (velocities) + 16 (wall distances)
        self.map_features = total_in_features - self.non_map_features

        # Validate our assumptions about the input shape
        assert self.map_features > 0, f"Map features calculation error: {self.map_features}"

        # Calculate actual map dimensions (should be square, but verify)
        self.map_dim = int(math.sqrt(self.map_features))
        assert self.map_dim * self.map_dim == self.map_features, f"Map is not square: {self.map_features} cells"

        # CNN for map processing
        self.map_cnn = nn.Sequential(
            # First convolutional layer
            # Input: 1 channel, 40x40 (assuming map is 40x40)
            # Output: 16 channels, 20x20 (after 3x3 kernel with stride 2)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Second convolutional layer
            # Input: 16 channels, 20x20
            # Output: 32 channels, 10x10 (after 3x3 kernel with stride 2)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Third convolutional layer
            # Input: 32 channels, 10x10
            # Output: 64 channels, 5x5 (after 3x3 kernel with stride 2)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Flatten the output for the linear layers
            nn.Flatten()
        )

        # Calculate CNN output size: 64 channels x 5 x 5 = 1600
        cnn_output_size = 64 * 5 * 5

        # Network for non-map features
        self.non_map_network = nn.Sequential(
            nn.Linear(self.non_map_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Combined feature network
        self.combined_network = nn.Sequential(
            nn.Linear(cnn_output_size + 64, 512),  # Combine CNN and non-map features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
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
        # Split input into map and non-map parts
        non_map_data = x[:, :self.non_map_features]
        map_data = x[:, self.non_map_features:]

        # Reshape map data for CNN
        # From [batch_size, map_features] to [batch_size, 1, map_dim, map_dim]
        batch_size = map_data.shape[0]
        map_data_reshaped = map_data.view(batch_size, 1, self.map_dim, self.map_dim)

        # Process map data through CNN
        map_features = self.map_cnn(map_data_reshaped)

        # Process non-map features
        non_map_features = self.non_map_network(non_map_data)

        # Combine features
        combined = torch.cat([map_features, non_map_features], dim=1)

        # Process through combined network
        features = self.combined_network(combined)

        # Dueling architecture
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

    def act(self, observation):
        """Select action based on a single observation"""
        observation_t = torch.as_tensor(observation, dtype=torch.float32)
        q_values = self(observation_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        return max_q_index.detach().item()