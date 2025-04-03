# import os
# from collections import deque
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import itertools
# import random
#
# import sys
# print(sys.path)
# import gym
# print(gym.__version__)
# import numpy as np
# print(np.__version__)
#
#
# GAMMA = 0.99
# LEARNING_RATE = 5e-4
# BATCH_SIZE = 32
# BUFFER_SIZE = 50000
# MIN_REPLAY_SIZE = 1000
# EPSILON_START = 1.0
# EPSILON_END = 0.02
# EPSILON_DECAY = 10000
# TARGET_UPDATE_FREQ = 1000
#
#
# class DQN(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#
#         in_features = int(np.prod(env.observation_space.shape))
#
#         self.net = nn.Sequential(
#             nn.Linear(in_features, 64),
#             nn.Tanh(),
#             nn.Linear(64, env.action_space.n)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#     def act(self, observation):
#         observation_t = torch.as_tensor(observation, dtype=torch.float32)
#         q_values = self(observation_t.unsqueeze(0))
#
#         max_q_index = torch.argmax(q_values, dim=1)[0]
#         action = max_q_index.detach().item()
#
#         return action
#
#
# def main():
#     env = gym.make("CartPole-v1")
#
#     replay_buffer = deque(maxlen=BUFFER_SIZE)
#     reward_buffer = deque([0.0], maxlen=100)
#
#     epsilon_reward = 0.0
#
#     q_network = DQN(env)
#     target_net = DQN(env)
#
#     target_net.load_state_dict(q_network.state_dict())  # Copy weights
#
#     optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
#
#     # initialize replay buffer
#     observation = env.reset()
#
#     for _ in range(MIN_REPLAY_SIZE):
#
#         action = env.action_space.sample()
#
#         new_observation, reward, is_done, _ = env.step(action)
#         transition = (observation, action, reward, is_done, new_observation)
#         replay_buffer.append(transition)
#         observation = new_observation
#
#         if is_done:
#             observation = env.reset()
#
#     # main training loop
#     observation = env.reset()
#
#     for step in itertools.count():
#
#         epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
#         rnd_sample = random.random()
#
#         if rnd_sample <= epsilon:
#             action = env.action_space.sample()
#         else:
#             action = q_network.act(observation)
#
#         new_observation, reward, is_done, _ = env.step(action)
#         transition = (observation, action, reward, is_done, new_observation)
#         replay_buffer.append(transition)
#         observation = new_observation
#
#         epsilon_reward += reward
#
#         if is_done:
#             observation = env.reset()
#
#             reward_buffer.append(epsilon_reward)
#             epsilon_reward = 0.0
#
#         # start gradient step
#         transitions = random.sample(replay_buffer, BATCH_SIZE)
#
#         observations = np.asarray([t[0] for t in transitions])
#         actions = np.asarray([t[1] for t in transitions])
#         rewards = np.asarray([t[2] for t in transitions])
#         dones = np.asarray([t[3] for t in transitions])
#         new_observations = np.asarray([t[4] for t in transitions])
#
#         observations_t = torch.as_tensor(observations, dtype=torch.float32)
#         actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
#         rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
#         dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
#         new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32)
#
#         # compute targets for loss func
#         target_q_values = target_net(new_observations_t)
#         max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
#
#         targets = rewards_t + GAMMA * (1-dones_t) * max_target_q_values # if state is terminal tar = reward
#
#         # compute loss
#         q_values = q_network(observations_t)
#
#         action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
#
#         loss = nn.functional.smooth_l1_loss(action_q_values, targets)
#
#         # gradient descent
#         optimizer.zero_grad()
#         loss.backwards()
#         optimizer.step()
#
#         # update target network
#         if step % TARGET_UPDATE_FREQ == 0:
#             target_net.load_state_dict(q_network.state_dict())
#
#         # logging
#         if step % 1000 == 0:
#             print()
#             print('step: ', step)
#             print("avg reward: ", np.mean(reward_buffer))
#
#
# if __name__ == "__main__":
#     main()


import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import random

import sys
print(sys.path)
import gymnasium as gym # change import gym to import gymnasium as gym
print(gym.__version__)
import numpy as np
print(np.__version__)


GAMMA = 0.99
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000


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
        action = max_q_index.detach().item()

        return action


def main():
    env = gym.make("CartPole-v1")

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    reward_buffer = deque([0.0], maxlen=100)

    epsilon_reward = 0.0

    q_network = DQN(env)
    target_net = DQN(env)

    target_net.load_state_dict(q_network.state_dict())  # Copy weights

    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # initialize replay buffer
    observation, _ = env.reset() # gymnasium returns observation and info

    for _ in range(MIN_REPLAY_SIZE):

        action = env.action_space.sample()

        new_observation, reward, terminated, truncated, _ = env.step(action) # gymnasium returns terminated and truncated
        is_done = terminated or truncated # create the is_done variable
        transition = (observation, action, reward, is_done, new_observation)
        replay_buffer.append(transition)
        observation = new_observation

        if is_done:
            observation, _ = env.reset() # gymnasium returns observation and info

    # main training loop
    observation, _ = env.reset() # gymnasium returns observation and info

    for step in itertools.count():

        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        rnd_sample = random.random()

        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = q_network.act(observation)

        new_observation, reward, terminated, truncated, _ = env.step(action)
        is_done = terminated or truncated # create the is_done variable
        transition = (observation, action, reward, is_done, new_observation)
        replay_buffer.append(transition)
        observation = new_observation

        epsilon_reward += reward

        if is_done:
            observation, _ = env.reset() # gymnasium returns observation and info

            reward_buffer.append(epsilon_reward)
            epsilon_reward = 0.0

        # after solved, watch it play
        if len(reward_buffer) >= 100:
            if np.mean(reward_buffer) >= 200:
                env_render = gym.make("CartPole-v1", render_mode="human")
                observation_render, _ = env_render.reset()
                while True:
                    action = q_network.act(observation_render)
                    observation_render, reward, terminated, truncated, _ = env_render.step(action)
                    env_render.render()
                    if terminated or truncated:
                        observation_render, _ = env_render.reset()

        # start gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        observations_t = torch.as_tensor(observations, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32)

        # compute targets for loss func
        target_q_values = target_net(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + GAMMA * (1-dones_t) * max_target_q_values # if state is terminal tar = reward

        # compute loss
        q_values = q_network(observations_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_network.state_dict())

        # logging
        if step % 1000 == 0:
            print()
            print('step: ', step)
            print("avg reward: ", np.mean(reward_buffer))
            print("epsilon: ", epsilon)


if __name__ == "__main__":
    main()