# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define a named tuple for storing transitions in the replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition in the replay memory"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions from the replay memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self, env, units):
        self.env = env
        self.n_actions = env.action_space.n
        state, _ = env.reset()
        self.n_observations = len(self.get_state(state))

        # Create the policy network and the target network
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def get_state(self, obs):
        """Convert the observation dictionary to a state vector"""
        state = np.concatenate((obs['grid'].flatten(),
                                obs['unit_positions'].flatten(),
                                obs['enemy_attacks'].flatten(),
                                obs['time_step'],
                                [obs['current_unit']],
                                obs['unit_health'],
                                obs['enemy_health'],
                                [obs['current_move_index']],
                                obs['current_dance_pattern'].flatten()))
        return state

    def select_action(self, state):
        """Select an action based on the current state using an epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = 0.05 + (0.9 - 0.05) * \
            math.exp(-1. * self.steps_done / 1000)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)