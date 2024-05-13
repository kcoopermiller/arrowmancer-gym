import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt
import math

BATCH_SIZE = 128
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Update rate of the target network

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
        self.memory = ReplayMemory(10000)
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

    def train_agent(self, agent, num_episodes, device='cuda'):
        """Train the DQN agent for a specified number of episodes"""
        losses = []
        episode_durations = []
        for _ in tqdm(range(num_episodes), desc='Training'):
            state, _ = agent.env.reset()
            state = torch.tensor(agent.get_state(state), dtype=torch.float32, device=device).unsqueeze(0)
            episode_loss = [0.0]
            episode_duration = 0
            for _ in count():
                action = self._select_action(agent, state, device)
                observation, reward, terminated, truncated, _ = agent.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(agent.get_state(observation), dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state
                episode_duration += 1

                loss = self._optimize_model(agent, device)
                if loss is not None:
                    episode_loss.append(loss)

                # Update the target network using a soft update
                target_model_state_dict = agent.target_model.state_dict()
                policy_model_state_dict = agent.policy_model.state_dict()
                for key in policy_model_state_dict:
                    target_model_state_dict[key] = policy_model_state_dict[key] * TAU + target_model_state_dict[key] * (1 - TAU)
                agent.target_model.load_state_dict(target_model_state_dict)

                if done:
                    losses.append(sum(episode_loss) / len(episode_loss))
                    episode_durations.append(episode_duration)
                    break

        print('Training completed')
        torch.save(agent.policy_model.state_dict(), 'Models/dqn.pth')

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Training Loss')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(episode_durations)
        plt.xlabel('Episode')
        plt.ylabel('Episode Duration')
        plt.title('Episode Durations')
        plt.show()


    def _optimize_model(self, agent, device='cuda'):
        """Perform a single optimization step"""
        if len(self.memory) < BATCH_SIZE:
            return None
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = agent.policy_model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = agent.target_model(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        agent.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(agent.policy_model.parameters(), 100)
        agent.optimizer.step()

        return loss.item()
    
    def _select_action(self, agent, state, device='cuda'):
        """Select an action based on the current state using an epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = 0.05 + (0.9 - 0.05) * \
            math.exp(-1. * agent.steps_done / 1000)
        agent.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return agent.policy_model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[agent.env.action_space.sample()]], device=device, dtype=torch.long)