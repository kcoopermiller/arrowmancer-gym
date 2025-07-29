import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt
import math

# GRPO Hyperparameters
BATCH_SIZE = 16
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 1e-4
GROUP_SIZE = 4  # Number of samples per state for group comparison
CLIP_EPSILON = 0.2  # PPO-style clipping parameter
ENTROPY_COEFF = 0.01  # Entropy regularization coefficient

# Define a named tuple for storing experiences in GRPO
Experience = namedtuple('Experience',
                       ('state', 'action', 'log_prob', 'reward', 'done', 'advantage'))

class GRPOBuffer:
    """Buffer to store experiences for GRPO training"""
    def __init__(self, capacity=10000):
        self.buffer = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save an experience"""
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class GRPOActor(nn.Module):
    """Policy network for GRPO"""
    def __init__(self, n_observations, n_actions):
        super(GRPOActor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, n_actions)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        # Use softmax to get action probabilities
        return F.softmax(self.layer5(x), dim=-1)

class GRPO(nn.Module):
    """Group Relative Policy Optimization (GRPO) implementation"""
    
    def __init__(self, n_observations, n_actions):
        super(GRPO, self).__init__()
        self.n_actions = n_actions
        self.buffer = GRPOBuffer(10000)
        
        # Policy network (actor)
        self.policy_net = GRPOActor(n_observations, n_actions)
        
        # For PPO-style updates, we need a reference to the old policy
        self.old_policy_net = GRPOActor(n_observations, n_actions)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
    def forward(self, x):
        """Forward pass through the policy network"""
        return self.policy_net(x)
    
    def select_action(self, state, deterministic=False):
        """Select an action using the current policy"""
        with torch.no_grad():
            probs = self.policy_net(state)
            if deterministic:
                return torch.argmax(probs, dim=-1)
            else:
                # Sample from the probability distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action, log_prob
    
    def collect_group_experiences(self, agent, state, group_size=GROUP_SIZE):
        """Collect a group of experiences from the same state for relative comparison"""
        group_experiences = []
        rewards = []
        
        for _ in range(group_size):
            # Reset environment to the same state for group sampling
            agent.env.reset()
            # Note: In a real implementation, you'd want to restore the exact state
            # For now, we'll collect experiences from similar states
            
            with torch.no_grad():
                action, log_prob = self.select_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = agent.env.step(action.item())
            done = terminated or truncated
            
            group_experiences.append({
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'done': done
            })
            rewards.append(reward)
        
        # Calculate relative advantages (core of GRPO)
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1.0
        
        # Normalize rewards relative to the group
        for i, exp in enumerate(group_experiences):
            advantage = (rewards[i] - mean_reward) / std_reward
            self.buffer.push(
                exp['state'],
                exp['action'],
                exp['log_prob'],
                exp['reward'],
                exp['done'],
                advantage
            )
        
        return group_experiences[-1]  # Return last experience for continuing episode
    
    def update_policy(self, agent):
        """Update the policy using GRPO algorithm"""
        if len(self.buffer) < BATCH_SIZE:
            return None
        
        # Sample batch from buffer
        experiences = self.buffer.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        old_log_prob_batch = torch.cat(batch.log_prob)
        advantage_batch = torch.tensor(batch.advantage, dtype=torch.float32)
        
        # Get current policy probabilities
        current_probs = self.policy_net(state_batch)
        current_dist = torch.distributions.Categorical(current_probs)
        current_log_probs = current_dist.log_prob(action_batch)
        
        # Get old policy probabilities (for PPO-style clipping)
        with torch.no_grad():
            old_probs = self.old_policy_net(state_batch)
            old_dist = torch.distributions.Categorical(old_probs)
            # Use stored log probs for better stability
        
        # Calculate probability ratio
        ratio = torch.exp(current_log_probs - old_log_prob_batch)
        
        # Calculate policy loss with clipping (PPO-style)
        surr1 = ratio * advantage_batch
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantage_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy regularization to encourage exploration
        entropy = current_dist.entropy().mean()
        entropy_loss = -ENTROPY_COEFF * entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Update policy
        agent.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        agent.optimizer.step()
        
        return total_loss.item()
    
    def update_old_policy(self):
        """Update the old policy network (soft update)"""
        # For GRPO, we can use hard updates every few steps, or soft updates
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
    
    def train_agent(self, agent, num_episodes, device='cuda'):
        """Train the GRPO agent for a specified number of episodes"""
        losses = []
        episode_rewards = []
        episode_durations = []
        
        print("Starting GRPO training...")
        
        for episode in tqdm(range(num_episodes), desc='GRPO Training'):
            state, _ = agent.env.reset()
            state = torch.tensor(agent.get_state(state), dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward = 0
            episode_duration = 0
            episode_losses = []
            
            for step in count():
                # Collect group experiences for GRPO
                if step % GROUP_SIZE == 0:  # Collect group every few steps
                    exp = self.collect_group_experiences(agent, state, GROUP_SIZE)
                    state = torch.tensor(agent.get_state(exp['reward']), dtype=torch.float32, device=device).unsqueeze(0) if not exp['done'] else None
                else:
                    # Regular action selection
                    action, log_prob = self.select_action(state)
                    next_state, reward, terminated, truncated, _ = agent.env.step(action.item())
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_duration += 1
                    
                    if done:
                        state = None
                    else:
                        state = torch.tensor(agent.get_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
                
                # Update policy
                if len(self.buffer) >= BATCH_SIZE and step % 4 == 0:  # Update every 4 steps
                    loss = self.update_policy(agent)
                    if loss is not None:
                        episode_losses.append(loss)
                
                # Update old policy periodically
                if step % 20 == 0:
                    self.update_old_policy()
                
                if state is None:  # Episode ended
                    break
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_durations.append(episode_duration)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Clear buffer periodically to prevent overfitting
            if episode % 50 == 0:
                self.buffer.clear()
        
        print('GRPO training completed')
        
        # Save the trained model
        torch.save(agent.policy_model.state_dict(), 'Models/grpo.pth')
        
        # Plot training curves
        if losses:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(losses)
            plt.xlabel('Episode')
            plt.ylabel('Average Loss')
            plt.title('GRPO Training Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Episode Reward')
            plt.title('Episode Rewards')
            
            plt.subplot(1, 3, 3)
            plt.plot(episode_durations)
            plt.xlabel('Episode')
            plt.ylabel('Episode Duration')
            plt.title('Episode Durations')
            
            plt.tight_layout()
            plt.show()
        
        return {
            'losses': losses,
            'episode_rewards': episode_rewards,
            'episode_durations': episode_durations
        }