import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt

class PPOActor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PPOActor, self).__init__()
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
        return F.softmax(self.layer5(x), dim=-1)

class PPOCritic(nn.Module):
    def __init__(self, n_observations):
        super(PPOCritic, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def push(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

class PPO:
    def __init__(self, n_observations, n_actions, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, entropy_coeff=0.01, value_coeff=0.5):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        
        self.actor = PPOActor(n_observations, n_actions)
        self.critic = PPOCritic(n_observations)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        self.memory = PPOMemory()
        self.mse_loss = nn.MSELoss()
        
    def get_action(self, state, device='cuda'):
        """Select an action based on the current state using the policy network"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        values = values + [next_value]
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def update(self, device='cuda'):
        """Update the policy and value networks"""
        # Convert lists to tensors
        states = torch.FloatTensor(self.memory.states).to(device)
        actions = torch.LongTensor(self.memory.actions).to(device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(device)
        rewards = self.memory.rewards
        values = self.memory.values
        dones = self.memory.dones
        
        # Compute returns using GAE
        next_value = 0  # Assuming episode ends
        returns = self.compute_gae(rewards, values, dones, next_value)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        advantages = returns - torch.FloatTensor(values).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy and value predictions
            action_probs = self.actor(states)
            current_values = self.critic(states).squeeze()
            
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratio for PPO clip loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(current_values, returns)
            entropy_loss = -self.entropy_coeff * entropy
            
            total_loss = actor_loss + self.value_coeff * critic_loss + entropy_loss
            
            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()
        
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()
    
    def train_agent(self, agent, num_episodes, update_interval=2048, device='cuda'):
        """Train the PPO agent for a specified number of episodes"""
        self.actor.to(device)
        self.critic.to(device)
        
        episode_rewards = []
        episode_durations = []
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        steps = 0
        
        for episode in tqdm(range(num_episodes), desc='Training PPO'):
            state, _ = agent.env.reset()
            state = agent.get_state(state)
            episode_reward = 0
            episode_duration = 0
            
            for _ in count():
                action, log_prob, value = self.get_action(state, device)
                next_state, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                self.memory.push(state, action, log_prob, reward, value, done)
                
                state = agent.get_state(next_state) if not done else None
                episode_reward += reward
                episode_duration += 1
                steps += 1
                
                # Update networks periodically or at episode end
                if steps % update_interval == 0 or done:
                    # Get next value for GAE computation
                    if not done and state is not None:
                        next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        with torch.no_grad():
                            next_value = self.critic(next_state_tensor).item()
                    else:
                        next_value = 0
                    
                    # Compute GAE with next_value
                    returns = self.compute_gae(self.memory.rewards, self.memory.values, 
                                             self.memory.dones, next_value)
                    
                    # Convert to tensors for update
                    states_tensor = torch.FloatTensor(self.memory.states).to(device)
                    actions_tensor = torch.LongTensor(self.memory.actions).to(device)
                    old_log_probs_tensor = torch.FloatTensor(self.memory.log_probs).to(device)
                    returns_tensor = torch.FloatTensor(returns).to(device)
                    values_tensor = torch.FloatTensor(self.memory.values).to(device)
                    
                    # Normalize advantages
                    advantages = returns_tensor - values_tensor
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # PPO update
                    for _ in range(self.k_epochs):
                        action_probs = self.actor(states_tensor)
                        current_values = self.critic(states_tensor).squeeze()
                        
                        dist = Categorical(action_probs)
                        new_log_probs = dist.log_prob(actions_tensor)
                        entropy = dist.entropy().mean()
                        
                        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                        
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = self.mse_loss(current_values, returns_tensor)
                        entropy_loss = -self.entropy_coeff * entropy
                        
                        total_loss = actor_loss + self.value_coeff * critic_loss + entropy_loss
                        
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                        self.optimizer.step()
                    
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    
                    self.memory.clear()
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_durations.append(episode_duration)
                    break
                    
                if not done:
                    state = agent.get_state(next_state)
        
        print('PPO Training completed')
        
        # Save models
        torch.save(self.actor.state_dict(), 'Models/ppo_actor.pth')
        torch.save(self.critic.state_dict(), 'Models/ppo_critic.pth')
        
        # Plot training metrics
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        
        plt.subplot(1, 3, 2)
        plt.plot(episode_durations)
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.title('Episode Durations')
        
        plt.subplot(1, 3, 3)
        if actor_losses:
            plt.plot(actor_losses, label='Actor Loss')
            plt.plot(critic_losses, label='Critic Loss')
            plt.plot(entropy_losses, label='Entropy Loss')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
        
        plt.tight_layout()
        plt.show()