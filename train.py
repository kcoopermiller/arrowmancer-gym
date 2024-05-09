import torch
import torch.nn as nn
import torch.nn.functional as F
from src import Agent, device, Transition, ArrowmancerEnv
from itertools import count
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 128
GAMMA = 0.99 # Discount factor
TAU = 0.005 # Update rate of the target network

def optimize_model(agent):
    """Perform a single optimization step"""
    if len(agent.memory) < BATCH_SIZE:
        return None
    transitions = agent.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
    agent.optimizer.step()

    return loss.item()

def train(agent, num_episodes):
    """Train the agent for a specified number of episodes"""
    losses = []
    episode_durations = []
    for i_episode in tqdm(range(num_episodes), desc='Training'):
        state, info = agent.env.reset()
        state = torch.tensor(agent.get_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        episode_loss = [0.0]
        episode_duration = 0
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = agent.env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(agent.get_state(observation), dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)

            state = next_state
            episode_duration += 1

            loss = optimize_model(agent)
            if loss is not None:
                episode_loss.append(loss)

            # Update the target network using a soft update
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                losses.append(sum(episode_loss)/len(episode_loss))
                episode_durations.append(episode_duration)
                break

    print('Training completed')
    torch.save(agent.policy_net.state_dict(), 'model.pth')

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

if __name__ == "__main__":
    units = [
        {'name': 'Celine', 'banner': 'standard'}, 
        {'name': 'Kepler', 'banner': 'standard'}, 
        {'name': 'Zorn', 'banner': 'standard'}, 
    ] 
    env = ArrowmancerEnv(units)
    agent = Agent(env, units)
    train(agent, num_episodes=1000)
