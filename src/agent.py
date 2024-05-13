import torch
import torch.optim as optim
import numpy as np

class Agent:
    def __init__(self, env, algorithm='dqn', device='cuda'):
        self.device = torch.device(device)
        self.env = env
        self.algorithm = algorithm.lower()
        self.n_actions = env.action_space.n
        state, _ = env.reset()
        self.n_observations = len(self.get_state(state))

        # Create the policy model based on the chosen algorithm
        if self.algorithm == 'dqn':
            from .models.dqn import DQN
            self.policy_model = DQN(self.n_observations, self.n_actions).to(self.device)
            self.target_model = DQN(self.n_observations, self.n_actions).to(self.device)
            self.target_model.load_state_dict(self.policy_model.state_dict())
        elif self.algorithm == 'ppo':
            pass
        elif self.algorithm == 'a2c':
            pass
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=1e-4, amsgrad=True)
        self.steps_done = 0

    def get_state(self, obs):
        """Convert the observation dictionary to a state vector"""
        state = np.concatenate((obs['grid'].flatten(),
                                obs['time_step'],
                                obs['unit_health'],
                                obs['enemy_health'],
                                obs['current_unit'],
                                obs['current_move_index'],
                                obs['current_dance_pattern'].flatten()))
        return state

    def train(self, num_episodes):
        """Train the agent for a specified number of episodes"""
        self.policy_model.train_agent(self, num_episodes, self.device)
