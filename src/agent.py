import torch
import torch.optim as optim
import numpy as np

# Kyaa~! This is our super kawaii AI agent class desu! 😍
class Agent:
    def __init__(self, env, algorithm='dqn', device='cuda'):
        # Initializing our AI-chan! OwO
        self.device = torch.device(device)  # Using GPU-senpai for maximum power! 💪
        self.env = env
        self.algorithm = algorithm.lower()
        self.n_actions = env.action_space.n  # So many possible actions, like choosing best waifu! 😳
        state, _ = env.reset()
        self.n_observations = len(self.get_state(state))  # Observing everything with our kawaii eyes desu~

        # Create the policy model based on the chosen algorithm (choosing our fighting style desu!)
        if self.algorithm == 'dqn':
            from .models.dqn import DQN
            # Creating our main neural network-chan! She will learn to be the best magical girl strategist! 🌟
            self.policy_model = DQN(self.n_observations, self.n_actions).to(self.device)
            # And her twin sister target network-chan for stable training! Twins are so moe~ 👯‍♀️
            self.target_model = DQN(self.n_observations, self.n_actions).to(self.device)
            self.target_model.load_state_dict(self.policy_model.state_dict())
        elif self.algorithm == 'ppo':
            pass
        elif self.algorithm == 'a2c':
            pass
        else:
            raise ValueError(f"B-baka! Invalid algorithm: {algorithm}! Only DQN-chan is implemented right now desu! (＞︿<)")

        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=1e-4, amsgrad=True)
        self.steps_done = 0

    def get_state(self, obs):
        """Convert the observation dictionary to a state vector (organizing all the kawaii data desu~)"""
        # Flattening all the magical girl data into one big sugoi vector! 🎀
        state = np.concatenate((obs['grid'].flatten(),  # The battlefield where our waifus dance!
                                obs['time_step'],       # Current moment in this epic anime battle!
                                obs['unit_health'],     # Health of our precious magical girls! ❤️
                                obs['enemy_health'],    # Enemy-kun's remaining HP desu
                                obs['current_unit'],    # Which magical girl is fighting now?
                                obs['current_move_index'],  # Progress in the kawaii dance sequence!
                                obs['current_dance_pattern'].flatten()))  # The sacred dance moves! 💃
        return state

    def train(self, num_episodes):
        """Train the agent for a specified number of episodes (teaching AI-chan the way of the magical girl!)"""
        # Time for intensive training arc! Our AI-chan will become the ultimate magical girl strategist! 🎯
        self.policy_model.train_agent(self, num_episodes, self.device)
