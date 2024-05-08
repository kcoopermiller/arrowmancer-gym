import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dance import dance_patterns
import pygame
import pandas as pd

class ArrowmancerEnv(gym.Env):
    def __init__(self, units):
        super(ArrowmancerEnv, self).__init__()
        self.grid_size = 3  # 3x3
        self.cell_size = 200  # 200x200 pixels
        self.screen = None
        self.padding = 75 # Padding around the grid
        self.time_step = 0
        self.enemy_attack_delay = 3  # Number of time steps before enemy attack activates
        self.units = self._get_units(units) 
        self.num_units = len(units)
        self.unit_health = np.ones(self.num_units) # Health points for each unit
        self.enemy_health = 1
        self.current_unit = 0  # Index of the current unit performing the dance
        self.current_move_index = 0  # Index of the current move in the dance pattern

        self.action_space = spaces.Discrete(15)  # (Up, Right, Down, Left, Attack) x 3 Units
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'unit_positions': spaces.Box(low=0, high=self.grid_size - 1, shape=(self.num_units, 2), dtype=int),
            'enemy_attacks': spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=int),
            'time_step': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=int),
            'current_unit': spaces.Discrete(self.num_units),
            'unit_health': spaces.Box(low=0, high=1, shape=(self.num_units,), dtype=float),
            'enemy_health': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'current_move_index': spaces.Discrete(6),
            'current_dance_pattern': spaces.Box(low=-13, high=13, shape=(6, 2), dtype=int)
        })
        self.reset()

    def reset(self):
        self.time_step = 0
        # Reset the grid to an empty state
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Randomly initialize unit positions on the grid
        self.unit_positions = np.random.randint(0, self.grid_size, size=(self.num_units, 2))
        # Reset health points
        self.unit_health = np.ones(self.num_units) 
        self.enemy_health = 1
        # Reset enemy attacks to an empty state
        self.enemy_attacks = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_unit = 0  # Reset the current unit to the first unit
        self.current_move_index = 0  # Reset the current move index to the beginning
        return self._get_obs()

    def step(self, action):
        unit = action // 5  # Choose the current unit to move based on the action
        action = action % 5
        unit_pos = self.unit_positions[unit]
        reward = 0

        if action < 4:  # Move action
            # Update the position of the current unit based on the action
            if action == 0:  # Move up
                new_pos = [unit_pos[0] - 1, unit_pos[1]]
            elif action == 1:  # Move right
                new_pos = [unit_pos[0], unit_pos[1] + 1]
            elif action == 2:  # Move down
                new_pos = [unit_pos[0] + 1, unit_pos[1]]
            elif action == 3:  # Move left
                new_pos = [unit_pos[0], unit_pos[1] - 1]

            # Check if the new position is valid and update the unit's position
            if self._is_valid_position(new_pos):
                # Check if another unit is present at the new position
                if self._is_unit_at_position(new_pos, unit):
                    # If so, swap the positions of the two units
                    # TODO: Should only be able to swap if adjacent to an edge
                    for i, pos in enumerate(self.unit_positions):
                        if (pos == new_pos).all() and i != unit:
                            self.unit_positions[i] = unit_pos
                            self.unit_positions[unit] = new_pos
                            break
                else:
                    self.unit_positions[unit] = new_pos

            # Check if the current unit's dance move is satisfied
            unit = self.units[self.current_unit] # Get the current unit's dance pattern info
            move = dance_patterns[unit['zodiac']][unit['dance']][self.current_move_index]
            if self._check_dance_move(move):
                reward = 1 + 0.1 * self.current_move_index  # 10% increase for combos
                self.current_move_index += 1
                # Move to the next unit if the current unit has completed all dance moves
                if self.current_move_index >= len(dance_patterns[unit['zodiac']][unit['dance']]):
                    self.current_unit = (self.current_unit + 1) % self.num_units
                    self.current_move_index = 0

        else:  # Attack action
            # TODO: Should only be able to attack if adjacent to enemy
            attack_strength = 0.05 + 0.02 * self.current_move_index  # 5% base strength + 2% increase for combos
            self.enemy_health -= attack_strength
            if self.enemy_health <= 0:
                reward = 100 - self.time_step  # Higher reward for faster enemy defeat
                terminated = True
                return self._get_obs()[0], reward, terminated, False, {}
            self.current_unit = (self.current_unit + 1) % self.num_units
            self.current_move_index = 0

        # Penalty for each time step
        reward -= 0.1
        self.time_step += 1

        # Decrement the delay for existing enemy attacks
        self.enemy_attacks = np.where(self.enemy_attacks > 0, self.enemy_attacks - 1, 0)        
        # Generate enemy attacks targeting either one grid or a column of 3 grids
        attack = np.random.choice([0, 1], p=[0.7, 0.3])  # 70% chance of no attack
        attack_type = np.random.choice(['single', 'column'], p=[0.8, 0.2])  # 80% chance of single grid attack
        if attack:
            if attack_type == 'single':
                # Attack a single random grid
                attack_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                self.enemy_attacks[attack_pos] = self.enemy_attack_delay  # Set the delay as the attack value
            else:
                # Attack a random column of 3 grids
                attack_col = np.random.randint(0, self.grid_size)
                self.enemy_attacks[:, attack_col] = self.enemy_attack_delay

        # Check if units are hit by enemy attacks
        terminated = False
        truncated = False
        for i, pos in enumerate(self.unit_positions):
            if self.enemy_attacks[pos[0], pos[1]] == 1:
                self.unit_health[i] -= 0.1  # Reduce unit health by 10%
                reward -= 0.1  # Penalty for unit hit
                self.enemy_attacks[pos[0], pos[1]] = 0 # Reset the enemy attack
                if self.unit_health[i] <= 0:
                    terminated = True
                    reward = -100  # Negative reward for unit defeat
                    break

        return self._get_obs()[0], reward, terminated, truncated, {}

    def _get_obs(self):
        # Update the grid representation with unit positions
        self.grid.fill(0)
        for pos in self.unit_positions:
            self.grid[pos[0], pos[1]] = 1
        unit = self.units[self.current_unit]
        current_dance_pattern = dance_patterns[unit['zodiac']][unit['dance']]
        # Change tuples to numpy arrays and pad the dance pattern with zeros
        current_dance_pattern = np.array([np.array(np.pad(move, (0, 2 - len(move)),'constant')) for move in current_dance_pattern])
        # Pad the dance pattern with zeros to have a fixed length of 6
        current_dance_pattern = np.pad(current_dance_pattern, ((0, 6 - len(current_dance_pattern)), (0, 0)), 'constant')
        return {
            'grid': self.grid,
            'unit_positions': self.unit_positions,
            'enemy_attacks': self.enemy_attacks,
            'time_step': np.array([self.time_step]), # Convert scalar to numpy array for consistency
            'current_unit': self.current_unit,
            'unit_health': self.unit_health,
            'enemy_health': np.array([self.enemy_health]), # Ditto
            'current_move_index': self.current_move_index,
            'current_dance_pattern': current_dance_pattern
        }, 0 # Dummy reward

    def _check_dance_move(self, move):
        for pos in move:
            if pos > 4 or pos < -4:
                if not self._check_anchor_move(pos):
                    return False
            else:
                if not self._check_non_anchor_move(pos):
                    return False
        return True

    def _check_non_anchor_move(self, move):
        current_pos = self.unit_positions[self.current_unit]
        target_pos = current_pos + self._offset_to_position(move)
        # Check if the target position is valid and if a unit is present at that position
        return self._is_valid_position(target_pos) and self._is_unit_at_position(target_pos, self.current_unit)

    def _check_anchor_move(self, move):
        current_pos = self.unit_positions[self.current_unit]
        # Check if the current unit is at the required anchor position
        if move == -13:
            return current_pos[0] == 0
        elif move == -11:
            return current_pos[1] == 0
        elif move == 11:
            return current_pos[1] == self.grid_size - 1
        elif move == 13:
            return current_pos[0] == self.grid_size - 1
        return False

    def _offset_to_position(self, offset):
        # Convert the offset to a position on the grid
        # Offset values:   Mapped positions:
        # -4 -3 -2         [-1, -1] [-1, 0] [-1, 1]
        # -1  0  1         [0, -1]  [0, 0]  [0, 1]
        # 2  3  4          [1, -1]  [1, 0]  [1, 1]
        return [(offset + 1) // 3 - 1, (offset + 1) % 3 - 1]

    def _is_valid_position(self, pos):
        # Check if the given position is within the grid boundaries
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _is_unit_at_position(self, pos, unit):
        # Check if a unit (other than the passed unit) is present at the given position
        for i, unit_pos in enumerate(self.unit_positions):
            if i != unit and (unit_pos == pos).all():
                return True
        return False

    def _get_units(self, units):
        # Get dance pattern info for each unit from the banner data
        for unit in units:
            df = pd.read_csv(f"assets/{unit['banner']}_banners/{unit['banner']}_banners.csv")
            if unit['name'] not in df['Name'].values:
                raise ValueError(f"Unit {unit['name']} not found in {unit['banner']} banners. Currently not supported: Curry XIII, Scarletti, Saika, Linnaeus, Clover, Marilyn")
            else:
                unit['zodiac'] = df[df['Name'] == unit['name']]['Zodiac'].values[0]
                unit['dance'] = df[df['Name'] == unit['name']]['Dance'].values[0]
                unit['health'] = df[df['Name'] == unit['name']]['HP'].values[0]
                unit['attack'] = df[df['Name'] == unit['name']]['Atk'].values[0]
        return units
    
    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            screen_width = self.grid_size * self.cell_size + 2 * self.padding
            screen_height = self.grid_size * self.cell_size + 2 * self.padding + 100  # Extra space for dance pattern info and enemy
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Arrowmancer")
        
        self.screen.fill((130, 139, 115)) # Fill the screen with a light green color

        # Grid rendering
        grid_top = self.padding + 100  # Add extra space above the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_rect = pygame.Rect(self.padding + j * self.cell_size, grid_top + i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (75, 68, 60), cell_rect)  # Draw cell background
                pygame.draw.rect(self.screen, (130, 139, 115), cell_rect, 2)  # Draw cell border                
                
                # Draw smaller pink square inside the cell
                pink_square_size = int(self.cell_size * 0.8)
                pink_square_rect = pygame.Rect(cell_rect.centerx - pink_square_size // 2, cell_rect.centery - pink_square_size // 2, pink_square_size, pink_square_size)
                pygame.draw.rect(self.screen, (130, 98, 107), pink_square_rect, 5)

                if self.enemy_attacks[i, j] > 0:
                    self._render_img("assets/purple.svg", cell_rect.centerx, cell_rect.centery)

                for k in range(self.num_units):
                    if (self.unit_positions[k] == [i, j]).all():
                        self._render_img(f"assets/standard_banners/images/{self.units[k]['name'].lower()}.png", cell_rect.centerx, cell_rect.centery)
                        health_bar_width = 50
                        health_bar_height = 10
                        health_bar_rect = pygame.Rect(cell_rect.centerx - health_bar_width // 2, cell_rect.centery + self.cell_size // 2 - health_bar_height - 5, health_bar_width, health_bar_height)
                        pygame.draw.rect(self.screen, (255, 0, 0), health_bar_rect)
                        health_bar_fill_rect = pygame.Rect(health_bar_rect.left, health_bar_rect.top, int(health_bar_width * self.unit_health[k]), health_bar_height)
                        pygame.draw.rect(self.screen, (0, 255, 0), health_bar_fill_rect)
                        break
        
        # Enemy rendering
        enemy_x = (self.grid_size * self.cell_size) // 2 + self.padding
        enemy_y = self.padding + 10 
        self._render_img("assets/nerd.svg", enemy_x, enemy_y)
        enemy_health_bar_width = 100
        enemy_health_bar_height = 10
        enemy_health_bar_rect = pygame.Rect(enemy_x - enemy_health_bar_width // 2, enemy_y + self.cell_size // 3 + 10, enemy_health_bar_width, enemy_health_bar_height)
        pygame.draw.rect(self.screen, (255, 0, 0), enemy_health_bar_rect)
        enemy_health_bar_fill_rect = pygame.Rect(enemy_health_bar_rect.left, enemy_health_bar_rect.top, int(enemy_health_bar_width * self.enemy_health), enemy_health_bar_height)
        pygame.draw.rect(self.screen, (0, 255, 0), enemy_health_bar_fill_rect)
        
        # Dance pattern info rendering
        unit = self.units[self.current_unit]
        dance_pattern = dance_patterns[unit['zodiac']][unit['dance']]
        font = pygame.font.Font(None, 32)
        move_texts = []
        for i, move in enumerate(dance_pattern):
            if i == self.current_move_index:
                move_texts.append(font.render(f"{move}", True, (0, 255, 255))) # Highlight the current move
            else:
                move_texts.append(font.render(f"{move}", True, (62, 80, 86)))

        zodiac_text = font.render(f"{unit['zodiac']} {unit['dance']} Dance Pattern: ", True, (62, 80, 86))
        zodiac_rect = zodiac_text.get_rect(x=self.padding, y=grid_top + self.grid_size * self.cell_size + 20) # Adjust the dance pattern info position
        self.screen.blit(zodiac_text, zodiac_rect)

        current_x = zodiac_rect.right + 5
        for move_text in move_texts:
            move_rect = move_text.get_rect(x=current_x, y=zodiac_rect.y)
            self.screen.blit(move_text, move_rect)
            current_x += move_text.get_width() + 5

        pygame.display.flip()

    # Helper function for image rendering
    def _render_img(self, file_path, x, y):
        svg_surface = pygame.image.load(file_path)
        svg_surface = pygame.transform.scale(svg_surface, (self.cell_size * 0.8, self.cell_size * 0.8))
        svg_rect = svg_surface.get_rect(center=(x, y))
        self.screen.blit(svg_surface, svg_rect)