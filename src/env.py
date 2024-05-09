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
        self.time_step = 0
        self.enemy_attack_delay = 3  # Number of time steps before enemy attack activates
        self.units = self._get_units(units) 
        self.num_units = len(units)
        self.unit_health = np.array([unit['health'] for unit in self.units])
        self.unit_attack = np.array([unit['attack'] for unit in self.units])
        self.enemy_health = 750
        self.current_unit = 0  # Index of the current unit performing the dance
        self.current_move_index = 0  # Index of the current move in the dance pattern

        self.action_space = spaces.Discrete(15)  # (Up, Right, Down, Left, Attack) x 3 Units
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=-3, high=3, shape=(self.grid_size, self.grid_size, 2), dtype=int), # Grid with unit and enemy attack positions
            'time_step': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=int),
            'unit_health': spaces.Box(low=0, high=1, shape=(self.num_units,), dtype=float),
            # 'unit_attack': spaces.Box(low=0, high=1, shape=(self.num_units,), dtype=float),
            'enemy_health': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'current_unit': spaces.Box(low=0, high=1, shape=(self.num_units,), dtype=int),
            'current_move_index': spaces.Box(low=0, high=1, shape=(6,), dtype=int),
            'current_dance_pattern': spaces.Box(low=-13, high=13, shape=(6, 2), dtype=int)
        })
        self.reset()

    def reset(self):
        self.time_step = 0
        # Reset the grid to an empty state
        self.grid = np.zeros((self.grid_size, self.grid_size, 2), dtype=int)
        # Randomly initialize unit positions on the grid
        positions = np.array(list(np.ndindex(self.grid_size, self.grid_size)))
        self.unit_positions = positions[np.random.choice(len(positions), size=self.num_units, replace=False)]
        # Reset health points
        self.unit_health = np.array([unit['health'] for unit in self.units])
        self.enemy_health = 750
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
                reward += 0.5 * (self.current_move_index + 1)
                self.current_move_index += 1
                # Move to the next unit if the current unit has completed all dance moves
                if self.current_move_index >= len(dance_patterns[unit['zodiac']][unit['dance']]):
                    reward += 5  # Bonus for completing the dance pattern
                    self.current_unit = (self.current_unit + 1) % self.num_units
                    self.current_move_index = 0

        else:  # Attack action
            if unit_pos[0] == 0 and unit_pos[1] == 1:  # Check if the unit is adjacent to the enemy
                attack_strength = self.unit_attack[unit] * 0.01 * (1 + 0.75 * self.current_move_index) # Scale by 100 and add 75% bonus for each combo move completed
                reward += attack_strength / 2
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
        attack = np.random.choice([0, 1], p=[0.65, 0.35])  # 65% chance of no attack
        attack_type = np.random.choice(['single', 'column'], p=[0.75, 0.25])  # 75% chance of single grid attack
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
                self.unit_health[i] -= 100 # Reduce unit health
                reward -= 5  # Penalty for unit hit
                self.enemy_attacks[pos[0], pos[1]] = 0 # Reset the enemy attack
                if self.unit_health[i] <= 0:
                    terminated = True
                    reward = -100  # Negative reward for unit defeat
                    break

        return self._get_obs()[0], reward, terminated, truncated, {}

    def _get_obs(self):
        # Update the grid representation with unit and enemy attack positions
        self.grid.fill(0)
        for i, pos in enumerate(self.unit_positions):
            self.grid[pos[0], pos[1], 0] = i + 1
        for i, row in enumerate(self.enemy_attacks):
            for j, attack in enumerate(row):
                self.grid[i, j, 1] = -attack

        unit = self.units[self.current_unit]
        current_dance_pattern = dance_patterns[unit['zodiac']][unit['dance']]
        # Change tuples to numpy arrays and pad the dance pattern with zeros
        current_dance_pattern = np.array([np.array(np.pad(move, (0, 2 - len(move)),'constant')) for move in current_dance_pattern])
        # Pad the dance pattern with zeros to have a fixed length of 6
        current_dance_pattern = np.pad(current_dance_pattern, ((0, 6 - len(current_dance_pattern)), (0, 0)), 'constant')
        return {
            'grid': self.grid,
            'time_step': np.array([self.time_step]), # Convert scalar to numpy array for consistency
            'unit_health': self.unit_health / max(self.unit_health), # Normalize health values
            'enemy_health': np.array([self.enemy_health / 750]),
            'current_unit': np.eye(self.num_units)[self.current_unit], # One-hot encoding of the current unit index
            'current_move_index': np.eye(6)[self.current_move_index],
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
        return [(offset + 1) // 3, (offset + 1) % 3 - 1]

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
            if unit['name'] not in df['Name'].values or unit['name'] in ['Curry XIII', 'Scarletti', 'Saika', 'Linnaeus', 'Clover', 'Marilyn']:
                raise ValueError(f"Unit {unit['name']} not found in {unit['banner']} banners. Currently not supported: Curry XIII, Scarletti, Saika, Linnaeus, Clover, Marilyn")
            else:
                unit['zodiac'] = df[df['Name'] == unit['name']]['Zodiac'].values[0]
                unit['dance'] = df[df['Name'] == unit['name']]['Dance'].values[0]
                unit['health'] = df[df['Name'] == unit['name']]['HP'].values[0]
                unit['attack'] = df[df['Name'] == unit['name']]['Atk'].values[0]
        return units
    
    def render(self, mode='human'):
        padding = 75 # Padding around the grid
        screen_width = self.grid_size * self.cell_size + 2 * padding
        screen_height = self.grid_size * self.cell_size + 2 * padding + 200  # Extra space for dance pattern info and enemy
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Arrowmancer")
        
        self.screen.fill((130, 139, 115)) # Fill the screen with a light green color

        # Grid rendering
        grid_top = padding + 100  # Add extra space above the grid 
        grid_bottom = grid_top + self.grid_size * self.cell_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_rect = pygame.Rect(padding + j * self.cell_size, grid_top + i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (75, 68, 60), cell_rect)  # Draw cell background
                pygame.draw.rect(self.screen, (130, 139, 115), cell_rect, 2)  # Draw cell border                
                
                # Draw smaller pink square inside the cell
                pink_square_size = int(self.cell_size * 0.8)
                pink_square_rect = pygame.Rect(cell_rect.centerx - pink_square_size // 2, cell_rect.centery - pink_square_size // 2, pink_square_size, pink_square_size)
                pygame.draw.rect(self.screen, (130, 98, 107), pink_square_rect, 5)

                if self.enemy_attacks[i, j] > 0:
                    self._render_img("assets/emojis/purple.svg", cell_rect.centerx, cell_rect.centery, (self.cell_size, self.cell_size))

                for k in range(self.num_units):
                    if (self.unit_positions[k] == [i, j]).all():
                        self._render_img(f"assets/standard_banners/images/{self.units[k]['name'].lower()}.png", cell_rect.centerx, cell_rect.centery, (self.cell_size * 0.8, self.cell_size * 0.8))
                        health_bar_width = 50
                        health_bar_height = 10
                        health_bar_rect = pygame.Rect(cell_rect.centerx - health_bar_width // 2, cell_rect.centery + self.cell_size // 2 - health_bar_height - 5, health_bar_width, health_bar_height)
                        pygame.draw.rect(self.screen, (255, 0, 0), health_bar_rect)
                        health_bar_fill_rect = pygame.Rect(health_bar_rect.left, health_bar_rect.top, int(health_bar_width * (self.unit_health[k] / self.units[k]['health'])), health_bar_height)
                        pygame.draw.rect(self.screen, (0, 255, 0), health_bar_fill_rect)
                        break
        
        # Enemy rendering
        enemy_x = (self.grid_size * self.cell_size) // 2 + padding
        enemy_y = padding + 10 
        self._render_img("assets/cat.png", enemy_x, enemy_y, (self.cell_size * 0.7, self.cell_size * 0.7))
        enemy_health_bar_width = 100
        enemy_health_bar_height = 10
        enemy_health_bar_rect = pygame.Rect(enemy_x - enemy_health_bar_width // 2, enemy_y + self.cell_size // 3 + 10, enemy_health_bar_width, enemy_health_bar_height)
        pygame.draw.rect(self.screen, (255, 0, 0), enemy_health_bar_rect)
        enemy_health_bar_fill_rect = pygame.Rect(enemy_health_bar_rect.left, enemy_health_bar_rect.top, int(enemy_health_bar_width * (self.enemy_health / 750)), enemy_health_bar_height)
        pygame.draw.rect(self.screen, (0, 255, 0), enemy_health_bar_fill_rect)

        # Combo and dance pattern box rendering
        unit = self.units[self.current_unit]
        font = pygame.font.Font(None, 40)
        combo_box_width = self.grid_size * self.cell_size
        combo_box_height = screen_height - grid_bottom - padding
        combo_box_x = padding
        combo_box_y = grid_bottom + padding  # Adjust the position below the grid with padding
        combo_box_rect = pygame.Rect(combo_box_x, combo_box_y, combo_box_width, combo_box_height)
        pygame.draw.rect(self.screen, (86,114,125), combo_box_rect)

        # TODO: use image rendering helper function
        unit_image = pygame.image.load(f"assets/standard_banners/images/{unit['name'].lower()}.png")
        unit_image = pygame.transform.scale(unit_image, (combo_box_height, combo_box_height))
        unit_rect = unit_image.get_rect(left=combo_box_rect.left, centery=combo_box_rect.centery)
        self.screen.blit(unit_image, unit_rect)

        # Draw the pink border on three sides (top, left, right)
        pygame.draw.line(self.screen, (242,149,245), (combo_box_rect.left, combo_box_rect.top), (combo_box_rect.right, combo_box_rect.top), 5)
        pygame.draw.line(self.screen, (242,149,245), (combo_box_rect.left, combo_box_rect.top), (combo_box_rect.left, combo_box_rect.bottom), 5)
        pygame.draw.line(self.screen, (242,149,245), (combo_box_rect.right, combo_box_rect.top), (combo_box_rect.right, combo_box_rect.bottom), 5)

        combo_text = font.render(f"COMBO {self.current_move_index}", True, (149,185,148))
        combo_text_rect = combo_text.get_rect(left=unit_rect.right + 20, centery=combo_box_rect.centery)      
        self.screen.blit(combo_text, combo_text_rect)

        # Dance pattern info rendering
        font = pygame.font.Font(None, 28)
        dance_pattern = dance_patterns[unit['zodiac']][unit['dance']]

        zodiac_text = font.render(f"{unit['zodiac']} {unit['dance']} Dance Pattern: ", True, (149,185,148))
        zodiac_rect = zodiac_text.get_rect(left=unit_rect.right + 20, top=combo_text_rect.bottom + 10)
        self.screen.blit(zodiac_text, zodiac_rect)

        current_x = zodiac_rect.right + 12
        for i, move in enumerate(dance_pattern):
            for num in move: 
                if i == self.current_move_index:
                    move_rect = self._render_img(f"assets/emojis/{num}.svg", current_x, zodiac_rect.y+8, (26, 26), (56,239,195))
                else:
                    move_rect = self._render_img(f"assets/emojis/{num}.svg", current_x, zodiac_rect.y+8, (26, 26), (149,185,148))
                current_x += move_rect.width


        pygame.display.flip()

    # Helper function for image rendering
    def _render_img(self, file_path, x, y, size, color=None):
        img_surface = pygame.image.load(file_path).convert_alpha()
        if color:
            img_surface.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_MAX) # Replace every visible pixel with white
            img_surface.fill(color, special_flags=pygame.BLEND_RGBA_MIN)
        img_surface = pygame.transform.scale(img_surface, size)
        img_rect = img_surface.get_rect(center=(x, y))
        self.screen.blit(img_surface, img_rect)
        return img_rect