import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dance import dance_patterns
import pygame

class ArrowmancerEnv(gym.Env):
    def __init__(self, units):
        super(ArrowmancerEnv, self).__init__()
        self.grid_size = 3  # 3x3
        self.cell_size = 200  # 200x200 pixels
        self.screen = None
        self.padding = 50 # Padding around the grid
        self.units = units 
        self.num_units = len(units)
        self.unit_health = np.ones(self.num_units) # Health points for each unit
        self.current_unit = 0  # Index of the current unit performing the dance
        self.current_move_index = 0  # Index of the current move in the dance pattern

        self.action_space = spaces.Discrete(12)  # (Up, Right, Down, Left) x 3 Units
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'unit_positions': spaces.Box(low=0, high=self.grid_size - 1, shape=(self.num_units, 2), dtype=int),
            'enemy_attacks': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'current_unit': spaces.Discrete(self.num_units),
            'unit_health': spaces.Box(low=0, high=1, shape=(self.num_units,), dtype=float),
            'current_move_index': spaces.Discrete(6),
            'current_dance_pattern': spaces.Box(low=-13, high=13, shape=(6, 2), dtype=int)
        })
        self.reset()

    def reset(self):
        # Reset the grid to an empty state
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Randomly initialize unit positions on the grid
        self.unit_positions = np.random.randint(0, self.grid_size, size=(self.num_units, 2))
        # Reset enemy attacks to an empty state
        self.enemy_attacks = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_unit = 0  # Reset the current unit to the first unit
        self.current_move_index = 0  # Reset the current move index to the beginning
        return self._get_obs()

    def step(self, action):
        unit = action // 4 # Choose the current unit to move based on the action
        action = action % 4
        unit_pos = self.unit_positions[unit]
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
            if self._is_unit_at_position(new_pos):  # TODO: Method is broken for this use case!! Fix it
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
        reward = 0
        unit = self.units[self.current_unit]
        move = dance_patterns[unit['name']][unit['level']][self.current_move_index]
        if self._check_dance_move(move):
            reward = 1 + 0.1 * self.current_move_index # 10% increase for combos, TODO: reset if the player attacks the enemy
            self.current_move_index += 1
            # Move to the next unit if the current unit has completed all dance moves
            if self.current_move_index >= len(dance_patterns[unit['name']][unit['level']]):
                self.current_unit = (self.current_unit + 1) % self.num_units
                self.current_move_index = 0

        # Generate enemy attacks targeting either one grid or a column of 3 grids with 20% probability
        # TODO: This should be on a timer I believe and the units should have the ability to dodge
        self.enemy_attacks = np.zeros((self.grid_size, self.grid_size), dtype=int)
        attack = np.random.choice([0, 1], p=[0.8,0.2]) # 80% chance of no attack
        attack_type = np.random.choice(['single', 'column'], p=[0.8, 0.2]) # 80% chance of single grid attack
        if attack:
            if attack_type == 'single':
                # Attack a single random grid
                attack_pos = tuple(np.random.randint(0, self.grid_size, size=2))
                self.enemy_attacks[attack_pos] = 1
            else:
                # Attack a random column of 3 grids
                attack_col = np.random.randint(0, self.grid_size)
                self.enemy_attacks[:, attack_col] = 1

        # Check if units are hit by enemy attacks
        # TODO: Add health points for units and decrement health points when hit
        # TODO: I also need to create a win condition for the player where they have to defeat the enemy
        terminated = False
        truncated = False
        for i, pos in enumerate(self.unit_positions):
            if self.enemy_attacks[pos[0], pos[1]] == 1:
                self.unit_health[i] -= 0.1  # Reduce unit health by 10%
                if self.unit_health[i] <= 0:
                    terminated = True
                    break

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Update the grid representation with unit positions
        self.grid.fill(0)
        for pos in self.unit_positions:
            self.grid[pos[0], pos[1]] = 1
        unit = self.units[self.current_unit]
        current_dance_pattern = dance_patterns[unit['name']][unit['level']]
        # Change tuples to numpy arrays and pad the dance pattern with zeros
        current_dance_pattern = np.array([np.array(np.pad(move, (0, 2 - len(move)),'constant')) for move in current_dance_pattern])
        # Pad the dance pattern with zeros to have a fixed length of 6
        current_dance_pattern = np.pad(current_dance_pattern, ((0, 6 - len(current_dance_pattern)), (0, 0)), 'constant')
        return {
            'grid': self.grid,
            'unit_positions': self.unit_positions,
            'enemy_attacks': self.enemy_attacks,
            'current_unit': self.current_unit,
            'unit_health': self.unit_health,
            'current_move_index': self.current_move_index,
            'current_dance_pattern': current_dance_pattern
        }, 0 # dummy reward

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
        return self._is_valid_position(target_pos) and self._is_unit_at_position(target_pos)

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

    def _is_unit_at_position(self, pos):
        # Check if a unit (other than the current unit) is present at the given position
        for i, unit_pos in enumerate(self.unit_positions):
            if i != self.current_unit and (unit_pos == pos).all():
                return True
        return False

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            screen_width = self.grid_size * self.cell_size + 2 * self.padding
            screen_height = self.grid_size * self.cell_size + 2 * self.padding + 50 # Extra space for dance pattern info
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Arrowmancer")
        
        self.screen.fill((130, 139, 115)) # Fill the screen with a light green color

        # Grid rendering
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_rect = pygame.Rect(self.padding + j * self.cell_size, self.padding + i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (75, 68, 60), cell_rect)  # Draw cell background
                pygame.draw.rect(self.screen, (130, 139, 115), cell_rect, 2)  # Draw cell border                
                
                # Draw smaller pink square inside the cell
                pink_square_size = int(self.cell_size * 0.8)
                pink_square_rect = pygame.Rect(cell_rect.centerx - pink_square_size // 2, cell_rect.centery - pink_square_size // 2, pink_square_size, pink_square_size)
                pygame.draw.rect(self.screen, (130, 98, 107), pink_square_rect, 5)
                
                for k in range(self.num_units):
                    if (self.unit_positions[k] == [i, j]).all():
                        # If dancing unit use different emoji
                        if k == self.current_unit:
                            self._render_emoji("💃", cell_rect.centerx, cell_rect.centery)
                        else:
                            self._render_emoji("🧙‍♀️", cell_rect.centerx, cell_rect.centery)
                        health_bar_width = 50
                        health_bar_height = 10
                        health_bar_rect = pygame.Rect(cell_rect.centerx - health_bar_width // 2, cell_rect.centery + self.cell_size // 2 - health_bar_height - 5, health_bar_width, health_bar_height)
                        pygame.draw.rect(self.screen, (255, 0, 0), health_bar_rect)
                        health_bar_fill_rect = pygame.Rect(health_bar_rect.left, health_bar_rect.top, int(health_bar_width * self.unit_health[k]), health_bar_height)
                        pygame.draw.rect(self.screen, (0, 255, 0), health_bar_fill_rect)
                        break
                if self.enemy_attacks[i, j] == 1:
                    self._render_emoji("🟪", cell_rect.centerx, cell_rect.centery)
        
        # Dance pattern info rendering
        unit = self.units[self.current_unit]
        dance_pattern = dance_patterns[unit['name']][unit['level']]
        font = pygame.font.Font(None, 32)
        move_texts = []
        for i, move in enumerate(dance_pattern):
            if i == self.current_move_index:
                move_texts.append(font.render(f"{move}", True, (0, 255, 255))) # Highlight the current move
            else:
                move_texts.append(font.render(f"{move}", True, (62, 80, 86)))

        name_text = font.render(f"{unit['name']} {unit['level']} Dance Pattern: ", True, (62, 80, 86))
        name_rect = name_text.get_rect(x=self.padding, y=self.grid_size * self.cell_size + self.padding + 30)
        self.screen.blit(name_text, name_rect)

        current_x = name_rect.right + 5
        for move_text in move_texts:
            move_rect = move_text.get_rect(x=current_x, y=name_rect.y)
            self.screen.blit(move_text, move_rect)
            current_x += move_text.get_width() + 5

        pygame.display.flip()

    # Helper function for emoji rendering
    def _render_emoji(self, emoji, x, y):
        font = pygame.font.Font('AppleColorEmoji.ttf', self.cell_size)
        text = font.render(emoji, True, (0, 0, 0))
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)