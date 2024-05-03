import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dance import dance_patterns

class ArrowmancerEnv(gym.Env):
    def __init__(self, units):
        super(ArrowmancerEnv, self).__init__()
        self.grid_size = 3  # 3x3
        self.units = units 
        self.num_units = len(units)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'unit_positions': spaces.Box(low=0, high=self.grid_size - 1, shape=(self.num_units, 2), dtype=int),
            'enemy_attacks': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'current_unit': spaces.Discrete(self.num_units),
            'current_move_index': spaces.Discrete(6)
        })
        self.current_unit = 0  # Index of the current unit performing the dance
        self.current_move_index = 0  # Index of the current move in the dance pattern
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
        # Get the current unit
        current_unit_pos = self.unit_positions[self.current_unit]
        # Update the position of the current unit based on the action
        if action == 0:  # Move up
            new_pos = [current_unit_pos[0] - 1, current_unit_pos[1]]
        elif action == 1:  # Move right
            new_pos = [current_unit_pos[0], current_unit_pos[1] + 1]
        elif action == 2:  # Move down
            new_pos = [current_unit_pos[0] + 1, current_unit_pos[1]]
        elif action == 3:  # Move left
            new_pos = [current_unit_pos[0], current_unit_pos[1] - 1]

        # Check if the new position is valid and update the unit's position
        if self._is_valid_position(new_pos):
            self.unit_positions[self.current_unit] = new_pos

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

        # Generate enemy attacks targeting either one grid or a column of 3 grids with 50% probability
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
        done = False
        for pos in self.unit_positions:
            if self.enemy_attacks[pos[0], pos[1]] == 1:
                done = True
                break

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Update the grid representation with unit positions
        self.grid.fill(0)
        for pos in self.unit_positions:
            self.grid[pos[0], pos[1]] = 1
        # Return the current observation as a dictionary
        return {
            'grid': self.grid,
            'unit_positions': self.unit_positions,
            'enemy_attacks': self.enemy_attacks,
            'current_unit': self.current_unit,
            'current_move_index': self.current_move_index
        }

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
        grid_size = self.grid_size
        unit_positions = self.unit_positions
        enemy_attacks = self.enemy_attacks

        # Create a visual representation of the grid
        grid_str = ""
        for i in range(grid_size):
            row_str = ""
            for j in range(grid_size):
                cell_str = "."
                for k in range(self.num_units):
                    if (unit_positions[k] == [i, j]).all():
                        # if dancing unit use different emoji
                        if k == self.current_unit:
                            cell_str = "💃"
                        else:
                            cell_str = "🧙‍♀️"
                        break
                if enemy_attacks[i, j] == 1:
                    cell_str = "🟪"
                row_str += cell_str + " "
            grid_str += row_str + "\n"

        # Create a visual representation of the dance patterns
        unit = self.units[self.current_unit]
        dance_pattern_str = f"{unit['name']} {unit['level']} Dance Pattern: "
        dance_pattern = dance_patterns[unit['name']][unit['level']]
        for i, move in enumerate(dance_pattern):
            if i == self.current_move_index:
                dance_pattern_str += f"[{move}] "
            else:
                dance_pattern_str += f"{move} "

        print(grid_str)
        print(dance_pattern_str)