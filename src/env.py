import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dance import dance_patterns

class ArrowmancerEnv(gym.Env):
    def __init__(self, units):
        super(ArrowmancerEnv, self).__init__()
        self.grid_size = 3
        self.units = units
        self.num_units = len(units)
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size] * self.num_units)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'unit_positions': spaces.Box(low=0, high=self.grid_size - 1, shape=(self.num_units, 2), dtype=int),
            'enemy_attacks': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=int),
            'dance_patterns': spaces.Box(low=0, high=1, shape=(self.num_units,), dtype=int),
            'current_unit': spaces.Discrete(self.num_units),
            'current_move_index': spaces.Discrete(6)
        })
        self.current_unit = 0
        self.current_move_index = 0
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.unit_positions = np.random.randint(0, self.grid_size, size=(self.num_units, 2))
        self.enemy_attacks = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.dance_patterns = np.zeros((self.num_units,), dtype=int)
        self.current_unit = 0
        self.current_move_index = 0
        return self._get_obs()

    def step(self, action):
        # Update unit positions based on the action
        for i in range(self.num_units):
            self.unit_positions[i] = action[i * 2: i * 2 + 2]

        # Check if the current unit's dance move is satisfied
        reward = 0
        unit = self.units[self.current_unit]
        if unit in dance_patterns:
            pattern_key = list(dance_patterns[unit].keys())[self.current_move_index // len(dance_patterns[unit]['6'])]
            move_index = self.current_move_index % len(dance_patterns[unit]['6'])
            move = dance_patterns[unit][pattern_key][move_index]
            if self._check_dance_move(move):
                reward = 1
                self.current_move_index += 1
                if self.current_move_index >= len(dance_patterns[unit][pattern_key]):
                    self.current_unit = (self.current_unit + 1) % self.num_units
                    self.current_move_index = 0

        # Generate random enemy attacks
        self.enemy_attacks = np.random.randint(0, 2, size=(self.grid_size, self.grid_size))

        # Check if units are hit by enemy attacks
        done = False
        for pos in self.unit_positions:
            if self.enemy_attacks[pos[0], pos[1]] == 1:
                done = True
                break

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        self.grid.fill(0)
        for pos in self.unit_positions:
            self.grid[pos[0], pos[1]] = 1
        return {
            'grid': self.grid,
            'unit_positions': self.unit_positions,
            'enemy_attacks': self.enemy_attacks,
            'dance_patterns': self.dance_patterns,
            'current_unit': self.current_unit,
            'current_move_index': self.current_move_index
        }

    def _check_dance_move(self, move):
        if isinstance(move, tuple):
            for offset in move:
                if not self._check_non_anchor_move(offset):
                    return False
        else:
            return self._check_anchor_move(move)
        return True

    def _check_non_anchor_move(self, offset):
        current_pos = self.unit_positions[self.current_unit]
        target_pos = current_pos + self._offset_to_position(offset)
        return self._is_valid_position(target_pos) and self._is_unit_at_position(target_pos)

    def _check_anchor_move(self, offset):
        current_pos = self.unit_positions[self.current_unit]
        if offset == -13:
            return current_pos[0] == 0
        elif offset == -11:
            return current_pos[0] == 0 and current_pos[1] == 0
        elif offset == 11:
            return current_pos[0] == 0 and current_pos[1] == self.grid_size - 1
        elif offset == 13:
            return current_pos[0] == self.grid_size - 1
        return False

    def _offset_to_position(self, offset):
        return [offset // 3 - 1, offset % 3 - 1]

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _is_unit_at_position(self, pos):
        for i, unit_pos in enumerate(self.unit_positions):
            if i != self.current_unit and (unit_pos == pos).all():
                return True
        return False

    def render(self, mode='human'):
        grid_size = self.grid_size
        unit_positions = self.unit_positions
        enemy_attacks = self.enemy_attacks
        dance_patterns = self.dance_patterns

        # Create a visual representation of the grid
        grid_str = ""
        for i in range(grid_size):
            row_str = ""
            for j in range(grid_size):
                cell_str = "."
                for k in range(self.num_units):
                    if (unit_positions[k] == [i, j]).all():
                        cell_str = str(k + 1)
                        break
                if enemy_attacks[i, j] == 1:
                    cell_str = "X"
                row_str += cell_str + " "
            grid_str += row_str + "\n"

        # Create a visual representation of the dance patterns
        dance_pattern_str = "Dance Patterns: "
        for i in range(self.num_units):
            if dance_patterns[i] == 1:
                dance_pattern_str += f"Unit {i + 1}: Dance "
            else:
                dance_pattern_str += f"Unit {i + 1}: No Dance "

        # Print the grid and dance patterns
        print(grid_str)
        print(dance_pattern_str)
