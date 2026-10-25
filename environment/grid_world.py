"""
Grid World Navigation Environment for GRAIL-NAV.
A 10x10 discrete simulator featuring static obstacles, procedural layout generation,
collision physics, potential-based shaping, and local occupancy field observations.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from config import EnvConfig, DEFAULT_ENV_CONFIG


class GridWorldEnv:
    """
    Gymnasium-compatible 2D Grid World navigation environment.
    """
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTION_DELTAS = {
        0: (-1, 0),  # Up (decrease row)
        1: (1, 0),   # Down (increase row)
        2: (0, -1),  # Left (decrease col)
        3: (0, 1),   # Right (increase col)
    }

    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or DEFAULT_ENV_CONFIG
        self.grid_size = self.config.grid_size
        self.start_pos = tuple(self.config.start_pos)
        self.goal_pos = tuple(self.config.goal_pos)
        self.obs_radius = self.config.obs_radius
        self.max_steps = self.config.max_steps

        # State tracking
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.grid: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.step_count: int = 0
        self.trajectory: List[Tuple[int, int]] = []

        # Calculate state observation dimension
        # (agent_x, agent_y) normalized: 2
        # (goal_x, goal_y) normalized: 2
        # (dx, dy) relative vector normalized: 2
        # normalized Manhattan distance: 1
        # normalized Euclidean distance: 1
        # local occupancy grid window: (2*obs_radius + 1)^2
        window_size = (2 * self.obs_radius + 1) ** 2
        self.state_dim = 2 + 2 + 2 + 1 + 1 + window_size

    def _is_valid_pos(self, r: int, c: int) -> bool:
        """Check if coordinates lie strictly within grid boundaries."""
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _bfs_path_exists(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Verify reachability from start to goal in given layout using Breadth-First Search."""
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            return False

        visited = set([start])
        queue = deque([start])

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True

            for dr, dc in self.ACTION_DELTAS.values():
                nr, nc = curr[0] + dr, curr[1] + dc
                if self._is_valid_pos(nr, nc) and grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def generate_layout(self, seed: Optional[int] = None, max_attempts: int = 50) -> np.ndarray:
        """
        Procedurally generate an obstacle layout guaranteed to have a viable path.
        """
        rng = np.random.RandomState(seed)
        num_cells = self.grid_size * self.grid_size
        num_obstacles = int(num_cells * self.config.obstacle_density)

        for _ in range(max_attempts):
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            all_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                         if (r, c) != self.start_pos and (r, c) != self.goal_pos]

            chosen_indices = rng.choice(len(all_cells), size=num_obstacles, replace=False)
            for idx in chosen_indices:
                r, c = all_cells[idx]
                grid[r, c] = 1

            if self._bfs_path_exists(grid, self.start_pos, self.goal_pos):
                return grid

        # Fallback to an empty grid with perimeter clear path if seed is unusually dense
        fallback_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        return fallback_grid

    def _get_local_occupancy(self) -> np.ndarray:
        """
        Extract flattened local occupancy window around the agent.
        Values:
          - Out of bounds or obstacle: 1.0
          - Goal: -1.0
          - Free space: 0.0
        """
        r_agent, c_agent = self.agent_pos
        local_window = []

        for dr in range(-self.obs_radius, self.obs_radius + 1):
            for dc in range(-self.obs_radius, self.obs_radius + 1):
                nr, nc = r_agent + dr, c_agent + dc
                if not self._is_valid_pos(nr, nc):
                    local_window.append(1.0)  # Boundary acts as an impenetrable wall
                elif self.grid[nr, nc] == 1:
                    local_window.append(1.0)  # Obstacle
                elif (nr, nc) == self.goal_pos:
                    local_window.append(-1.0)  # Goal signature
                else:
                    local_window.append(0.0)  # Free navigable space

        return np.array(local_window, dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        """
        Construct a normalized state representation vector.
        """
        norm_factor = float(self.grid_size - 1)
        r_a, c_a = self.agent_pos
        r_g, c_g = self.goal_pos

        # Normalized coordinates [0, 1]
        agent_norm = [r_a / norm_factor, c_a / norm_factor]
        goal_norm = [r_g / norm_factor, c_g / norm_factor]

        # Normalized relative displacement [-1, 1]
        rel_dr = (r_g - r_a) / norm_factor
        rel_dc = (c_g - c_a) / norm_factor
        rel_norm = [rel_dr, rel_dc]

        # Normalized distance metrics
        manhattan_dist = (abs(r_g - r_a) + abs(c_g - c_a)) / (2.0 * norm_factor)
        euclidean_dist = np.sqrt((r_g - r_a)**2 + (c_g - c_a)**2) / (np.sqrt(2.0) * norm_factor)

        # Local barrier sensing
        local_occ = self._get_local_occupancy()

        obs = np.concatenate([
            np.array(agent_norm, dtype=np.float32),
            np.array(goal_norm, dtype=np.float32),
            np.array(rel_norm, dtype=np.float32),
            np.array([manhattan_dist, euclidean_dist], dtype=np.float32),
            local_occ
        ], dtype=np.float32)

        return obs

    def reset(self, seed: Optional[int] = None, layout: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment state for a new episode.
        """
        self.step_count = 0
        self.agent_pos = self.start_pos
        self.trajectory = [self.agent_pos]

        if layout is not None:
            self.grid = layout.copy()
        else:
            self.grid = self.generate_layout(seed=seed)

        obs = self.get_observation()
        info = {
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "seed": seed
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment dynamics.
        
        Returns:
            observation (np.ndarray): Next state vector.
            reward (float): Immediate reward signal.
            terminated (bool): True if goal reached or collision occurred.
            truncated (bool): True if step count exceeds max_steps.
            info (dict): Auxiliary diagnostic metadata.
        """
        self.step_count += 1
        prev_pos = self.agent_pos
        prev_manhattan = abs(self.goal_pos[0] - prev_pos[0]) + abs(self.goal_pos[1] - prev_pos[1])

        dr, dc = self.ACTION_DELTAS.get(action, (0, 0))
        target_r, target_c = prev_pos[0] + dr, prev_pos[1] + dc

        reward = 0.0
        terminated = False
        truncated = False
        collision = False
        success = False
        invalid_move = False

        # 1. Check boundary wall collision
        if not self._is_valid_pos(target_r, target_c):
            # Invalid movement: agent hits outer boundary wall and stays in place
            invalid_move = True
            reward = self.config.reward_invalid_move + self.config.reward_step
            # Agent stays at prev_pos
            curr_pos = prev_pos
        # 2. Check obstacle collision
        elif self.grid[target_r, target_c] == 1:
            # Obstacle collision: immediate fatal crash, episode terminates
            collision = True
            terminated = True
            reward = self.config.reward_collision
            curr_pos = (target_r, target_c)
            self.agent_pos = curr_pos
            self.trajectory.append(curr_pos)
        # 3. Check goal reached
        elif (target_r, target_c) == self.goal_pos:
            success = True
            terminated = True
            reward = self.config.reward_goal
            curr_pos = (target_r, target_c)
            self.agent_pos = curr_pos
            self.trajectory.append(curr_pos)
        # 4. Valid movement into empty navigable cell
        else:
            curr_pos = (target_r, target_c)
            self.agent_pos = curr_pos
            self.trajectory.append(curr_pos)

            curr_manhattan = abs(self.goal_pos[0] - curr_pos[0]) + abs(self.goal_pos[1] - curr_pos[1])
            # Potential-based shaping: delta distance (strictly telescoping, prevents cycling exploits)
            distance_delta = float(prev_manhattan - curr_manhattan)
            progress_reward = self.config.distance_reward_factor * distance_delta
            reward = self.config.reward_step + progress_reward

        # Episode step truncation
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        obs = self.get_observation()
        info = {
            "success": success,
            "collision": collision,
            "invalid_move": invalid_move,
            "step_count": self.step_count,
            "agent_pos": self.agent_pos,
            "path_length": len(self.trajectory) - 1
        }

        return obs, float(reward), terminated, truncated, info
