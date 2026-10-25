"""
Unit tests for the GridWorldEnv simulator.
"""

import numpy as np
import pytest

from config import EnvConfig
from environment.grid_world import GridWorldEnv


def test_environment_reset_and_state_dimension():
    """Verify reset returns valid observation with exact expected dimensions."""
    cfg = EnvConfig(grid_size=10, obs_radius=2)
    env = GridWorldEnv(config=cfg)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    # 2 pos + 2 goal + 2 rel + 1 manhattan + 1 euclidean + (2*2+1)^2 = 8 + 25 = 33
    assert obs.shape == (33,)
    assert env.state_dim == 33
    assert info["start_pos"] == (0, 0)
    assert info["goal_pos"] == (9, 9)


def test_action_space_and_movements():
    """Verify actions obey defined 4-action space and update coordinates correctly."""
    cfg = EnvConfig(grid_size=10, obs_radius=2, max_steps=50)
    env = GridWorldEnv(config=cfg)
    # Empty layout without obstacles for deterministic motion test
    empty_layout = np.zeros((10, 10), dtype=np.int8)
    env.reset(layout=empty_layout)

    # Action DOWN (1): row 0 -> 1
    obs, reward, terminated, truncated, info = env.step(GridWorldEnv.ACTION_DOWN)
    assert env.agent_pos == (1, 0)
    assert not terminated

    # Action RIGHT (3): col 0 -> 1
    obs, reward, terminated, truncated, info = env.step(GridWorldEnv.ACTION_RIGHT)
    assert env.agent_pos == (1, 1)
    assert not terminated


def test_boundary_handling_and_invalid_move():
    """Verify moving out of bounds is penalized and keeps agent in place."""
    cfg = EnvConfig(grid_size=10, obs_radius=2)
    env = GridWorldEnv(config=cfg)
    empty_layout = np.zeros((10, 10), dtype=np.int8)
    env.reset(layout=empty_layout)

    # Agent is at (0, 0). Move UP (0) hits top boundary wall
    obs, reward, terminated, truncated, info = env.step(GridWorldEnv.ACTION_UP)
    assert env.agent_pos == (0, 0)
    assert info["invalid_move"] is True
    assert reward <= cfg.reward_invalid_move
    assert not terminated


def test_obstacle_collision_detection():
    """Verify fatal collision with obstacle terminates episode immediately."""
    cfg = EnvConfig(grid_size=10, obs_radius=2)
    env = GridWorldEnv(config=cfg)
    layout = np.zeros((10, 10), dtype=np.int8)
    layout[1, 0] = 1  # Obstacle directly below start (0, 0)
    env.reset(layout=layout)

    obs, reward, terminated, truncated, info = env.step(GridWorldEnv.ACTION_DOWN)
    assert terminated is True
    assert info["collision"] is True
    assert reward == cfg.reward_collision


def test_goal_reaching_termination():
    """Verify reaching the target cell terminates episode and grants goal reward."""
    cfg = EnvConfig(grid_size=10, obs_radius=2)
    env = GridWorldEnv(config=cfg)
    empty_layout = np.zeros((10, 10), dtype=np.int8)
    env.reset(layout=empty_layout)
    # Manually place agent adjacent to goal (9, 8)
    env.agent_pos = (9, 8)
    env.trajectory = [(9, 8)]

    # Move RIGHT into goal (9, 9)
    obs, reward, terminated, truncated, info = env.step(GridWorldEnv.ACTION_RIGHT)
    assert env.agent_pos == (9, 9)
    assert terminated is True
    assert info["success"] is True
    assert reward == cfg.reward_goal


def test_procedural_layout_reachability():
    """Verify procedurally generated maps maintain a viable path from start to goal."""
    env = GridWorldEnv()
    for seed in [101, 102, 103, 104, 105]:
        layout = env.generate_layout(seed=seed)
        assert layout[0, 0] == 0  # Start is free
        assert layout[9, 9] == 0  # Goal is free
        assert env._bfs_path_exists(layout, (0, 0), (9, 9)) is True
