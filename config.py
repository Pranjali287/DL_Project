"""
Configuration settings for GRAIL-NAV:
Generalization-Aware Deep Reinforcement Learning for Autonomous Robot Navigation in Unseen Obstacle Environments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EnvConfig:
    """Environment configuration parameters."""
    grid_size: int = 10
    start_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (9, 9)
    obstacle_density: float = 0.18  # ~18% obstacle cells
    obs_radius: int = 2  # (2*radius + 1)^2 local grid = 5x5 = 25 cells
    max_steps: int = 60  # Truncate episode if exceeded
    
    # Reward definitions
    reward_goal: float = 100.0
    reward_collision: float = -100.0
    reward_invalid_move: float = -10.0  # Hitting boundary wall
    reward_step: float = -1.0           # Time penalty per step
    distance_reward_factor: float = 1.5 # Potential-based progress: factor * (prev_dist - curr_dist)

    # Action mappings
    # 0: Up (-1, 0), 1: Down (+1, 0), 2: Left (0, -1), 3: Right (0, +1)
    action_dim: int = 4


@dataclass
class DQNConfig:
    """DQN Agent and Neural Network hyperparameters."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 0.001
    gamma: float = 0.99
    buffer_capacity: int = 20000
    batch_size: int = 64
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.996  # Gradual exponential decay
    
    # Target Network update
    target_update_episodes: int = 10  # Synchronize target network every N episodes
    grad_clip_norm: float = 10.0


@dataclass
class TrainConfig:
    """Experiment and training loop parameters."""
    episodes: int = 1200
    fixed_train_seed: int = 42
    diverse_train_seed_start: int = 1000
    eval_freq: int = 100
    log_freq: int = 50
    device: str = "cpu"


@dataclass
class EvalConfig:
    """Evaluation protocol parameters."""
    num_test_episodes: int = 50
    # Seed ranges explicitly isolated from training to prevent test leakage
    unseen_seed_start: int = 9000
    device: str = "cpu"


# Default instances
DEFAULT_ENV_CONFIG = EnvConfig()
DEFAULT_DQN_CONFIG = DQNConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()
