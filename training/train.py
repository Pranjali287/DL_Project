"""
Training loop for GRAIL-NAV DQN Autonomous Navigation.
Supports both Fixed-Map and Diverse-Map training regimens.
"""

from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from agent.dqn_agent import DQNAgent
from config import (
    DEFAULT_DQN_CONFIG,
    DEFAULT_ENV_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    MODELS_DIR,
    RESULTS_DIR,
    DQNConfig,
    EnvConfig,
    TrainConfig,
)
from environment.grid_world import GridWorldEnv


def set_seed(seed: int) -> None:
    """Set global seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_agent(
    scenario: str = "fixed",
    env_config: Optional[EnvConfig] = None,
    dqn_config: Optional[DQNConfig] = None,
    train_config: Optional[TrainConfig] = None,
    episodes_override: Optional[int] = None
) -> Dict[str, Union[float, str, Path]]:
    """
    Train a DQN navigation agent.

    Args:
        scenario: "fixed" or "diverse"
        env_config: Environment parameters
        dqn_config: DQN hyperparameters
        train_config: Training regimen parameters
        episodes_override: Optional episode count override for quick runs/tests

    Returns:
        dict: Training summary metrics and artifact paths.
    """
    assert scenario in ["fixed", "diverse"], f"Invalid scenario '{scenario}'. Must be 'fixed' or 'diverse'."

    env_cfg = env_config or DEFAULT_ENV_CONFIG
    dqn_cfg = dqn_config or DEFAULT_DQN_CONFIG
    train_cfg = train_config or DEFAULT_TRAIN_CONFIG
    num_episodes = episodes_override or train_cfg.episodes

    # Initialize environment and seed
    set_seed(train_cfg.fixed_train_seed)
    env = GridWorldEnv(config=env_cfg)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.config.action_dim,
        config=dqn_cfg,
        device=train_cfg.device
    )

    # Pre-generate fixed layout if running fixed scenario
    fixed_layout = None
    if scenario == "fixed":
        fixed_layout = env.generate_layout(seed=train_cfg.fixed_train_seed)

    print(f"\n=======================================================")
    print(f" [GRAIL-NAV] Starting Training - Scenario: {scenario.upper()}")
    print(f" Total Episodes: {num_episodes} | Grid Size: {env_cfg.grid_size}x{env_cfg.grid_size}")
    print(f" Observation Dim: {env.state_dim} | Buffer Capacity: {dqn_cfg.buffer_capacity}")
    print(f"=======================================================\n")

    history: List[Dict] = []
    start_time = time.time()

    # Track rolling statistics
    recent_rewards = []
    recent_successes = []

    for ep in range(1, num_episodes + 1):
        # Determine environment layout
        if scenario == "fixed":
            state, _ = env.reset(layout=fixed_layout)
        else:
            # Diverse scenario: distinct procedurally generated layout per episode
            ep_seed = train_cfg.diverse_train_seed_start + ep
            state, _ = env.reset(seed=ep_seed)

        ep_reward = 0.0
        ep_losses = []
        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated):
            action = agent.act(state, eval_mode=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition in replay memory
            done = terminated
            agent.remember(state, action, reward, next_state, done)

            # Optimize network parameters
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward
            step_count += 1

        # Decay exploration rate
        agent.decay_epsilon()

        # Target network synchronization
        if ep % dqn_cfg.target_update_episodes == 0:
            agent.update_target_network()

        # Record metrics
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        success = int(info.get("success", False))
        collision = int(info.get("collision", False))

        recent_rewards.append(ep_reward)
        recent_successes.append(success)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
            recent_successes.pop(0)

        record = {
            "episode": ep,
            "scenario": scenario,
            "reward": ep_reward,
            "steps": step_count,
            "success": success,
            "collision": collision,
            "loss": mean_loss,
            "epsilon": agent.epsilon,
            "rolling_reward_50": float(np.mean(recent_rewards)),
            "rolling_success_rate_50": float(np.mean(recent_successes)),
            "elapsed_seconds": round(time.time() - start_time, 2)
        }
        history.append(record)

        if ep % train_cfg.log_freq == 0 or ep == num_episodes:
            print(
                f"Ep {ep:4d}/{num_episodes} | "
                f"Reward: {ep_reward:7.2f} (Avg50: {record['rolling_reward_50']:6.2f}) | "
                f"SuccRate50: {record['rolling_success_rate_50']*100:5.1f}% | "
                f"Steps: {step_count:2d} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Time: {record['elapsed_seconds']:.1f}s"
            )

    total_training_time = round(time.time() - start_time, 2)

    # Save Checkpoint
    checkpoint_path = MODELS_DIR / f"dqn_{scenario}.pt"
    agent.save(checkpoint_path)
    print(f"\n[+] Saved model checkpoint to: {checkpoint_path}")

    # Save History CSV
    csv_path = RESULTS_DIR / f"training_{scenario}.csv"
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved training history metrics to: {csv_path}")

    summary = {
        "scenario": scenario,
        "episodes": num_episodes,
        "final_rolling_reward": float(np.mean(recent_rewards)),
        "final_rolling_success_rate": float(np.mean(recent_successes)),
        "total_training_time_seconds": total_training_time,
        "checkpoint_path": checkpoint_path,
        "csv_path": csv_path
    }
    return summary
