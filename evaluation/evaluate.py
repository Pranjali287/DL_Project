"""
Evaluation engine for GRAIL-NAV.
Evaluates agents under deterministic greedy policy across familiar and unseen maps.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from agent.dqn_agent import DQNAgent
from config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_EVAL_CONFIG,
    MODELS_DIR,
    RESULTS_DIR,
    EnvConfig,
    EvalConfig,
)
from environment.grid_world import GridWorldEnv
from .metrics import MetricsResult, compute_navigation_metrics


def evaluate_agent(
    model_path: Path,
    test_type: str = "unseen",  # "familiar" or "unseen"
    num_episodes: int = 50,
    seed_start: int = 9000,
    familiar_layout_seed: int = 42,
    env_config: Optional[EnvConfig] = None,
    device: str = "cpu"
) -> Tuple[MetricsResult, List[Dict]]:
    """
    Evaluate a saved DQN policy deterministically.
    """
    env_cfg = env_config or DEFAULT_ENV_CONFIG
    env = GridWorldEnv(config=env_cfg)

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.config.action_dim,
        device=device
    )
    agent.load(model_path)

    # Fixed familiar layout if testing familiar mode for fixed agent
    familiar_layout = None
    if test_type == "familiar":
        familiar_layout = env.generate_layout(seed=familiar_layout_seed)

    episode_records = []
    total_inference_time = 0.0

    for i in range(num_episodes):
        if test_type == "familiar":
            state, _ = env.reset(layout=familiar_layout)
        else:
            # Held-out unseen layout
            map_seed = seed_start + i
            state, _ = env.reset(seed=map_seed)

        ep_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            t0 = time.perf_counter()
            # Strict greedy exploitation during evaluation (no epsilon exploration)
            action = agent.act(state, eval_mode=True)
            total_inference_time += (time.perf_counter() - t0)

            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1

        record = {
            "episode_idx": i + 1,
            "reward": round(ep_reward, 2),
            "steps": step_count,
            "success": bool(info["success"]),
            "collision": bool(info["collision"]),
            "trajectory": env.trajectory
        }
        episode_records.append(record)

    metrics = compute_navigation_metrics(episode_records, total_inference_time)
    return metrics, episode_records


def run_full_comparative_evaluation(
    num_episodes: int = 50,
    unseen_seed_start: int = 9000,
    env_config: Optional[EnvConfig] = None
) -> pd.DataFrame:
    """
    Run comprehensive 2x2 experimental matrix:
    [Fixed Agent, Diverse Agent] x [Familiar Environment, Held-out Unseen Environments]
    """
    fixed_model = MODELS_DIR / "dqn_fixed.pt"
    diverse_model = MODELS_DIR / "dqn_diverse.pt"

    assert fixed_model.exists(), f"Fixed model missing at {fixed_model}. Please train fixed agent first."
    assert diverse_model.exists(), f"Diverse model missing at {diverse_model}. Please train diverse agent first."

    print("\n=======================================================")
    print(" [GRAIL-NAV] Executing Full Comparative Research Protocol")
    print(f" Evaluation Episodes per Condition: {num_episodes}")
    print(f" Held-Out Unseen Seeds: {unseen_seed_start} - {unseen_seed_start + num_episodes - 1}")
    print("=======================================================\n")

    experiments = [
        {"agent": "Fixed-Map Agent", "model": fixed_model, "env_type": "Familiar Map", "test_type": "familiar"},
        {"agent": "Fixed-Map Agent", "model": fixed_model, "env_type": "Unseen Maps", "test_type": "unseen"},
        {"agent": "Diverse-Map Agent", "model": diverse_model, "env_type": "Familiar Map", "test_type": "familiar"},
        {"agent": "Diverse-Map Agent", "model": diverse_model, "env_type": "Unseen Maps", "test_type": "unseen"},
    ]

    results_list = []
    for exp in experiments:
        print(f"-> Evaluating {exp['agent']} on {exp['env_type']}...")
        metrics, _ = evaluate_agent(
            model_path=exp["model"],
            test_type=exp["test_type"],
            num_episodes=num_episodes,
            seed_start=unseen_seed_start,
            env_config=env_config
        )
        res = {
            "Agent Policy": exp["agent"],
            "Environment Condition": exp["env_type"],
            "Success Rate (%)": metrics.success_rate_pct,
            "Collision Rate (%)": metrics.collision_rate_pct,
            "Timeout Rate (%)": metrics.timeout_rate_pct,
            "Avg Reward": metrics.avg_reward,
            "Std Reward": metrics.std_reward,
            "Avg Steps": metrics.avg_steps,
            "Avg Succ Path Length": metrics.avg_successful_path_length,
            "Inference Latency (ms)": metrics.avg_inference_latency_ms
        }
        results_list.append(res)

    df_results = pd.DataFrame(results_list)
    out_csv = RESULTS_DIR / "evaluation_summary.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Comparative evaluation complete. Results written to: {out_csv}")
    print("\n" + df_results.to_string(index=False) + "\n")
    return df_results
