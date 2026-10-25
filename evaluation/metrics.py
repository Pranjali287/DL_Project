"""
Evaluation metrics computation for GRAIL-NAV.
Defines formal denominators, statistical measures, and summary representations.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class MetricsResult:
    """Statistical summary of navigation evaluation performance."""
    total_episodes: int
    success_count: int
    collision_count: int
    timeout_count: int
    success_rate_pct: float
    collision_rate_pct: float
    timeout_rate_pct: float
    avg_reward: float
    std_reward: float
    avg_steps: float
    avg_successful_path_length: float
    avg_inference_latency_ms: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_navigation_metrics(episode_records: List[Dict], total_inference_time_sec: float) -> MetricsResult:
    """
    Compute rigorous evaluation metrics across test episodes.

    Definitions:
    - Success Rate: (Number of successful goal reaches / Total episodes) * 100
    - Collision Rate: (Number of fatal obstacle collisions / Total episodes) * 100
    - Timeout Rate: (Number of episodes reaching step limit without termination / Total episodes) * 100
    - Average Steps: Mean step count across ALL episodes
    - Average Successful Path Length: Mean step count ONLY across SUCCESSFUL episodes
      (Prevents failed fast crashes from artificially deflating reported path length)
    - Average Cumulative Reward: Mean total undiscounted reward per episode
    - Inference Latency: Mean execution time per policy decision (ms)
    """
    total = len(episode_records)
    if total == 0:
        return MetricsResult(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    success_count = sum(1 for ep in episode_records if ep["success"])
    collision_count = sum(1 for ep in episode_records if ep["collision"])
    timeout_count = sum(1 for ep in episode_records if not ep["success"] and not ep["collision"])

    rewards = [ep["reward"] for ep in episode_records]
    steps_all = [ep["steps"] for ep in episode_records]
    successful_steps = [ep["steps"] for ep in episode_records if ep["success"]]

    total_decision_steps = sum(steps_all)
    avg_latency_ms = (
        (total_inference_time_sec / total_decision_steps) * 1000.0
        if total_decision_steps > 0 else 0.0
    )

    avg_succ_len = float(np.mean(successful_steps)) if len(successful_steps) > 0 else float("nan")

    return MetricsResult(
        total_episodes=total,
        success_count=success_count,
        collision_count=collision_count,
        timeout_count=timeout_count,
        success_rate_pct=round((success_count / total) * 100.0, 2),
        collision_rate_pct=round((collision_count / total) * 100.0, 2),
        timeout_rate_pct=round((timeout_count / total) * 100.0, 2),
        avg_reward=round(float(np.mean(rewards)), 2),
        std_reward=round(float(np.std(rewards)), 2),
        avg_steps=round(float(np.mean(steps_all)), 2),
        avg_successful_path_length=round(avg_succ_len, 2) if not np.isnan(avg_succ_len) else -1.0,
        avg_inference_latency_ms=round(avg_latency_ms, 3)
    )
