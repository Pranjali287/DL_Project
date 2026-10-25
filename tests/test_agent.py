"""
Unit tests for DQN Agent, Replay Buffer, Network, and Evaluation Metrics.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from agent.dqn_agent import DQNAgent
from agent.dqn_network import QNetwork
from agent.replay_buffer import ReplayBuffer
from config import DQNConfig, TrainConfig, EvalConfig
from evaluation.metrics import compute_navigation_metrics


def test_replay_buffer_push_and_sample():
    """Verify replay buffer correctly retains and samples transitions."""
    buffer = ReplayBuffer(capacity=100)
    state = np.zeros(33, dtype=np.float32)
    next_state = np.ones(33, dtype=np.float32)

    for i in range(10):
        buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=(i % 5 == 0))

    assert len(buffer) == 10
    batch_size = 4
    s, a, r, s_prime, d = buffer.sample(batch_size)

    assert s.shape == (batch_size, 33)
    assert a.shape == (batch_size, 1)
    assert r.shape == (batch_size, 1)
    assert s_prime.shape == (batch_size, 33)
    assert d.shape == (batch_size, 1)


def test_q_network_dimensions():
    """Verify Q-network outputs exactly 4 action values."""
    net = QNetwork(state_dim=33, action_dim=4, hidden_dims=[64, 64])
    dummy_input = torch.randn(8, 33)
    q_vals = net(dummy_input)
    assert q_vals.shape == (8, 4)


def test_dqn_act_and_eval_mode():
    """Verify greedy action selection under eval_mode."""
    agent = DQNAgent(state_dim=33, action_dim=4)
    dummy_state = np.zeros(33, dtype=np.float32)

    # Test actions are within [0, 3]
    for _ in range(20):
        action = agent.act(dummy_state, eval_mode=True)
        assert action in [0, 1, 2, 3]


def test_bellman_target_terminal_mask():
    """
    Verify Bellman target calculation zeroes out next-state value when transition is terminal.
    Target = r + gamma * max_a' Q(s', a') * (1 - done)
    When done = 1.0, Target MUST equal r exactly.
    """
    agent = DQNAgent(state_dim=4, action_dim=2)
    # Populate buffer with known terminal transition
    s = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    s_prime = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    reward = 100.0
    agent.remember(s, action=1, reward=reward, next_state=s_prime, done=True)

    states, actions, rewards, next_states, dones = agent.memory.sample(1)

    with torch.no_grad():
        max_next_q = agent.target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + (agent.gamma * max_next_q * (1.0 - dones))

    # Expect target_q to be exactly the reward because done=True
    assert torch.isclose(target_q, torch.tensor([[100.0]])).item()


def test_model_save_and_load(tmp_path: Path):
    """Verify agent weights are preserved identically through save and reload."""
    agent1 = DQNAgent(state_dim=16, action_dim=4)
    save_file = tmp_path / "test_model.pt"
    agent1.save(save_file)

    agent2 = DQNAgent(state_dim=16, action_dim=4)
    agent2.load(save_file)

    for p1, p2 in zip(agent1.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)


def test_metrics_calculation():
    """Verify evaluation metrics computation and edge-case handling."""
    dummy_records = [
        {"success": True, "collision": False, "reward": 85.0, "steps": 18},
        {"success": False, "collision": True, "reward": -105.0, "steps": 5},
        {"success": True, "collision": False, "reward": 92.0, "steps": 22},
    ]
    metrics = compute_navigation_metrics(dummy_records, total_inference_time_sec=0.03)

    assert metrics.total_episodes == 3
    assert metrics.success_count == 2
    assert metrics.collision_count == 1
    assert metrics.success_rate_pct == pytest.approx(66.67, 0.05)
    assert metrics.collision_rate_pct == pytest.approx(33.33, 0.05)
    # Average successful path length only over the 2 successful episodes: (18 + 22) / 2 = 20.0
    assert metrics.avg_successful_path_length == 20.0


def test_no_test_seed_leakage():
    """Verify that training seed ranges and test evaluation seed ranges are strictly disjoint."""
    train_cfg = TrainConfig()
    eval_cfg = EvalConfig()

    train_seeds = set(range(train_cfg.diverse_train_seed_start, train_cfg.diverse_train_seed_start + train_cfg.episodes))
    train_seeds.add(train_cfg.fixed_train_seed)

    test_seeds = set(range(eval_cfg.unseen_seed_start, eval_cfg.unseen_seed_start + eval_cfg.num_test_episodes))

    # Overlap must be empty
    overlap = train_seeds.intersection(test_seeds)
    assert len(overlap) == 0, f"Critical test leakage detected! Overlapping seeds: {overlap}"
