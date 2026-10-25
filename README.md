# GRAIL-NAV: Generalization-Aware Deep Reinforcement Learning for Autonomous Robot Navigation in Unseen Obstacle Environments

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Tests-13%2F13%20Passing-brightgreen.svg)

---

## 1. Project Overview & Motivation
Autonomous mobile robots require collision-free, energy-efficient navigation to reach designated targets. While classic path-planners (e.g., A*, Dijkstra) rely on complete pre-mapped global geometry, Deep Reinforcement Learning (DRL) offers reactive, sensor-to-action decision making. However, standard Deep Q-Networks (DQN) suffer from **spatial overfitting**: policies trained on a single obstacle layout memorize fixed coordinate milestones and fail catastrophically when deployed into unseen obstacle configurations.

**GRAIL-NAV** systematically investigates this phenomenon by implementing a pure Deep Q-Network (DQN) navigation architecture and evaluating whether **diverse procedural layout training** produces invariant obstacle-avoidance heuristics compared to **fixed-layout training**.

---

## 2. Research Hypothesis & Core Contribution
> **Hypothesis:** Exposing a DQN agent to diverse procedurally generated obstacle topologies during training prevents coordinate memorization and enables generalization to held-out unseen layouts with significantly lower collision rates.

### Key Contributions:
- **Zero-Dependency 10×10 Grid World Simulator:** Built from scratch with Gymnasium-style `reset()` and `step()` interfaces, collision physics, and Breadth-First Search (BFS) reachability validation.
- **Pure DQN Implementation:** Adheres strictly to Mnih et al. (2015) with online Q-network, target network synchronization, experience replay buffer (20,000 transitions), and Bellman targets with terminal-state masks.
- **Telescoping Potential-Based Reward Shaping:** Combines goal (+100) and collision (-100) penalties with distance differences ($\Delta \Phi$) that mathematically cancel out over cycles, eliminating looping exploits.
- **Empirical Generalization Matrix:** A $2 \times 2$ comparative evaluation across 50 strictly held-out unseen map seeds.

---

## 3. System Architecture & State-Action Design

```
GRAIL-NAV/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── environment/
│   ├── __init__.py
│   └── grid_world.py
├── agent/
│   ├── __init__.py
│   ├── dqn_network.py
│   ├── replay_buffer.py
│   └── dqn_agent.py
├── training/
│   ├── __init__.py
│   └── train.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   └── metrics.py
├── visualization/
│   ├── __init__.py
│   ├── plot_results.py
│   └── render_environment.py
├── tests/
│   ├── test_environment.py
│   └── test_agent.py
├── models/
├── results/
└── docs/
    ├── problem_statement.md
    ├── methodology.md
    ├── experimental_results.md
    └── limitations.md
```

### State Space (33 Continuous Features)
$$s = [\bar{x}_{rob}, \bar{y}_{rob}, \bar{x}_{goal}, \bar{y}_{goal}, \Delta\bar{x}, \Delta\bar{y}, \bar{d}_{manhattan}, \bar{d}_{euclidean}, \mathbf{O}_{local(5\times 5)}]$$
- Normalized Robot Coordinates $[0, 1]$ (2 values)
- Normalized Goal Coordinates $[0, 1]$ (2 values)
- Relative Goal Displacement Vector $[-1, 1]$ (2 values)
- Normalized Manhattan & Euclidean Distances $[0, 1]$ (2 values)
- Local $5 \times 5$ Occupancy Field ($r=2$): 25 values ($1.0$ for obstacles/walls, $-1.0$ for goal, $0.0$ for open space).

### Action Space (Discrete $|\mathcal{A}| = 4$)
- `0`: Move UP $(-1, 0)$
- `1`: Move DOWN $(+1, 0)$
- `2`: Move LEFT $(0, -1)$
- `3`: Move RIGHT $(0, +1)$

---

## 4. Empirical Evaluation Results

Evaluated across 50 held-out unseen map seeds ($9000 \dots 9049$) under deterministic greedy action selection ($\epsilon = 0.0$):

| Agent Policy | Environment Condition | Success Rate (%) | Collision Rate (%) | Timeout Rate (%) | Avg Cumulative Reward | Avg Steps | Decision Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Map Agent** | Familiar Map (Seed 42) | **100.0%** | **0.0%** | 0.0% | **+108.50** | 18.00 | 0.597 ms |
| **Fixed-Map Agent** | Held-Out Unseen Maps | **0.0%** | **18.0%** | 82.0% | **-304.14** | 50.04 | 0.638 ms |
| **Diverse-Map Agent** | Familiar Map (Seed 42) | 0.0% | **0.0%** | 100.0% | -48.00 | 60.00 | 0.639 ms |
| **Diverse-Map Agent** | Held-Out Unseen Maps | **14.0%** | **0.0%** | 86.0% | **-20.99** | 54.12 | 0.720 ms |

### Key Findings:
1. **Memorization Collapse in Fixed Agent:** The Fixed Agent achieved 100% on its training layout, but dropped to **0% success** on unseen layouts with **18% collisions** and **-304.14 reward**.
2. **Zero-Collision Safety in Diverse Agent:** The Diverse Agent achieved a **0.0% collision rate** across unseen environments and successfully reached goals on **14% of unseen maps** with an average reward of **-20.99**.

---

## 5. Installation & Quickstart

### 1. Prerequisites
- Python 3.10+
- PyTorch, NumPy, Matplotlib, Pandas, Pytest

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/
```

### 3. Train Navigation Agents
Train Fixed-Map Agent:
```bash
python main.py --mode train --scenario fixed
```

Train Diverse-Map Agent:
```bash
python main.py --mode train --scenario diverse
```

### 4. Evaluate Saved Policies
Evaluate Fixed Agent on Familiar and Unseen Maps:
```bash
python main.py --mode evaluate --scenario fixed
```

Evaluate Diverse Agent on Familiar and Unseen Maps:
```bash
python main.py --mode evaluate --scenario diverse
```

### 5. Generate Full Comparative Report and Publication Plots
```bash
python main.py --mode report
```

### 6. Visualize 2D Policy Trajectory
```bash
python main.py --mode visualize --seed 9005
```

---

## 6. Generated Visualizations & Artifacts
The following publication-grade plots and metrics are generated in `results/`:
- `results/plots/plot_training_curves.png`: Training rewards and 50-episode moving averages.
- `results/plots/plot_success_progression.png`: 50-episode rolling success rate comparison.
- `results/plots/plot_generalization_comparison.png`: Bar chart comparing Success & Collision rates.
- `results/plots/plot_path_length_comparison.png`: Path length efficiency comparison.
- `results/plots/trajectory_seed_9005.png`: Rendered 2D grid navigation trajectory from trained policy.
- `results/evaluation_summary.csv`: Machine-readable evaluation table.

---

## 7. Documentation
- [Problem Statement](file:///docs/problem_statement.md)
- [Dataset & Environment Specification](file:///docs/dataset_information.md)
- [Methodology & Mathematics](file:///docs/methodology.md)
- [Experimental Results & Viva Voce Guide](file:///docs/experimental_results.md)
- [Systemic Limitations & Future Work](file:///docs/limitations.md)
