# GRAIL-NAV: Problem Statement

## 1. Domain Context
Autonomous mobile robot navigation in constrained environments is a cornerstone capability for warehouse logistics, planetary exploration, search-and-rescue, and service robotics. Traditional algorithms (such as Dijkstra, A*, or Rapidly-exploring Random Trees [RRT]) rely on complete geometric maps of the environment and execute static graph searches. When operating in dynamic or previously uncharted terrains, static path planning suffers from high computational replanning latency, vulnerability to unmapped obstacles, and lack of adaptability.

## 2. The Core Machine Learning Challenge
Deep Reinforcement Learning (DRL), specifically Deep Q-Networks (DQN), enables an autonomous agent to synthesize an end-to-end mapping from sensory observations directly to discrete motor actions through trial-and-error environmental interactions.

However, a fundamental failure mode in DRL navigation is **layout overfitting**:
- When a DQN policy is trained in a **single fixed obstacle configuration**, the agent frequently memorizes the exact spatial geometry and coordinate milestones of that training map rather than learning generalized collision avoidance primitives and goal-seeking heuristics.
- Consequently, when deployed into an unseen obstacle configuration with identical obstacle density and grid dimensions, the agent's performance collapses—exhibiting high collision rates, looping trajectories, and deadlocks.

## 3. Objective of GRAIL-NAV
The objective of this research engineering project is to investigate the generalization capacity of Deep Q-Networks:
> **Research Question:** Does training a DQN agent across procedurally generated, diverse obstacle layouts cultivate invariant obstacle avoidance heuristics and improve goal-reaching success rates on unseen layouts compared to training on a fixed layout?

## 4. Scope and Algorithmic Constraints
1. **Algorithm Boundary:** Deep Q-Network (DQN) is the sole machine-learning and reinforcement learning algorithm utilized (incorporating experience replay, target network synchronization, and epsilon-greedy exploration). No policy-gradient methods (PPO/SAC), recurrent models (LSTM), or supervised classifiers are permitted.
2. **Compute Environment:** Completely executable on commodity CPU hardware without requiring dedicated GPU acceleration.
3. **Environment:** A custom, fully reproducible 10×10 grid world with randomized and fixed obstacle topologies, discrete 4-directional kinematics, potential-based reward shaping, and local occupancy field perception.
