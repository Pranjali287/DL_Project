# GRAIL-NAV: Project Limitations & Future Research Directions

## 1. Systemic Limitations

### 1.1 Discrete Grid Discretization vs. Continuous Kinematics
- **Limitation:** GRAIL-NAV simulates robot motion on a discrete 10×10 grid with orthogonal movements (Up, Down, Left, Right). 
- **Physical Reality:** Real physical mobile robots (e.g., TurtleBot, differential drive, or Ackermann steering platforms) operate in continuous Euclidean space $\mathbb{R}^2 \times SO(2)$ governed by non-holonomic kinematic constraints, velocity bounds, angular acceleration limits, and wheel slip.

### 1.2 Static Obstacle Topology
- **Limitation:** Obstacle fields in this study are generated at episode initialization and remain static throughout the episode.
- **Physical Reality:** Real-world logistics and warehouse environments feature dynamic actors (pedestrians, forklifts, opening/closing automatic doors). Pure spatial observation vectors lack temporal velocity vectors $(\dot{x}_{obs}, \dot{y}_{obs})$.

### 1.3 Sample Complexity of Model-Free DQN
- **Limitation:** Model-free Deep Q-Learning requires several hundred episodes of exploratory trial-and-error interaction to backpropagate sparse goal rewards into stable action-value estimates.
- **Comparison to Classical Planners:** Classical graph planners (A*, Dijkstra) synthesize optimal paths in milliseconds when global map geometry is fully known, without requiring prior training. DQN's advantage emerges primarily when navigating under uncertainty or partial observability where fast reactive reflexes are mandatory.

### 1.4 Global Goal Vector Dependency
- **Limitation:** The state observation vector assumes the agent has continuous, exact access to its global coordinates and the goal coordinates (simulated GPS/odometry).
- **Physical Reality:** In GPS-denied indoor environments, localization is subject to sensor drift, requiring simultaneous localization and mapping (SLAM) or landmark recognition.

### 1.5 Deterministic Actuation Dynamics
- **Limitation:** Transition dynamics $\mathcal{P}(s' | s, a)$ are deterministic; a commanded "Right" action moves the agent exactly one unit right unless blocked.
- **Physical Reality:** Real ground robots experience odometry noise, surface traction variation, and actuator response delays.

---

## 2. Recommended Future Research Directions

1. **Continuous Action Spaces:** Transition from discrete DQN to continuous actor-critic formulations (e.g., Soft Actor-Critic [SAC] or Twin Delayed DDPG [TD3]) commanding continuous linear velocity $v \in [0, v_{max}]$ and angular velocity $\omega \in [-\omega_{max}, \omega_{max}]$.
2. **LiDAR / Range-Sensor Simulators:** Replace discrete local occupancy windows with simulated 2D planar LiDAR laser scan rays (360-degree range beams) for realistic sensor simulation.
3. **Dynamic Actor Modeling:** Introduce moving obstacles with random or reactive trajectories, necessitating spatio-temporal state buffers or recurrent observation processing.
4. **Sim-to-Real Transfer & Domain Randomization:** Inject randomized action delays, motor noise, and observation Gaussian noise during training to evaluate zero-shot sim-to-real transfer on physical robotic hardware.
