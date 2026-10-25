# GRAIL-NAV: Methodology & Mathematical Formulation

## 1. Markov Decision Process (MDP) Formulation
The autonomous robot navigation problem is modeled as an MDP defined by the 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

### 1.1 State Space $\mathcal{S}$
A compact 33-dimensional normalized continuous vector representing global goal orientation and local obstacle perception:
$$s = \left[ \bar{x}_{rob}, \bar{y}_{rob}, \bar{x}_{goal}, \bar{y}_{goal}, \Delta \bar{x}, \Delta \bar{y}, \bar{d}_{manhattan}, \bar{d}_{euclidean}, \mathbf{O}_{local} \right]$$
- $\bar{x}_{rob}, \bar{y}_{rob} \in [0, 1]$: Normalized coordinates of the robot in the $N \times N$ grid ($N=10$).
- $\bar{x}_{goal}, \bar{y}_{goal} \in [0, 1]$: Normalized coordinates of the target destination.
- $\Delta \bar{x}, \Delta \bar{y} \in [-1, 1]$: Relative displacement vector pointing from robot to goal.
- $\bar{d}_{manhattan}, \bar{d}_{euclidean} \in [0, 1]$: Normalized distance metrics providing immediate scalar proximity.
- $\mathbf{O}_{local} \in \mathbb{R}^{25}$: Flattened $5 \times 5$ local occupancy grid centered on the robot ($r_{obs} = 2$). Cell values encode:
  - $1.0$: Impenetrable boundary wall or static obstacle.
  - $-1.0$: Target goal cell.
  - $0.0$: Free navigable space.

### 1.2 Action Space $\mathcal{A}$
Discrete action set of size $|\mathcal{A}| = 4$:
- $a = 0$: Move UP $(-1, 0)$
- $a = 1$: Move DOWN $(+1, 0)$
- $a = 2$: Move LEFT $(0, -1)$
- $a = 3$: Move RIGHT $(0, +1)$

### 1.3 Reward Formulation $\mathcal{R}(s, a, s')$
The reward function is formulated to guarantee safety, promote swift termination, and eliminate cycle exploitation:
$$R(s, a, s') = R_{step} + R_{event} + R_{potential}(s, s')$$
- $R_{event} = +100.0$ if $s'$ is the goal cell (Terminal).
- $R_{event} = -100.0$ if $s'$ is an obstacle cell (Fatal collision, Terminal).
- $R_{event} = -10.0$ if action $a$ attempts to cross the grid boundary (Invalid move; agent remains in place).
- $R_{step} = -1.0$: Time-step penalty incentivizing shortest paths.
- $R_{potential}(s, s') = c \cdot \left( d_{manhattan}(s, \text{goal}) - d_{manhattan}(s', \text{goal}) \right)$ where $c = 1.5$:
  - When moving closer to the goal, the robot receives $+1.5$.
  - When moving farther, the robot receives $-1.5$.
  - **Telescoping Guarantee:** Over any closed cyclic trajectory $A \to B \to A$, the net sum of distance potential rewards is identically zero ($\Delta d + (-\Delta d) = 0$), while accumulating $-2.0$ in step penalties. This mathematically prevents back-and-forth oscillation loops.

---

## 2. Deep Q-Network (DQN) Architecture & Algorithm

### 2.1 Multi-Layer Perceptron (MLP)
- **Input Layer:** 33 neurons (normalized state vector).
- **Hidden Layer 1:** 128 neurons, ReLU activation, Kaiming Normal initialization.
- **Hidden Layer 2:** 128 neurons, ReLU activation, Kaiming Normal initialization.
- **Output Layer:** 4 neurons representing $Q(s, a)$ for each action, Xavier Uniform initialization.

### 2.2 Bellman Optimality and Target Network
The agent maintains two separate neural parameter sets:
1. Online network parameters $\theta$ (updated at every gradient step).
2. Target network parameters $\theta^-$ (frozen and periodically synchronized: $\theta^- \leftarrow \theta$ every 10 episodes).

The Temporal Difference (TD) target is computed with an explicit terminal mask:
$$y_i = r_i + \gamma (1 - d_i) \max_{a' \in \mathcal{A}} Q(s'_i, a'; \theta^-)$$
where $d_i \in \{0, 1\}$ indicates whether state $s'_i$ was terminal. If $d_i = 1$ (goal reached or fatal collision), the bootstrap term $(1 - d_i) \max Q$ zeroes out identically, ensuring the Bellman equation does not extrapolate values beyond episode termination.

### 2.3 Loss Function and Optimization
The network is optimized using the Huber (Smooth $L_1$) loss:
$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \text{Smooth}_{L1} \left( Q(s_i, a_i; \theta) - y_i \right)$$
$$\text{Smooth}_{L1}(u) = \begin{cases} 0.5 u^2 & \text{if } |u| < 1 \\ |u| - 0.5 & \text{otherwise} \end{cases}$$
Optimization is performed via the Adam optimizer ($\alpha = 0.001$) combined with gradient norm clipping ($\|g\|_2 \le 10.0$) to prevent exploding gradients.

### 2.4 Experience Replay
Transitions $(s, a, r, s', d)$ are stored in a cyclic replay memory buffer $\mathcal{D}$ with capacity $|\mathcal{D}| = 20,000$. Minibatches of size $B = 64$ are sampled uniformly at random to break temporal autocorrelation in sequential robot trajectories.

---

## 3. Experimental Protocol: Fixed vs. Diverse Regimens
To rigorously test the generalization hypothesis:
1. **Condition A (Fixed-Map):** The agent is trained exclusively on a single fixed obstacle configuration generated with seed 42.
2. **Condition B (Diverse-Map):** The agent is trained across procedural layouts generated with a unique seed per episode ($1000 + e$). Each layout is verified via Breadth-First Search (BFS) to guarantee a reachable path from start $(0, 0)$ to goal $(9, 9)$.
3. **Evaluation Protocol:** Both trained policies are tested under deterministic greedy action selection ($\epsilon = 0$) across:
   - Familiar layouts (Seed 42).
   - 50 Held-out Unseen layouts generated from strictly isolated seeds ($9000 \dots 9049$).
