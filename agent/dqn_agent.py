"""
Deep Q-Network (DQN) Agent for Autonomous Navigation.
Implements double-buffered Q-learning with target synchronization,
experience replay, Huber temporal-difference loss, and epsilon-greedy exploration.
"""

from pathlib import Path
import random
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import DQNConfig, DEFAULT_DQN_CONFIG
from .dqn_network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Standard DQN Agent adhering strictly to Mnih et al. (2015) mechanics.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        config: Optional[DQNConfig] = None,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or DEFAULT_DQN_CONFIG
        self.device = torch.device(device)

        # Networks: Online Q-Network and Target Q-Network
        self.online_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)

        # Synchronize target network initially
        self.update_target_network()
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate
        )

        # Experience Replay Buffer
        self.memory = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            device=str(self.device)
        )

        # Exploration hyperparameters
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.gamma = self.config.gamma
        self.grad_clip_norm = self.config.grad_clip_norm

        self.total_steps = 0

    def update_target_network(self) -> None:
        """Copy weights from online network to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select an action using an epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Flattened observation vector.
            eval_mode (bool): If True, disables exploration and enforces pure greedy argmax.
            
        Returns:
            int: Action index in [0, action_dim - 1].
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        self.online_net.train()
        return action

    def decay_epsilon(self) -> None:
        """Decay exploration rate after an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store interaction experience in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def learn(self, batch_size: Optional[int] = None) -> Optional[float]:
        """
        Sample a minibatch and perform one gradient descent optimization step.
        
        Returns:
            float: Temporal difference loss value, or None if buffer has insufficient samples.
        """
        batch_size = batch_size or self.config.batch_size
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # 1. Compute Q(s, a) using online network
        current_q = self.online_net(states).gather(1, actions)

        # 2. Compute Bellman Target y = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            # Zero out bootstrap value for terminal transitions (goal reach or obstacle collision)
            target_q = rewards + (self.gamma * max_next_q * (1.0 - dones))

        # 3. Compute Huber / Smooth L1 Loss for TD error robustness
        loss = F.smooth_l1_loss(current_q, target_q)

        # 4. Gradient optimization with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model checkpoint to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }
        torch.save(checkpoint, path)

    def load(self, filepath: Union[str, Path]) -> None:
        """Load model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.total_steps = checkpoint.get("total_steps", 0)
