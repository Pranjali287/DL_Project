"""
PyTorch Multi-Layer Perceptron Q-Network for GRAIL-NAV.
"""

from typing import List
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Deep Q-Network approximating the state-action value function Q(s, a).
    """
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dims: List[int] = None):
        super(QNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Final action-value output layer
        out_layer = nn.Linear(prev_dim, action_dim)
        nn.init.xavier_uniform_(out_layer.weight)
        nn.init.constant_(out_layer.bias, 0.0)
        layers.append(out_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q(s, a) for all discrete actions a.
        
        Args:
            state (torch.Tensor): Shape (batch_size, state_dim) or (state_dim,)
        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_dim)
        """
        return self.network(state)
