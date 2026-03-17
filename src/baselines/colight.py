
"""
CoLight Agent Implementation

This module provides an implementation of the CoLight algorithm, a GNN-based
MARL method for traffic signal control.
"""

import torch
import torch.nn as nn

from src.phase1.gnn_encoder import TrafficGNNEncoder

class CoLightAgent(nn.Module):
    """
    A simplified implementation of the CoLight algorithm.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.gnn = TrafficGNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type="GAT",
        )
        self.q_head = nn.Linear(out_dim, 4) # 4 phases

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CoLight model.
        """
        x = self.gnn(x, edge_index)
        q_values = self.q_head(x)
        return q_values

    def predict(self, obs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict the best action based on the Q-values.
        """
        q_values = self.forward(obs, edge_index)
        return torch.argmax(q_values, dim=1)
