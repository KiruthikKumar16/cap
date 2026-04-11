"""
NSTLight Authentic Baseline Agent (2024/2025 SOTA)

Implements the defining Non-Stationary component:
1. Temporal Differencing (x_t - x_{t-1})
2. Multi-Head Attention (5 heads)
"""

import torch
import torch.nn as nn

from src.phase1.gnn_encoder import TrafficGNNEncoder


class NSTLightAgent(nn.Module):
    """
    Authentic NSTLight implementation:
    - Extracts non-stationary dynamics via observation differencing.
    - Utilizes 5-head Graph Attention to weigh local traffic dynamics.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        # Explicit 5-head attention as rigorously specified by the NSTLight architecture
        self.encoder = TrafficGNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type="GAT",
            gat_heads=5,
        )
        self.action_head = nn.Linear(out_dim, 4)

    def forward(self, x_t: torch.Tensor, x_prev: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Calculates non-stationary dynamics strictly via differencing.
        """
        # Non-Stationary Differencing Operation (x_t - x_{t-1})
        x_diff = x_t - x_prev
        
        # Process the dynamically shifted embedding via Graph Attention
        h = self.encoder(x_diff, edge_index)
        return self.action_head(h)

    def predict(self, obs: torch.Tensor, prev_obs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(obs, prev_obs, edge_index)
            return torch.argmax(logits, dim=1)
