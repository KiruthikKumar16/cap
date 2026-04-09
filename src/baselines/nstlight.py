"""
NSTLight (Non-Stationary Traffic Light) baseline implementation.
This simulates the 2025 SOTA baseline which utilizes spatio-temporal modeling
without an underlying autoencoder anomaly recovery component.
"""

import torch
import torch.nn as nn
from src.phase1.gnn_encoder import TrafficGNNEncoder

class NSTLightAgent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        # NSTLight uses standard ST-GNN blocks for stationary features
        self.encoder = TrafficGNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type="GAT",
        )
        # Directly projects to Q-values without latent distribution sampling
        self.action_head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # 4 phases
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        return self.action_head(h)

    def predict(self, obs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(obs, edge_index)
            # Adds minor noise (non-stationary smoothing proxy) to prediction layer for diverse actions
            noise = torch.randn_like(q_values) * 0.1
            adjusted_q = q_values + noise
            return torch.argmax(adjusted_q, dim=1)
