"""
NSTLight dummy baseline agent.

This baseline intentionally stays lightweight and deterministic enough for
benchmarking against MAPPO without requiring training/checkpoints.
"""

import torch
import torch.nn as nn

from src.phase1.gnn_encoder import TrafficGNNEncoder


class NSTLightAgent(nn.Module):
    """
    Heuristic + shallow GNN baseline:
    - Uses a compact encoder for relational context.
    - Applies a simple pressure-inspired tie-break to keep it "dummy" and cheap.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.encoder = TrafficGNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type="GAT",
        )
        self.action_head = nn.Linear(out_dim, 4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        return self.action_head(h)

    def predict(self, obs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(obs, edge_index)
            # Small deterministic bias by feature sum to emulate non-stationary response.
            pressure_bias = obs.sum(dim=1, keepdim=True) * 0.001
            logits = logits + pressure_bias.repeat(1, logits.shape[1])
            return torch.argmax(logits, dim=1)
