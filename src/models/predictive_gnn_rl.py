
"""
Predictive GNN-RL Model for Traffic Control

This module combines a Spatio-Temporal GNN (ST-GNN) for traffic forecasting
with a GNN-based DQN for reinforcement learning-based control.
"""

import torch
import torch.nn as nn
from typing import Tuple

from src.models.st_gnn import SpatialTemporalAutoencoder
from src.phase1.gnn_encoder import TrafficGNNEncoder

class PredictiveGNNRL(nn.Module):
    """
    A unified model that first predicts future traffic states and then uses
    those predictions to make control decisions.
    """
    def __init__(
        self,
        st_gnn_in_dim: int,
        st_gnn_hidden_dim: int,
        st_gnn_heads: int,
        st_gnn_layers: int,
        st_gnn_dropout: float,
        st_gnn_horizon: int,
        rl_gnn_in_dim: int,
        rl_gnn_hidden_dim: int,
        rl_gnn_embedding_dim: int,
        rl_gnn_layers: int,
        rl_gnn_type: str,
        rl_gnn_heads: int,
        rl_gnn_dropout: float,
    ):
        super().__init__()

        self.forecaster = SpatialTemporalAutoencoder(
            in_dim=st_gnn_in_dim,
            hidden_dim=st_gnn_hidden_dim,
            heads=st_gnn_heads,
            layers=st_gnn_layers,
            dropout=st_gnn_dropout,
            horizon=st_gnn_horizon,
            use_graph=True,
            temporal_type="gru",
        )

        self.controller = TrafficGNNEncoder(
            in_dim=rl_gnn_in_dim,
            hidden_dim=rl_gnn_hidden_dim,
            out_dim=rl_gnn_embedding_dim,
            num_layers=rl_gnn_layers,
            gnn_type=rl_gnn_type,
            gat_heads=rl_gnn_heads,
            dropout=rl_gnn_dropout,
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        First, forecast the future traffic state, then use the forecast to
        generate control embeddings.

        Args:
            x_seq: A sequence of historical traffic states [B, H, N, F]
            edge_index: The graph connectivity

        Returns:
            A tuple of (control_embeddings, forecasted_state)
        """
        # We only need the forecasted state, not the reconstruction
        _, forecast = self.forecaster(x_seq, edge_index)

        # Use the last forecasted step as the input to the controller
        # forecast is [B, H, N, F], we take the last step [B, N, F]
        predicted_state = forecast[:, -1, :, :]

        batch_size = predicted_state.shape[0]
        if batch_size > 1:
            all_embeddings = []
            for i in range(batch_size):
                single_embedding = self.controller(predicted_state[i], edge_index)
                all_embeddings.append(single_embedding)
            return torch.cat(all_embeddings, dim=0), forecast
        else:
            control_embedding = self.controller(predicted_state.squeeze(0), edge_index)
            return control_embedding, forecast
