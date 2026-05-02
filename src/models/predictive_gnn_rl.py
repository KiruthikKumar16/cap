
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
        self.st_gnn_in_dim = st_gnn_in_dim
        self.st_gnn_hidden_dim = st_gnn_hidden_dim

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

        self.input_proj = (
            nn.Identity()
            if st_gnn_in_dim == rl_gnn_in_dim
            else nn.Linear(st_gnn_in_dim, rl_gnn_in_dim)
        )
        
        self.forecast_decode = nn.Identity()

        self.controller = TrafficGNNEncoder(
            in_dim=rl_gnn_in_dim,
            hidden_dim=rl_gnn_hidden_dim,
            out_dim=rl_gnn_embedding_dim,
            num_layers=rl_gnn_layers,
            gnn_type=rl_gnn_type,
            gat_heads=rl_gnn_heads,
            dropout=rl_gnn_dropout,
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        x_seq = x_seq.to(device)
        edge_index = edge_index.to(device)

        recon, mean_forecast, variance_forecast = self.forecaster(x_seq, edge_index)
        predicted_state = mean_forecast[:, -1, :, :] # [B, N, feature_dim]
        
        batch_size = predicted_state.shape[0]
        if batch_size > 1:
            all_node_embeddings = []
            all_global_embeddings = []
            for i in range(batch_size):
                x = self.input_proj(predicted_state[i])
                node_embedding = self.controller(x, edge_index)
                global_embedding = torch.mean(node_embedding, dim=0, keepdim=True)
                
                all_node_embeddings.append(node_embedding)
                all_global_embeddings.append(global_embedding)
                
            return torch.cat(all_node_embeddings, dim=0), torch.cat(all_global_embeddings, dim=0), mean_forecast, variance_forecast
        else:
            x = self.input_proj(predicted_state.squeeze(0))
            node_embedding = self.controller(x, edge_index)
            global_embedding = torch.mean(node_embedding, dim=0, keepdim=True)
            return node_embedding, global_embedding, mean_forecast, variance_forecast

    def compute_forecasting_loss(
        self, 
        mean_forecast: torch.Tensor,
        actual_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates loss by decoding the latent forecast back to physical dimensions.
        mean_forecast: [B, H_out, N, in_dim]
        actual_next: [B, N, in_dim] or [B, H_out, N, in_dim]
        """
        # Take the last projected step
        decoded_prediction = self.forecast_decode(mean_forecast[:, -1, :, :])
        
        if actual_next.dim() == 4:
            actual_next = actual_next[:, -1, :, :]
            
        return torch.nn.functional.mse_loss(decoded_prediction, actual_next)
