
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

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        First, forecast the future traffic state, then use the forecast to
        generate control embeddings.

        Args:
            x_seq: A sequence of historical traffic states [B, H, N, F]
            edge_index: The graph connectivity

        Returns:
            A tuple of (node_embeddings, global_graph_embedding, mean_forecast, variance_forecast)
        """
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        x_seq = x_seq.to(device)
        edge_index = edge_index.to(device)

        # We only need the forecasted state, not the reconstruction
        recon, mean_forecast, variance_forecast = self.forecaster(x_seq, edge_index)

        # Use the last forecasted step as the input to the controller
        # forecast is [B, H, N, F], we take the last step [B, N, F]
        predicted_state = mean_forecast[:, -1, :, :]

        batch_size = predicted_state.shape[0]
        if batch_size > 1:
            all_node_embeddings = []
            all_global_embeddings = []
            for i in range(batch_size):
                node_embedding = self.controller(predicted_state[i], edge_index)
                global_embedding = torch.mean(node_embedding, dim=0, keepdim=True)
                
                all_node_embeddings.append(node_embedding)
                all_global_embeddings.append(global_embedding)
                
            return torch.cat(all_node_embeddings, dim=0), torch.cat(all_global_embeddings, dim=0), mean_forecast, variance_forecast
        else:
            node_embedding = self.controller(predicted_state.squeeze(0), edge_index)
            global_embedding = torch.mean(node_embedding, dim=0, keepdim=True)
            return node_embedding, global_embedding, mean_forecast, variance_forecast

    def compute_forecasting_loss(
        self, 
        x_seq: torch.Tensor, 
        edge_index: torch.Tensor, 
        y_future: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Negative Log Likelihood loss for forecasting.
        y_future: [B, H_out, N, F]
        """
        _, mean, var = self.forward(x_seq, edge_index)[2:] # Get mean and var from forward
        
        # NLL Loss for Gaussian distribution
        # Loss = 0.5 * (log(var) + (y - mean)^2 / var)
        precision = 1.0 / (var + 1e-6)
        loss = 0.5 * (torch.log(var + 1e-6) + (y_future - mean)**2 * precision)
        return loss.mean()
