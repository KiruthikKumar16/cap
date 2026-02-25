from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, H, D]
        return self.encoder(x)


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph = use_graph
        modules = []
        last_dim = in_dim
        for _ in range(layers):
            if use_graph:
                modules.append(GATv2Conv(last_dim, hidden_dim, heads=heads, dropout=dropout))
                last_dim = hidden_dim * heads
            else:
                modules.append(nn.Linear(last_dim, hidden_dim))
                last_dim = hidden_dim
        self.layers = nn.ModuleList(modules)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = last_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F]
        b, n, f = x.shape
        x = x.reshape(b * n, f)
        for layer in self.layers:
            if self.use_graph:
                x = layer(x, edge_index)
            else:
                x = layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return x.reshape(b, n, -1)


class SpatialTemporalAutoencoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 3,
        use_graph: bool = True,
        temporal_type: str = "gru",
        temporal_heads: int = 2,
        temporal_ff_mult: int = 2,
        temporal_layers: int = 1,
    ) -> None:
        super().__init__()
        self.spatial = SpatialEncoder(in_dim, hidden_dim, heads, layers, dropout, use_graph=use_graph)
        self.temporal_type = temporal_type
        self.horizon = horizon
        temporal_in = self.spatial.out_dim
        if temporal_type == "gru":
            self.temporal = nn.GRU(temporal_in, hidden_dim, batch_first=True)
            temporal_in = hidden_dim
        elif temporal_type == "transformer":
            self.temporal = TemporalTransformer(
                d_model=temporal_in,
                n_heads=temporal_heads,
                ff_mult=temporal_ff_mult,
                num_layers=temporal_layers,
                dropout=dropout,
            )
        self.recon_head = nn.Sequential(
            nn.Linear(temporal_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(temporal_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * in_dim),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: [B, H, N, F]
            edge_index: [2, E]
        Returns:
            recon: [B, N, F] reconstruction of last step
            forecast: [B, horizon, N, F]
        """
        b, h, n, f = x_seq.shape
        spatial_outputs = []
        for t in range(h):
            spatial_outputs.append(self.spatial(x_seq[:, t], edge_index))  # [B, N, S]
        spatial_stack = torch.stack(spatial_outputs, dim=1)  # [B, H, N, S]
        seq_flat = spatial_stack.reshape(b * n, h, -1)
        if self.temporal_type == "gru":
            seq_out, _ = self.temporal(seq_flat)  # [B*N, H, hidden]
        elif self.temporal_type == "transformer":
            seq_out = self.temporal(seq_flat)  # [B*N, H, hidden]
        else:
            seq_out = seq_flat
        final_state = seq_out[:, -1]  # [B*N, hidden]
        recon = self.recon_head(final_state).reshape(b, n, f)
        forecast = self.forecast_head(final_state).reshape(b, n, self.horizon, f)
        forecast = forecast.permute(0, 2, 1, 3)  # [B, horizon, N, F]
        return recon, forecast

