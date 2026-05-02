
import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

try:
    from torch_geometric.nn import GATConv, GATv2Conv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=2, layers=1, dropout=0.1, use_graph=True):
        super().__init__()
        self.use_graph = use_graph and TORCH_GEOMETRIC_AVAILABLE
        
        modules = []
        last_dim = in_dim
        for _ in range(layers):
            if self.use_graph:
                modules.append(GATv2Conv(last_dim, hidden_dim, heads=heads, dropout=dropout))
            else:
                modules.append(nn.Linear(last_dim, hidden_dim * heads))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
            last_dim = hidden_dim * heads
            
        self.layers = nn.ModuleList(modules)
        self.out_dim = last_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
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
        layers: int = 1, 
        dropout: float = 0.1, 
        horizon: int = 3,
        use_graph: bool = True,
        temporal_type: str = "gru"
    ):
        super().__init__()
        self.horizon = horizon
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.temporal_type = temporal_type
        
        self.spatial = SpatialEncoder(in_dim, hidden_dim, heads, layers, dropout, use_graph=use_graph)
        
        temporal_in = self.spatial.out_dim
        if temporal_type == "gru":
            self.temporal = nn.GRU(temporal_in, hidden_dim, batch_first=True)
        else:
            self.temporal = nn.Sequential(
                nn.Linear(temporal_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Reconstruction head (H steps)
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

        # Forecasting head (H_out steps)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * in_dim),
        )
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * in_dim),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_seq: [B, H, N, F]
        b, h, n, f = x_seq.shape
        
        x_spatial = []
        for i in range(h):
            x_spatial.append(self.spatial(x_seq[:, i], edge_index))
        
        x_spatial = torch.stack(x_spatial, dim=1) # [B, H, N, D]
        x_spatial = x_spatial.permute(0, 2, 1, 3).reshape(b * n, h, -1)  # [B*N, H, D]
        
        if self.temporal_type == "gru":
            _, h_n = self.temporal(x_spatial)
            x_temporal = h_n[-1]  # [B*N, hidden_dim]
        else:
            x_temporal = self.temporal(x_spatial)[:, -1, :]  # [B*N, hidden_dim]

        # Reconstruction
        recon = self.recon_head(x_temporal).reshape(b, n, f)

        # Forecasting
        mean_forecast = self.mean_head(x_temporal).reshape(b, n, self.horizon, f).permute(0, 2, 1, 3)
        log_var_forecast = self.var_head(x_temporal).reshape(b, n, self.horizon, f).permute(0, 2, 1, 3)
        
        variance_forecast = torch.exp(log_var_forecast)
        
        return recon, mean_forecast, variance_forecast
