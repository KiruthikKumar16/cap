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
        self.mean_head = nn.Sequential(
            nn.Linear(temporal_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * in_dim),
        )
        self.var_head = nn.Sequential(
            nn.Linear(temporal_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * in_dim),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ST-GNN Autoencoder.
        
        Args:
            x_seq: Input sequence [B, H, N, F]
            edge_index: Graph connectivity
            
        Returns:
            recon: Reconstructed last step [B, N, F]
            mean_forecast: Predicted future sequence [B, H_out, N, F]
            variance_forecast: Predicted variance of future sequence [B, H_out, N, F] (Uncertainty-aware)
        """
        b, h, n, f = x_seq.shape
        # Spatial encoding per step
        x_spatial = []
        for i in range(h):
            x_spatial.append(self.spatial(x_seq[:, i], edge_index))
        x_spatial = torch.stack(x_spatial, dim=1)  # [B, H, N, D]

        # Temporal encoding
        x_spatial = x_spatial.permute(0, 2, 1, 3).reshape(b * n, h, -1)  # [B*N, H, D]
        if self.temporal_type == "gru":
            _, h_n = self.temporal(x_spatial)
            x_temporal = h_n.squeeze(0)  # [B*N, D]
        else:
            x_temporal = self.temporal(x_spatial)[:, -1, :]  # [B*N, D]

        # Reconstruction head (last step)
        recon = self.recon_head(x_temporal).reshape(b, n, f)

        # Forecasting head (H_out steps)
        # We predict mean and log_variance for each step
        mean_forecast = self.mean_head(x_temporal).reshape(b, n, self.horizon, f).permute(0, 2, 1, 3)
        log_var_forecast = self.var_head(x_temporal).reshape(b, n, self.horizon, f).permute(0, 2, 1, 3)
        
        # Uncertainty = exp(log_var)
        variance_forecast = torch.exp(log_var_forecast)

        return recon, mean_forecast, variance_forecast

    def mc_dropout_predict(
        self, 
        x_seq: torch.Tensor, 
        edge_index: torch.Tensor, 
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for robust uncertainty estimation.
        (Patent Angle: Uncertainty-aware anomaly detection using Bayesian ST-GNNs)
        
        Args:
            x_seq: Input sequence [B, H, N, F]
            edge_index: Graph connectivity
            num_samples: Number of MC dropout samples
            
        Returns:
            Combined mean and total variance (epistemic + aleatoric)
        """
        self.train()  # Enable dropout during inference
        
        means = []
        vars_aleatoric = []
        
        for _ in range(num_samples):
            _, m, v = self.forward(x_seq, edge_index)
            means.append(m)
            vars_aleatoric.append(v)
            
        means = torch.stack(means)  # [S, B, H, N, F]
        vars_aleatoric = torch.stack(vars_aleatoric)
        
        # Predictive Mean
        final_mean = means.mean(dim=0)
        
        # Epistemic Uncertainty (Variance of means)
        var_epistemic = means.var(dim=0)
        
        # Aleatoric Uncertainty (Mean of variances)
        var_aleatoric = vars_aleatoric.mean(dim=0)
        
        # Total Uncertainty
        total_variance = var_epistemic + var_aleatoric
        
        self.eval() # Restore to eval mode
        return final_mean, total_variance

