"""
Training script for the ST-GNN-based anomaly detector (Phase 2).

This module provides a light-weight training loop around the
`SpatialTemporalAutoencoder` defined in `src.models.st_gnn`.
It is designed to support both real datasets and a placeholder
mode with randomly generated traffic sequences so that the
pipeline can be tested end-to-end without external data.
"""

from typing import Iterable

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.st_gnn import SpatialTemporalAutoencoder
from src.phase2.synthetic_data import (
    SyntheticTrafficSequenceDataset,
    build_fully_connected_edge_index,
)




def train_one_epoch(
    model: SpatialTemporalAutoencoder,
    data_loader: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    edge_index: torch.Tensor,
    recon_weight: float = 1.0,
    forecast_weight: float = 1.0,
) -> float:
    """
    Train the model for one epoch.

    Loss = recon_weight * L_recon + forecast_weight * L_forecast,
    where both terms are MSE losses.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    mse = nn.MSELoss()

    for batch in data_loader:
        # batch: [B, H+1, N, F]
        batch = batch.to(device)
        x_seq = batch[:, :-1]  # [B, H, N, F]
        target_last = batch[:, -1]  # [B, N, F]
        target_forecast = batch[:, 1:]  # [B, H, N, F]

        optimizer.zero_grad()
        recon, forecast = model(x_seq, edge_index)
        loss_recon = mse(recon, target_last)
        loss_forecast = mse(forecast, target_forecast)
        loss = recon_weight * loss_recon + forecast_weight * loss_forecast

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ST-GNN anomaly detector (Phase 2)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--horizon", type=int, default=3, help="Temporal horizon (H)")
    parser.add_argument("--num_nodes", type=int, default=4, help="Number of intersections (nodes)")
    parser.add_argument("--num_features", type=int, default=12, help="Number of node features")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=2, help="GAT heads")
    parser.add_argument("--layers", type=int, default=2, help="Number of GAT layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs/phase2", help="Directory to save model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model = SpatialTemporalAutoencoder(
        in_dim=args.num_features,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        horizon=args.horizon,
        use_graph=True,
        temporal_type="gru",
    ).to(device)

    # Synthetic normal dataset (no anomalies for training)
    dataset = SyntheticTrafficSequenceDataset(
        num_samples=512,
        horizon=args.horizon,
        num_nodes=args.num_nodes,
        num_features=args.num_features,
        anomaly_prob=0.0,
        return_labels=False,
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    edge_index = build_fully_connected_edge_index(args.num_nodes, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Phase 2 anomaly detector training (placeholder data)...")
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            edge_index=edge_index,
        )
        print(f"Epoch {epoch:03d} | Loss: {loss:.6f}")

    model_path = os.path.join(args.output_dir, "st_gnn_anomaly_detector.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Saved anomaly detector model to: {model_path}")


if __name__ == "__main__":
    main()

