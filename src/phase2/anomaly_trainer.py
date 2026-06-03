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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.st_gnn import SpatialTemporalAutoencoder


def build_fully_connected_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Build a fully-connected directed edge_index.
    """
    src, dst = torch.meshgrid(
        torch.arange(num_nodes, dtype=torch.long),
        torch.arange(num_nodes, dtype=torch.long),
        indexing="ij",
    )
    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
    return edge_index.to(device)




def train_one_epoch(
    model: SpatialTemporalAutoencoder,
    source_loader: Iterable[torch.Tensor],
    target_loader: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    edge_index: torch.Tensor,
    recon_weight: float = 1.0,
    forecast_weight: float = 1.0,
    domain_weight: float = 1.0,
) -> float:
    """
    Train the model for one epoch using UDA.

    Source Loss = recon_weight * L_recon + forecast_weight * L_forecast + domain_weight * L_domain(0)
    Target Loss = domain_weight * L_domain(1)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    
    target_iter = iter(target_loader)

    for source_batch in source_loader:
        # Get target batch
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        optimizer.zero_grad()
        
        # --- Source Domain Pass ---
        source_batch = source_batch.to(device)
        s_x_seq = source_batch[:, :-1]
        s_target_last = source_batch[:, -1]
        s_target_forecast = source_batch[:, 1:]

        # Alpha is the GRL reversal weight
        p = float(num_batches) / max(1, len(source_loader))
        alpha = 2. / (1. + torch.exp(torch.tensor(-10. * p))) - 1
        alpha = float(alpha)

        s_recon, s_mean, s_var, s_domain = model(s_x_seq, edge_index, alpha=alpha)
        
        loss_recon = mse(s_recon, s_target_last)
        loss_forecast = mse(s_mean, s_target_forecast)
        
        s_domain_labels = torch.zeros_like(s_domain)
        s_domain_loss = bce(s_domain, s_domain_labels)
        
        # --- Target Domain Pass ---
        target_batch = target_batch.to(device)
        t_x_seq = target_batch[:, :-1]
        _, _, _, t_domain = model(t_x_seq, edge_index, alpha=alpha)
        
        t_domain_labels = torch.ones_like(t_domain)
        t_domain_loss = bce(t_domain, t_domain_labels)
        
        # --- Total Loss ---
        domain_loss = s_domain_loss + t_domain_loss
        loss = recon_weight * loss_recon + forecast_weight * loss_forecast + domain_weight * domain_loss

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
    parser.add_argument("--source_data", type=str, default="", help="Path to source domain SUMO dataset.")
    parser.add_argument("--target_data", type=str, default="", help="Path to target domain (real-world) dataset.")
    parser.add_argument("--use_dann", type=bool, default=True, help="Enable UDA via DANN.")
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

    def get_loader(data_file, is_target=False):
        if data_file and os.path.exists(data_file):
            print(f"Loading {'target' if is_target else 'source'} dataset from: {data_file}")
            tensor_data = torch.load(data_file)
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(tensor_data)
            def collate_unpacked(batch):
                return torch.stack([item[0] for item in batch], dim=0)
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_unpacked)
        else:
            raise FileNotFoundError(
                f"Missing {'target' if is_target else 'source'} dataset: {data_file}. "
                "Research-grade results require real SUMO data. "
                "Run 'scripts/generate_anomaly_data.py' first."
            )

    source_loader = get_loader(args.source_data, is_target=False)
    target_loader = get_loader(args.target_data, is_target=True)

    edge_index = build_fully_connected_edge_index(args.num_nodes, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Phase 2 anomaly detector training (placeholder data)...")
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
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

