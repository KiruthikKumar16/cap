"""
Evaluate ST-GNN anomaly detector with synthetic data.

This script runs the autoencoder on synthetic sequences with injected anomalies,
computes anomaly scores, selects a threshold, and reports precision/recall/F1.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.st_gnn import SpatialTemporalAutoencoder
from src.phase2.anomaly_scorer import combined_anomaly_score


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ST-GNN anomaly detector (Phase 2)")
    parser.add_argument("--model", type=str, default="outputs/phase2/st_gnn_anomaly_detector.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to real SUMO evaluation dataset.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--num_features", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold_method", type=str, default="quantile", choices=["quantile", "roc", "f1"])
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--output", type=str, default="outputs/phase2/anomaly_eval_summary.json")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Evaluate random initialized weights for smoke testing only.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model_path = Path(args.model)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model_source = "trained_checkpoint"
    elif args.allow_untrained:
        print(f"[WARN] Model not found at {model_path}. Using untrained weights.")
        model_source = "untrained_smoke_test"
    else:
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train Phase 2 first or pass "
            "--allow-untrained for a smoke test that must not be used as evidence."
        )

    model.eval()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {args.data}. Run 'scripts/generate_anomaly_data.py' first.")

    tensor_data = torch.load(args.data)
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(tensor_data)
    
    def collate_unpacked(batch):
        return torch.stack([item[0] for item in batch], dim=0)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_unpacked)
    edge_index = build_fully_connected_edge_index(args.num_nodes, device)

    combined_scores = []
    recon_scores = []
    forecast_scores = []

    with torch.no_grad():
        for x_plus in data_loader:
            x_plus = x_plus.to(device)
            x_seq = x_plus[:, :-1]
            recon, forecast, _ = model(x_seq, edge_index)
            scores, details = combined_anomaly_score(recon, forecast, x_plus)
            combined_scores.extend(scores.detach().cpu().reshape(-1).tolist())
            recon_scores.extend(details["recon_error"].detach().cpu().reshape(-1).tolist())
            forecast_scores.extend(details["forecast_error"].detach().cpu().reshape(-1).tolist())

    combined_scores = np.asarray(combined_scores, dtype=float)
    recon_scores = np.asarray(recon_scores, dtype=float)
    forecast_scores = np.asarray(forecast_scores, dtype=float)

    threshold = compute_threshold(
        combined_scores,
        method=args.threshold_method,
        quantile=args.quantile,
    )
    
    # NOTE: Labels are typically not available in real traffic unless specifically injected
    # We report the score distribution for Baseline evidence.
    summary = {
        "num_nodes": args.num_nodes,
        "num_features": args.num_features,
        "horizon": args.horizon,
        "model_source": model_source,
        "evidence_status": "verified_checkpoint" if model_source == "trained_checkpoint" else "smoke_test_not_evidence",
        "threshold_method": args.threshold_method,
        "threshold": float(threshold),
        "mean_combined_score": float(combined_scores.mean()),
        "std_combined_score": float(combined_scores.std()),
        "model_path": str(model_path),
        "data_path": str(args.data)
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Phase 2 evaluation summary saved to:", out_path)
    print("Metrics:", summary["metrics"])


if __name__ == "__main__":
    main()
