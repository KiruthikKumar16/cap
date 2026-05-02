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
from src.phase2.anomaly_scorer import combined_anomaly_score, reconstruction_error, forecasting_error
from src.phase2.synthetic_data import SyntheticTrafficSequenceDataset, build_fully_connected_edge_index
from src.utils.metrics import compute_threshold, evaluate_anomalies


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ST-GNN anomaly detector (Phase 2)")
    parser.add_argument("--model", type=str, default="outputs/phase2/st_gnn_anomaly_detector.pt")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--num_features", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--anomaly_prob", type=float, default=0.1)
    parser.add_argument("--anomaly_scale", type=float, default=0.6)
    parser.add_argument("--anomaly_span", type=int, default=1)
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

    dataset = SyntheticTrafficSequenceDataset(
        num_samples=args.samples,
        horizon=args.horizon,
        num_nodes=args.num_nodes,
        num_features=args.num_features,
        anomaly_prob=args.anomaly_prob,
        anomaly_scale=args.anomaly_scale,
        anomaly_span=args.anomaly_span,
        return_labels=True,
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    edge_index = build_fully_connected_edge_index(args.num_nodes, device)

    combined_scores = []
    recon_scores = []
    forecast_scores = []
    labels = []

    with torch.no_grad():
        for x_plus, batch_labels in data_loader:
            x_plus = x_plus.to(device)
            x_seq = x_plus[:, :-1]
            recon, forecast, _ = model(x_seq, edge_index)
            scores, details = combined_anomaly_score(recon, forecast, x_plus)
            combined_scores.extend(scores.detach().cpu().reshape(-1).tolist())
            recon_scores.extend(details["recon_error"].detach().cpu().reshape(-1).tolist())
            forecast_scores.extend(details["forecast_error"].detach().cpu().reshape(-1).tolist())
            labels.extend(batch_labels.reshape(-1).cpu().numpy().astype(int).tolist())

    combined_scores = np.asarray(combined_scores, dtype=float)
    recon_scores = np.asarray(recon_scores, dtype=float)
    forecast_scores = np.asarray(forecast_scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    threshold_kwargs = {"labels": labels} if args.threshold_method in {"roc", "f1"} else {}
    threshold = compute_threshold(
        combined_scores,
        method=args.threshold_method,
        quantile=args.quantile,
        **threshold_kwargs,
    )
    metrics = evaluate_anomalies(combined_scores, labels, threshold)

    recon_th = compute_threshold(recon_scores, method=args.threshold_method, quantile=args.quantile, **threshold_kwargs)
    recon_metrics = evaluate_anomalies(recon_scores, labels, recon_th)
    forecast_th = compute_threshold(forecast_scores, method=args.threshold_method, quantile=args.quantile, **threshold_kwargs)
    forecast_metrics = evaluate_anomalies(forecast_scores, labels, forecast_th)

    z_scores = (combined_scores - combined_scores.mean()) / (combined_scores.std() + 1e-9)
    z_th = 2.0
    z_metrics = evaluate_anomalies(z_scores, labels, z_th)

    summary = {
        "samples": args.samples,
        "num_nodes": args.num_nodes,
        "num_features": args.num_features,
        "horizon": args.horizon,
        "anomaly_prob": args.anomaly_prob,
        "anomaly_scale": args.anomaly_scale,
        "anomaly_span": args.anomaly_span,
        "model_source": model_source,
        "evidence_status": "verified_checkpoint" if model_source == "trained_checkpoint" else "smoke_test_not_evidence",
        "threshold_method": args.threshold_method,
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "methods": {
            "combined": {
                "label": "Ours (Recon+Forecast)",
                "threshold": threshold,
                "metrics": metrics,
            },
            "recon_only": {
                "label": "Recon-only",
                "threshold": recon_th,
                "metrics": recon_metrics,
            },
            "forecast_only": {
                "label": "Forecast-only",
                "threshold": forecast_th,
                "metrics": forecast_metrics,
            },
            "z_score": {
                "label": "Z-Score Baseline",
                "threshold": z_th,
                "metrics": z_metrics,
            },
        },
        "model_path": str(model_path),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Phase 2 evaluation summary saved to:", out_path)
    print("Metrics:", summary["metrics"])


if __name__ == "__main__":
    main()
