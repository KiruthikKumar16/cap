"""
Evaluate ST-GNN anomaly detector with synthetic data.

This script runs the autoencoder on synthetic sequences with injected anomalies,
computes anomaly scores, selects a threshold, and reports precision/recall/F1.
"""

import argparse
import json
from pathlib import Path

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
    else:
        print(f"[WARN] Model not found at {model_path}. Using untrained weights.")

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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    edge_index = build_fully_connected_edge_index(args.num_nodes, device)

    all_scores = []
    all_recon_scores = []
    all_forecast_scores = []
    all_z_scores = []
    all_labels = []

    with torch.no_grad():
        for x_plus, labels in loader:
            x_plus = x_plus.to(device)  # [B, H+1, N, F]
            labels = labels.to(device)  # [B, N]
            x_input = x_plus[:, :-1]  # [B, H, N, F]
            recon, forecast = model(x_input, edge_index)
            scores, details = combined_anomaly_score(recon, forecast, x_plus)
            recon_scores = details["recon_error"]
            forecast_scores = details["forecast_error"]

            # Z-score baseline using last step vs sequence mean/std
            x_last = x_plus[:, -1]
            seq_mean = x_input.mean(dim=1)
            seq_std = x_input.std(dim=1) + 1e-6
            z = (x_last - seq_mean).abs() / seq_std
            z_scores = z.mean(dim=-1)  # [B, N]

            all_scores.append(scores.detach().cpu().numpy().reshape(-1))
            all_recon_scores.append(recon_scores.detach().cpu().numpy().reshape(-1))
            all_forecast_scores.append(forecast_scores.detach().cpu().numpy().reshape(-1))
            all_z_scores.append(z_scores.detach().cpu().numpy().reshape(-1))
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))

    scores = np.concatenate(all_scores, axis=0)
    recon_scores = np.concatenate(all_recon_scores, axis=0)
    forecast_scores = np.concatenate(all_forecast_scores, axis=0)
    z_scores = np.concatenate(all_z_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    def _eval_method(method_scores: np.ndarray):
        threshold = compute_threshold(
            method_scores,
            method=args.threshold_method,
            quantile=args.quantile,
            labels=labels,
        )
        metrics = evaluate_anomalies(method_scores, labels, threshold)
        return float(threshold), {k: float(v) for k, v in metrics.items()}

    threshold, metrics = _eval_method(scores)
    recon_th, recon_metrics = _eval_method(recon_scores)
    forecast_th, forecast_metrics = _eval_method(forecast_scores)
    z_th, z_metrics = _eval_method(z_scores)

    summary = {
        "samples": args.samples,
        "num_nodes": args.num_nodes,
        "num_features": args.num_features,
        "horizon": args.horizon,
        "anomaly_prob": args.anomaly_prob,
        "anomaly_scale": args.anomaly_scale,
        "anomaly_span": args.anomaly_span,
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
