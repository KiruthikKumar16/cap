import argparse
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data.graph_builder import TemporalGraphDataset, build_edge_index, train_val_test_split, window_sequences
from src.data.sumo_sim import SyntheticTrafficSimulator
from src.models.st_gnn import SpatialTemporalAutoencoder
from src.training.train import STGNNLitModule
from src.utils.metrics import compute_threshold, smooth_scores


def _load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_demo_dataset(cfg):
    sim = SyntheticTrafficSimulator(
        timesteps=cfg["data"]["sim"]["timesteps"],
        num_nodes=cfg["data"]["sim"]["num_nodes"],
        feature_dim=cfg["data"]["sim"]["feature_dim"],
        incident_rate=cfg["data"]["sim"]["incident_rate"],
        seed=cfg["experiment"]["seed"],
    )
    features, adjacency, incidents = sim.run()
    windows = window_sequences(features, incidents, cfg["data"]["window"]["history"], cfg["data"]["window"]["horizon"])
    train_w, val_w, test_w = train_val_test_split(
        windows,
        train_split=cfg["data"]["window"]["train_split"],
        val_split=cfg["data"]["window"]["val_split"],
    )
    edge_index = build_edge_index(adjacency)
    test_ds = TemporalGraphDataset(test_w, edge_index)
    return test_ds, features, edge_index


def _load_model(cfg, checkpoint: Path, device: torch.device):
    model = SpatialTemporalAutoencoder(
        in_dim=cfg["data"]["sim"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["gat_heads"],
        layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
        horizon=cfg["data"]["window"]["horizon"],
        use_gru=cfg["model"]["use_gru"],
    )
    lit = STGNNLitModule(
        model=model,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        horizon=cfg["data"]["window"]["horizon"],
    )
    state = torch.load(checkpoint, map_location=device)
    lit.load_state_dict(state["state_dict"])
    lit.eval().to(device)
    return lit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/latest.ckpt")
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.title("Traffic Anomaly Detection (ST-GNN)")
    st.caption("Synthetic demo; replace with SUMO/OSM data for real deployments.")

    test_ds, features, edge_index = _prepare_demo_dataset(cfg)
    loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        model = _load_model(cfg, checkpoint_path, device)
    else:
        st.warning(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
        model = SpatialTemporalAutoencoder(
            in_dim=features.shape[-1],
            hidden_dim=cfg["model"]["hidden_dim"],
            heads=cfg["model"]["gat_heads"],
            layers=cfg["model"]["gnn_layers"],
            dropout=cfg["model"]["dropout"],
            horizon=cfg["data"]["window"]["horizon"],
            use_gru=cfg["model"]["use_gru"],
        ).to(device)

    scores = []
    crit = torch.nn.MSELoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, forecast = model(batch.x, batch.edge_index)
            recon_err = crit(recon, batch.x[:, -1]).mean(dim=(1, 2))
            forecast_err = crit(forecast, batch.y).mean(dim=(1, 2, 3))
            score = (recon_err + forecast_err).cpu().numpy()
            scores.extend(score.tolist())
    scores = smooth_scores(np.array(scores), window=cfg["thresholding"]["smooth_window"])
    threshold = compute_threshold(scores, cfg["thresholding"]["method"], cfg["thresholding"]["quantile"])
    preds = (scores >= threshold).astype(int)

    st.subheader("Anomaly Scores")
    st.line_chart({"score": scores, "threshold": [threshold] * len(scores)})

    st.subheader("Alerts")
    alert_indices = np.where(preds == 1)[0]
    if len(alert_indices) == 0:
        st.success("No anomalies detected in the demo window.")
    else:
        st.error(f"Detected {len(alert_indices)} anomalies at windows: {alert_indices.tolist()}")

    st.caption("Adjust thresholds and retrain for deployment scenarios.")


if __name__ == "__main__":
    main()

