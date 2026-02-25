import argparse
from pathlib import Path
from typing import Dict, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from src.data.graph_builder import (
    TemporalGraphDataset,
    build_edge_index,
    train_val_test_split,
    window_sequences,
)
from src.data.sumo_sim import SyntheticTrafficSimulator, simulate_with_sumo
from src.models.st_gnn import SpatialTemporalAutoencoder
from src.utils.metrics import compute_threshold, detection_lead_time, evaluate_anomalies, smooth_scores


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _prepare_data(cfg: Dict) -> Tuple[TemporalGraphDataset, TemporalGraphDataset, TemporalGraphDataset, torch.Tensor, torch.Tensor]:
    data_cfg = cfg["data"]
    if data_cfg["mode"] == "synthetic":
        simulator = SyntheticTrafficSimulator(
            timesteps=data_cfg["sim"]["timesteps"],
            num_nodes=data_cfg["sim"]["num_nodes"],
            feature_dim=data_cfg["sim"]["feature_dim"],
            incident_rate=data_cfg["sim"]["incident_rate"],
            seed=cfg["experiment"]["seed"],
        )
        features, adjacency, incidents = simulator.run()
    elif data_cfg["mode"] == "sumo":
        features, adjacency, incidents = simulate_with_sumo(
            net_file=data_cfg["sumo"]["net_file"],
            route_file=data_cfg["sumo"]["route_file"],
            timesteps=data_cfg["sim"]["timesteps"],
            step_length=data_cfg["sumo"]["step_length"],
        )
        if incidents is None:
            incidents = (features[..., 0] < 0).astype(int)  # placeholder labels
    else:
        raise ValueError(f"Unknown data mode: {data_cfg['mode']}")

    history = data_cfg["window"]["history"]
    horizon = data_cfg["window"]["horizon"]
    incidents = incidents if incidents is not None else None
    windows = window_sequences(features, incidents, history, horizon)
    train_w, val_w, test_w = train_val_test_split(
        windows,
        train_split=data_cfg["window"]["train_split"],
        val_split=data_cfg["window"]["val_split"],
    )
    edge_index = build_edge_index(adjacency)
    train_ds = TemporalGraphDataset(train_w, edge_index)
    val_ds = TemporalGraphDataset(val_w, edge_index)
    test_ds = TemporalGraphDataset(test_w, edge_index)
    return train_ds, val_ds, test_ds, edge_index, features


class STGNNLitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        horizon: int,
        mask_ratio: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.horizon = horizon
        self.criterion = nn.MSELoss()
        self.mask_ratio = mask_ratio

    def _mask_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1 - self.mask_ratio))
        return x * mask

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        masked_x = self._mask_input(batch.x)
        recon, forecast = self.forward(masked_x, batch.edge_index)
        loss_recon = self.criterion(recon, batch.x[:, -1])
        loss_forecast = self.criterion(forecast, batch.y)
        loss = loss_recon + loss_forecast
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, forecast = self.forward(batch.x, batch.edge_index)
        loss_recon = self.criterion(recon, batch.x[:, -1])
        loss_forecast = self.criterion(forecast, batch.y)
        loss = loss_recon + loss_forecast
        score = (loss_recon + loss_forecast).detach()
        self.log("val/loss", loss, prog_bar=True)
        return {"val_loss": loss, "score": score}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def _compute_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[list, list]:
    model.eval()
    scores, labels = [], []
    crit = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, forecast = model(batch.x, batch.edge_index)
            recon_err = crit(recon, batch.x[:, -1]).mean(dim=(1, 2))  # [B]
            forecast_err = crit(forecast, batch.y).mean(dim=(1, 2, 3))  # [B]
            score = (recon_err + forecast_err).cpu().numpy()
            scores.extend(score.tolist())
            if hasattr(batch, "incident"):
                labels.extend(batch.incident.max(dim=1).values.cpu().numpy().tolist())
            else:
                labels.extend([0] * len(score))
    return scores, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GNN anomaly detection training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    L.seed_everything(cfg["experiment"]["seed"], workers=True)

    output_dir = Path(cfg["experiment"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    _ensure_dir(checkpoints_dir)

    train_ds, val_ds, test_ds, edge_index, features = _prepare_data(cfg)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True),
        "val": DataLoader(val_ds, batch_size=cfg["training"]["batch_size"]),
        "test": DataLoader(test_ds, batch_size=cfg["training"]["batch_size"]),
    }

    in_dim = features.shape[-1]
    temporal_cfg = cfg["model"]["temporal"]
    model = SpatialTemporalAutoencoder(
        in_dim=in_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["gat_heads"],
        layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
        horizon=cfg["data"]["window"]["horizon"],
        use_graph=cfg["model"]["use_graph"],
        temporal_type=temporal_cfg["type"],
        temporal_heads=temporal_cfg.get("n_heads", 2),
        temporal_ff_mult=temporal_cfg.get("ff_mult", 2),
        temporal_layers=temporal_cfg.get("num_layers", 1),
    )
    lit_model = STGNNLitModule(
        model=model,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        horizon=cfg["data"]["window"]["horizon"],
        mask_ratio=cfg["training"]["input_mask_ratio"],
    )

    device = cfg["training"]["device"]
    trainer = L.Trainer(
        accelerator="auto" if device == "auto" else device,
        devices="auto",
        max_epochs=cfg["training"]["max_epochs"],
        gradient_clip_val=cfg["training"]["grad_clip"],
        logger=CSVLogger(save_dir=output_dir, name="logs"),
        log_every_n_steps=5,
    )

    trainer.fit(lit_model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    ckpt_path = checkpoints_dir / "latest.ckpt"
    trainer.save_checkpoint(ckpt_path)

    # Threshold selection on validation
    device = lit_model.device
    val_scores, val_labels = _compute_scores(lit_model, loaders["val"], device)
    val_scores = smooth_scores(np.array(val_scores), window=cfg["thresholding"]["smooth_window"])
    threshold = compute_threshold(val_scores, cfg["thresholding"]["method"], cfg["thresholding"]["quantile"])

    test_scores, test_labels = _compute_scores(lit_model, loaders["test"], device)
    test_scores = smooth_scores(np.array(test_scores), window=cfg["thresholding"]["smooth_window"])

    metrics = {}
    if len(test_labels) > 0:
        metrics = evaluate_anomalies(np.array(test_scores), np.array(test_labels), threshold)
        preds = (np.array(test_scores) >= threshold).astype(int)
        lead = detection_lead_time(preds, np.array(test_labels))
        if lead is not None:
            metrics["lead_time"] = lead

    summary = {
        "threshold": threshold,
        "metrics": metrics,
        "checkpoint": str(ckpt_path),
    }
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Validation threshold: {threshold:.4f}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()

