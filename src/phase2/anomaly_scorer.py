"""
Anomaly scoring utilities for ST-GNN autoencoder.

Computes reconstruction and forecasting errors from the
SpatialTemporalAutoencoder outputs and converts them into
per-node anomaly scores that can be fed into Phase 1 reward
shaping or used independently for incident detection.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def reconstruction_error(
    recon: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute reconstruction error between reconstructed and target features.

    Args:
        recon: Reconstructed features, shape [B, N, F]
        target: Target features (typically last step), shape [B, N, F]
        reduction: "none", "mean", or "sum"

    Returns:
        Tensor of reconstruction errors.
        - If reduction == "none": [B, N]
        - Else: scalar tensor
    """
    mse = F.mse_loss(recon, target, reduction="none").mean(dim=-1)  # [B, N]
    if reduction == "none":
        return mse
    if reduction == "mean":
        return mse.mean()
    if reduction == "sum":
        return mse.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def forecasting_error(
    forecast: torch.Tensor,
    target_seq: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute forecasting error over the prediction horizon.

    Args:
        forecast: Forecasted sequence, shape [B, H, N, F]
        target_seq: Target sequence for the same horizon, shape [B, H, N, F]
        reduction: "none", "mean", or "sum"

    Returns:
        Tensor of forecasting errors.
        - If reduction == "none": [B, N]
        - Else: scalar tensor
    """
    # MSE over horizon and feature dimensions
    mse = F.mse_loss(forecast, target_seq, reduction="none").mean(dim=(1, 3))  # [B, N]
    if reduction == "none":
        return mse
    if reduction == "mean":
        return mse.mean()
    if reduction == "sum":
        return mse.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def combined_anomaly_score(
    recon: torch.Tensor,
    forecast: torch.Tensor,
    x_seq: torch.Tensor,
    alpha_recon: float = 0.5,
    alpha_forecast: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined anomaly score from reconstruction and forecasting errors.

    Args:
        recon: Reconstructed last step, [B, N, F]
        forecast: Forecasted sequence, [B, H, N, F]
        x_seq: Input sequence, [B, H, N, F]
        alpha_recon: Weight for reconstruction error
        alpha_forecast: Weight for forecasting error

    Returns:
        scores: Per-node anomaly scores, shape [B, N]
        details: Dict with individual components:
            - "recon_error": [B, N]
            - "forecast_error": [B, N]
    """
    # Target for reconstruction: last step in the input sequence
    target_last = x_seq[:, -1]  # [B, N, F]
    # Target for forecasting: subsequent steps (truncate to horizon)
    horizon = forecast.shape[1]
    target_forecast = x_seq[:, 1 : 1 + horizon]  # [B, H, N, F]

    recon_err = reconstruction_error(recon, target_last, reduction="none")  # [B, N]
    forecast_err = forecasting_error(forecast, target_forecast, reduction="none")  # [B, N]

    scores = alpha_recon * recon_err + alpha_forecast * forecast_err
    details = {
        "recon_error": recon_err,
        "forecast_error": forecast_err,
    }
    return scores, details

