
"""
Congestion Risk Model

This module calculates a congestion risk score based on forecasted traffic density.
"""

import torch

class CongestionRiskModel:
    """
    Calculates a risk score based on the density of vehicles in the forecasted
    traffic state, incorporating uncertainty.
    """
    def __init__(
        self, 
        density_threshold: float = 0.8, 
        risk_penalty_factor: float = 1.0, 
        risk_sensitivity: float = 0.5,
        spillback_threshold: float = 0.9,
        accident_sensitivity: float = 0.2
    ):
        self.density_threshold = density_threshold
        self.risk_penalty_factor = risk_penalty_factor
        self.risk_sensitivity = risk_sensitivity
        self.spillback_threshold = spillback_threshold
        self.accident_sensitivity = accident_sensitivity

    def calculate_risk(self, mean_forecast: torch.Tensor, variance_forecast: torch.Tensor) -> float:
        """
        Calculate a multi-faceted probabilistic risk score.
        (Patent Angle: Risk-aware decision making using probabilistic congestion and spillback forecasting)
        
        Args:
            mean_forecast: The predicted future traffic state [B, N, F]
            variance_forecast: The predicted variance [B, N, F]
        """
        # Feature Mapping (based on src/phase1/feature_extractor.py):
        # 5: total_queue_length (normalized)
        # 7: total_waiting_time (normalized)
        # 8: mean_speed (normalized)
        # 9: vehicle_count (normalized)
        
        # Use ellipsis to always target the last dimension (Features)
        # Works for both [B, N, F] and [B, H, N, F] shapes
        queue_len = mean_forecast[..., 5]
        waiting_time = mean_forecast[..., 7]
        mean_speed = mean_forecast[..., 8]
        veh_count = mean_forecast[..., 9]
        
        # 1. Probabilistic Congestion Risk
        density_proxy = 0.7 * queue_len + 0.3 * waiting_time
        congestion_risk = torch.mean(torch.relu(density_proxy - self.density_threshold))
        
        # 2. Congestion Spillback Probability (Patent Angle)
        # Likelihood that the queue exceeds intersection capacity
        spillback_prob = torch.mean(torch.sigmoid((queue_len - self.spillback_threshold) * 10))
        
        # 3. Accident Likelihood (Patent Angle)
        # High density + High Speed Variance (using forecasted variance as a proxy for turbulence)
        # Also penalized if speed is high while count is high (risky flow)
        speed_variance = variance_forecast[..., 8]
        accident_risk = torch.mean(veh_count * speed_variance * self.accident_sensitivity)
        
        # 4. Uncertainty Penalty
        # Total model uncertainty across all critical features
        uncertainty = torch.mean(variance_forecast[..., [5, 7, 8]])
        
        # Unified Risk Score
        total_risk = (
            1.0 * congestion_risk + 
            0.5 * spillback_prob + 
            0.3 * accident_risk + 
            self.risk_sensitivity * uncertainty
        )
        
        return self.risk_penalty_factor * total_risk.item()
