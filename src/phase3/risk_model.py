
"""
Congestion Risk Model

This module calculates a congestion risk score based on forecasted traffic density.
"""

import torch

class CongestionRiskModel:
    """
    Calculates a risk score based on the density of vehicles in the forecasted
    traffic state.
    """
    def __init__(self, density_threshold: float = 0.8, risk_penalty_factor: float = 1.0):
        self.density_threshold = density_threshold
        self.risk_penalty_factor = risk_penalty_factor

    def calculate_risk(self, forecasted_state: torch.Tensor) -> float:
        """
        Calculate the congestion risk score.

        Args:
            forecasted_state: The predicted future traffic state from the ST-GNN.

        Returns:
            A scalar risk score.
        """
        # Assuming density is one of the features in the forecasted_state
        # This is a simplified placeholder. A real implementation would need to
        # know the index of the density feature.
        # Let's assume the last feature is density.
        density = forecasted_state[:, :, -1]
        
        # Calculate risk for densities exceeding the threshold
        risk = torch.mean(torch.relu(density - self.density_threshold))
        
        return self.risk_penalty_factor * risk.item()
