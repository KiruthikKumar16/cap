"""
Predictive Control Module for Phase 3

Implements proactive traffic control by predicting anomalies before they occur,
allowing the RL agent to preemptively adjust traffic signals.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from pathlib import Path
from collections import deque


class AnomalyPredictor:
    """Predicts future anomalies based on current trends."""

    def __init__(
        self,
        history_length: int = 10,
        prediction_horizon: int = 3,
        velocity_threshold: float = 0.1,
    ):
        """
        Initialize anomaly predictor.

        Args:
            history_length: Number of past steps to consider
            prediction_horizon: Number of steps to predict ahead
            velocity_threshold: Threshold for change rate detection
        """
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.velocity_threshold = velocity_threshold

        self.score_history: Dict[str, deque] = {}  # Per-intersection histories
        self.velocity_history: Dict[str, deque] = {}  # Rate of change tracking
        self.predictions: Dict[str, float] = {}

    def update(self, current_scores: Dict[str, float]) -> None:
        """Update history with current anomaly scores."""
        for intersection_id, score in current_scores.items():
            if intersection_id not in self.score_history:
                self.score_history[intersection_id] = deque(
                    maxlen=self.history_length
                )
                self.velocity_history[intersection_id] = deque(
                    maxlen=self.history_length - 1
                )

            # Calculate velocity (rate of change)
            if len(self.score_history[intersection_id]) > 0:
                prev_score = self.score_history[intersection_id][-1]
                velocity = score - prev_score
                self.velocity_history[intersection_id].append(velocity)

            self.score_history[intersection_id].append(score)

    def predict(self) -> Dict[str, Tuple[float, float]]:
        """
        Predict future anomaly scores.

        Returns:
            Dict mapping intersection_id to (predicted_score, confidence)
        """
        predictions = {}

        for intersection_id in self.score_history.keys():
            if len(self.score_history[intersection_id]) < 2:
                predictions[intersection_id] = (0.0, 0.1)  # Low confidence
                continue

            # Linear extrapolation
            scores = list(self.score_history[intersection_id])
            velocities = list(self.velocity_history[intersection_id])

            if len(velocities) > 0:
                avg_velocity = np.mean(velocities)
                last_score = scores[-1]

                # Predict future score
                predicted_score = last_score + avg_velocity * self.prediction_horizon

                # Confidence based on velocity stability
                velocity_std = np.std(velocities) if len(velocities) > 1 else 0.0
                confidence = 1.0 / (1.0 + velocity_std)  # Higher std = lower confidence
            else:
                predicted_score = scores[-1]
                confidence = 0.5

            predictions[intersection_id] = (
                max(0.0, predicted_score),
                float(confidence),
            )

        self.predictions = {k: v[0] for k, v in predictions.items()}

        return predictions

    def should_preempt_anomaly(self, intersection_id: str, current_threshold: float) -> bool:
        """Check if we should preemptively act on a predicted anomaly."""
        if intersection_id not in self.predictions:
            return False
        
        predicted_score = self.predictions[intersection_id]
        return predicted_score > current_threshold


class CongestionWaveForecaster:
    """
    Predicts spatial-temporal congestion wave propagation.
    (Patent Angle: Proactive traffic signal control using spatio-temporal congestion forecasting)
    """
    def __init__(self, propagation_speed: float = 0.5, dissipation_rate: float = 0.1):
        """
        Args:
            propagation_speed: Speed at which congestion spreads to neighbors (0 to 1)
            dissipation_rate: Rate at which congestion clears over time
        """
        self.propagation_speed = propagation_speed
        self.dissipation_rate = dissipation_rate
        self.wave_history: List[torch.Tensor] = []

    def forecast_wave_propagation(
        self, 
        current_density: torch.Tensor, 
        edge_index: torch.Tensor,
        steps: int = 5
    ) -> torch.Tensor:
        """
        Predict future congestion zones by simulating wave propagation on the graph.
        
        Args:
            current_density: Current normalized density per node [N]
            edge_index: Graph connectivity
            steps: How many steps ahead to forecast
            
        Returns:
            Forecasted density map [steps, N]
        """
        num_nodes = current_density.shape[0]
        forecasts = []
        
        state = current_density.clone()
        
        for _ in range(steps):
            # 1. Local Dissipation
            state = state * (1.0 - self.dissipation_rate)
            
            # 2. Spatial Propagation (Wave spreading to neighbors)
            new_state = state.clone()
            row, col = edge_index
            
            # For each edge (u, v), u transfers some "congestion wave" to v
            propagation = state[row] * self.propagation_speed
            new_state.index_add_(0, col, propagation)
            
            # Clip to [0, 1]
            state = torch.clamp(new_state, 0.0, 1.0)
            forecasts.append(state)
            
        return torch.stack(forecasts)

    def identify_future_bottlenecks(
        self, 
        forecasted_waves: torch.Tensor, 
        threshold: float = 0.7
    ) -> List[int]:
        """Identify node indices that will become bottlenecks within the forecast horizon."""
        # Max density reached over the horizon for each node
        max_density, _ = torch.max(forecasted_waves, dim=0)
        bottlenecks = (max_density > threshold).nonzero().squeeze().tolist()
        
        if isinstance(bottlenecks, int):
            return [bottlenecks]
        return bottlenecks or []
        self, intersection_id: str, threshold: float = 0.5
    ) -> bool:
        """
        Determine if we should preemptively control to avoid predicted anomaly.

        Args:
            intersection_id: ID of intersection
            threshold: Anomaly threshold

        Returns:
            True if preemptive action recommended
        """
        if intersection_id not in self.predictions:
            return False

        return self.predictions[intersection_id] > threshold


class PredictiveTrafficController:
    """Uses anomaly predictions to enable proactive traffic control."""

    def __init__(self, anomaly_controller, prediction_horizon: int = 3):
        """
        Initialize predictive controller.

        Args:
            anomaly_controller: Instance of AnomalyAwareTrafficController
            prediction_horizon: Steps ahead to predict
        """
        self.anomaly_controller = anomaly_controller
        self.predictor = AnomalyPredictor(prediction_horizon=prediction_horizon)
        self.preemptive_actions = {}

    def get_preemptive_action(
        self, intersections: List[str], current_scores: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Get preemptive traffic control actions based on predictions.

        Args:
            intersections: List of intersection IDs
            current_scores: Current anomaly scores dict

        Returns:
            Dict mapping intersection_id to recommended action
        """
        # Update predictor with current scores
        current_scores_only = {
            k: v["smoothed_score"] for k, v in current_scores.items()
        }
        self.predictor.update(current_scores_only)

        # Get predictions
        predictions = self.predictor.predict()

        # Determine preemptive actions
        actions = {}
        for intersection_id, (pred_score, confidence) in predictions.items():
            if pred_score > self.anomaly_controller.anomaly_threshold:
                # High anomaly predicted - take preemptive action
                actions[intersection_id] = self._get_action_for_anomaly(
                    intersection_id, pred_score
                )
            else:
                actions[intersection_id] = "normal"

        self.preemptive_actions = actions
        return actions

    def _get_action_for_anomaly(
        self, intersection_id: str, predicted_severity: float
    ) -> str:
        """Determine appropriate preemptive action."""
        if predicted_severity > 0.8:
            return "extend_green"  # Give more time to clear traffic
        elif predicted_severity > 0.6:
            return "balance_phases"  # Balance green time across phases
        elif predicted_severity > 0.4:
            return "prioritize_main"  # Prioritize main flow direction
        else:
            return "normal"

    def get_summary(self) -> Dict:
        """Get summary of predictive control state."""
        return {"preemptive_actions": self.preemptive_actions, "predictions": self.predictor.predictions}
