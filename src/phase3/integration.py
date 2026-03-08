"""
Phase 3: Integration Module

Connects Phase 1 (GNN+RL traffic control) with Phase 2 (anomaly detection)
to enable anomaly-aware traffic management.
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
from pathlib import Path
from enum import Enum
from collections import deque
import logging

from src.models.st_gnn import SpatialTemporalAutoencoder
from src.phase2.anomaly_scorer import combined_anomaly_score


class AnomalyType(Enum):
    """Types of traffic anomalies."""
    NORMAL = "normal"
    CONGESTION = "congestion"
    ACCIDENT = "accident"
    UNUSUAL_FLOW = "unusual_flow"


class AnomalyAwareTrafficController:
    """
    Integrates Phase 1 traffic control with Phase 2 anomaly detection.

    Provides anomaly scores to the reward function for proactive traffic management.
    Enhanced with multi-type anomaly detection, adaptive thresholds, and explainability.
    """

    def __init__(
        self,
        anomaly_model_path: str,
        device: str = "auto",
        anomaly_threshold: float = 0.5,
        anomaly_weight: float = 0.1,
        enable_anomaly_awareness: bool = True,
        adaptive_threshold: bool = True,
        smoothing_window: int = 5,
        confidence_interval: bool = True,
        multi_anomaly_types: bool = True,
    ):
        """
        Initialize anomaly-aware controller.

        Args:
            anomaly_model_path: Path to trained ST-GNN anomaly detector
            device: Device for anomaly model ("auto", "cpu", "cuda")
            anomaly_threshold: Initial threshold for anomaly detection
            anomaly_weight: Weight for anomaly penalty in reward
            enable_anomaly_awareness: Whether to use anomaly-aware rewards
            adaptive_threshold: Whether to adapt threshold based on history
            smoothing_window: Window size for temporal smoothing
            confidence_interval: Whether to compute confidence intervals
            multi_anomaly_types: Whether to classify anomaly types
        """
        self.anomaly_model_path = Path(anomaly_model_path)
        self.base_threshold = anomaly_threshold
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_weight = anomaly_weight
        self.enable_anomaly_awareness = enable_anomaly_awareness
        self.adaptive_threshold = adaptive_threshold
        self.smoothing_window = smoothing_window
        self.confidence_interval = confidence_interval
        self.multi_anomaly_types = multi_anomaly_types

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load anomaly model
        self.anomaly_model = None
        self._load_anomaly_model()

        # State tracking for temporal sequences
        self.feature_history: List[np.ndarray] = []
        self.max_history_length = 3  # For 3-step horizon

        # Enhanced tracking
        self.score_history = deque(maxlen=100)  # For adaptive threshold
        self.smoothed_scores = {}  # For temporal smoothing
        self.confidence_intervals = {}  # For uncertainty estimation
        self.anomaly_explanations = []  # For explainability

        # Setup logging
        self.logger = logging.getLogger("AnomalyController")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Load anomaly model
        self.anomaly_model = None
        self._load_anomaly_model()

        # State tracking for temporal sequences
        self.feature_history: List[np.ndarray] = []
        self.max_history_length = 3  # For 3-step horizon

    def _load_anomaly_model(self) -> None:
        """Load the trained ST-GNN anomaly detector."""
        if not self.anomaly_model_path.exists():
            print(f"Warning: Anomaly model not found at {self.anomaly_model_path}")
            self.anomaly_model = None
            return

        try:
            # Load model architecture (you may need to adjust these parameters)
            self.anomaly_model = SpatialTemporalAutoencoder(
                in_dim=12,  # Feature dimension (should match Phase 1)
                hidden_dim=64,
                heads=2,
                layers=2,
                dropout=0.1,
                horizon=3,
                use_graph=True,
                temporal_type="gru",
            ).to(self.device)

            # Load trained weights
            state_dict = torch.load(self.anomaly_model_path, map_location=self.device)
            self.anomaly_model.load_state_dict(state_dict)
            self.anomaly_model.eval()

            print(f"Loaded anomaly model from {self.anomaly_model_path}")

        except Exception as e:
            print(f"Error loading anomaly model: {e}")
            self.anomaly_model = None

    def get_anomaly_scores(
        self,
        current_features: np.ndarray,
        edge_index: Optional[torch.Tensor] = None
    ) -> Optional[Dict[str, Dict]]:
        """
        Get enhanced anomaly scores for current traffic state.

        Args:
            current_features: Current traffic features [num_nodes, feature_dim]
            edge_index: Graph edge index for GNN

        Returns:
            Dictionary mapping intersection IDs to anomaly info dicts, or None if unavailable
            Each dict contains: 'score', 'smoothed_score', 'confidence_interval', 'anomaly_type', 'is_anomaly'
        """
        if not self.enable_anomaly_awareness or self.anomaly_model is None:
            return None

        try:
            # Add current features to history
            self.feature_history.append(current_features.copy())
            if len(self.feature_history) > self.max_history_length:
                self.feature_history.pop(0)

            # Need at least horizon+1 steps for prediction
            if len(self.feature_history) < self.max_history_length + 1:
                return {f"intersection_{i}": {
                    'score': 0.0,
                    'smoothed_score': 0.0,
                    'confidence_interval': (0.0, 0.0),
                    'anomaly_type': AnomalyType.NORMAL.value,
                    'is_anomaly': False
                } for i in range(current_features.shape[0])}

            # Prepare input sequence [batch=1, horizon+1, nodes, features]
            sequence = np.stack(self.feature_history[-self.max_history_length-1:], axis=0)
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension

            # Convert to torch tensors
            x_input = torch.from_numpy(sequence[:, :-1]).float().to(self.device)  # [1, H, N, F]
            x_target = torch.from_numpy(sequence[:, -1:]).float().to(self.device)  # [1, 1, N, F]

            if edge_index is None:
                # Create fully connected edge index if not provided
                num_nodes = current_features.shape[0]
                edge_index = self._create_fully_connected_edges(num_nodes).to(self.device)

            # Get anomaly scores
            with torch.no_grad():
                recon, forecast = self.anomaly_model(x_input, edge_index)
                scores, _ = combined_anomaly_score(recon, forecast, x_target)

                # Convert to numpy
                scores_np = scores.squeeze().cpu().numpy()

                # Process each intersection
                anomaly_info = {}
                for i, raw_score in enumerate(scores_np):
                    intersection_id = f"intersection_{i}"

                    # Temporal smoothing
                    smoothed_score = self._apply_temporal_smoothing(intersection_id, raw_score)

                    # Confidence interval
                    ci = self._compute_confidence_interval(intersection_id, raw_score) if self.confidence_interval else (raw_score, raw_score)

                    # Adaptive threshold
                    current_threshold = self._get_adaptive_threshold() if self.adaptive_threshold else self.anomaly_threshold

                    # Multi-anomaly classification
                    anomaly_type = self._classify_anomaly_type(raw_score, smoothed_score, current_features[i])

                    # Determine if anomaly
                    is_anomaly = smoothed_score > current_threshold

                    anomaly_info[intersection_id] = {
                        'score': float(raw_score),
                        'smoothed_score': float(smoothed_score),
                        'confidence_interval': (float(ci[0]), float(ci[1])),
                        'anomaly_type': anomaly_type.value,
                        'is_anomaly': is_anomaly,
                        'threshold': float(current_threshold)
                    }

                    # Update history for adaptive threshold
                    self.score_history.append(raw_score)

                # Log explanations
                self._log_anomaly_explanations(anomaly_info)

                return anomaly_info

        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {e}")
            return None

    def _create_fully_connected_edges(self, num_nodes: int) -> torch.Tensor:
        """Create fully connected edge index for graph."""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.extend([[i, j], [j, i]])  # Bidirectional

        # Remove duplicates and create tensor
        edges = list(set(tuple(edge) for edge in edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index

    def _apply_temporal_smoothing(self, intersection_id: str, current_score: float) -> float:
        """Apply temporal smoothing to anomaly scores."""
        if intersection_id not in self.smoothed_scores:
            self.smoothed_scores[intersection_id] = deque(maxlen=self.smoothing_window)

        self.smoothed_scores[intersection_id].append(current_score)

        # Exponential moving average
        if len(self.smoothed_scores[intersection_id]) == 1:
            return current_score
        else:
            alpha = 0.3  # Smoothing factor
            prev_smoothed = list(self.smoothed_scores[intersection_id])[-2]
            return alpha * current_score + (1 - alpha) * prev_smoothed

    def _compute_confidence_interval(self, intersection_id: str, current_score: float) -> Tuple[float, float]:
        """Compute confidence interval for anomaly score."""
        if intersection_id not in self.confidence_intervals:
            self.confidence_intervals[intersection_id] = []

        self.confidence_intervals[intersection_id].append(current_score)
        if len(self.confidence_intervals[intersection_id]) < 10:
            return (current_score * 0.8, current_score * 1.2)  # Default CI

        scores = np.array(self.confidence_intervals[intersection_id][-20:])  # Last 20 scores
        mean = np.mean(scores)
        std = np.std(scores)

        # 95% confidence interval
        margin = 1.96 * std / np.sqrt(len(scores))
        return (max(0, mean - margin), mean + margin)

    def _get_adaptive_threshold(self) -> float:
        """Compute adaptive threshold based on score history."""
        if len(self.score_history) < 20:
            return self.base_threshold

        scores = np.array(list(self.score_history))
        mean = np.mean(scores)
        std = np.std(scores)

        # Adaptive threshold: mean + 2*std (2-sigma rule)
        adaptive_threshold = mean + 2 * std

        # Smooth the threshold update
        self.anomaly_threshold = 0.9 * self.anomaly_threshold + 0.1 * adaptive_threshold

        return self.anomaly_threshold

    def _classify_anomaly_type(self, raw_score: float, smoothed_score: float, features: np.ndarray) -> AnomalyType:
        """Classify the type of anomaly based on features and scores."""
        if not self.multi_anomaly_types:
            return AnomalyType.NORMAL if smoothed_score <= self.anomaly_threshold else AnomalyType.UNUSUAL_FLOW

        # Extract feature insights (assuming features include queue_length, waiting_time, throughput)
        queue_length = features[0] if len(features) > 0 else 0
        waiting_time = features[1] if len(features) > 1 else 0
        throughput = features[2] if len(features) > 2 else 0

        if smoothed_score > self.anomaly_threshold * 1.5:
            if queue_length > 10 and waiting_time > 50:  # High congestion indicators
                return AnomalyType.CONGESTION
            elif throughput < 2:  # Very low throughput
                return AnomalyType.ACCIDENT
            else:
                return AnomalyType.UNUSUAL_FLOW
        else:
            return AnomalyType.NORMAL

    def _log_anomaly_explanations(self, anomaly_info: Dict[str, Dict]) -> None:
        """Log explanations for anomaly detections."""
        anomalies_detected = [k for k, v in anomaly_info.items() if v['is_anomaly']]

        if anomalies_detected:
            explanation = f"Anomalies detected at: {anomalies_detected}"
            types = [anomaly_info[k]['anomaly_type'] for k in anomalies_detected]
            explanation += f" | Types: {types}"
            scores = [f"{k}: {anomaly_info[k]['smoothed_score']:.3f}" for k in anomalies_detected]
            explanation += f" | Scores: {scores}"

            self.logger.info(explanation)
            self.anomaly_explanations.append({
                'timestamp': np.datetime64('now'),
                'anomalies': anomalies_detected,
                'types': types,
                'scores': {k: anomaly_info[k]['smoothed_score'] for k in anomalies_detected}
            })

    def get_anomaly_penalty(self, anomaly_info: Optional[Dict[str, Dict]]) -> float:
        """
        Calculate enhanced anomaly penalty for reward function.

        Args:
            anomaly_info: Dictionary of anomaly info per intersection

        Returns:
            Penalty value (positive for penalty, will be subtracted from reward)
        """
        if anomaly_info is None or not self.enable_anomaly_awareness:
            return 0.0

        # Calculate weighted penalty based on anomaly types and severity
        total_penalty = 0.0
        anomaly_count = 0

        for intersection, info in anomaly_info.items():
            if info['is_anomaly']:
                # Type-specific weights
                type_multiplier = {
                    AnomalyType.CONGESTION.value: 1.2,
                    AnomalyType.ACCIDENT.value: 1.5,
                    AnomalyType.UNUSUAL_FLOW.value: 1.0
                }.get(info['anomaly_type'], 1.0)

                # Severity based on smoothed score
                severity = max(0, info['smoothed_score'] - info['threshold'])

                # Confidence-based weighting (higher confidence = higher penalty)
                ci_width = info['confidence_interval'][1] - info['confidence_interval'][0]
                confidence_weight = 1.0 / (1.0 + ci_width)  # Lower CI width = higher confidence

                penalty = self.anomaly_weight * severity * type_multiplier * confidence_weight
                total_penalty += penalty
                anomaly_count += 1

        # Average penalty across anomalous intersections
        if anomaly_count > 0:
            return total_penalty / anomaly_count
        else:
            return 0.0

    def is_anomaly_detected(self, anomaly_info: Optional[Dict[str, Dict]]) -> bool:
        """
        Check if any intersection shows anomalous behavior.

        Args:
            anomaly_info: Dictionary of anomaly info per intersection

        Returns:
            True if anomaly detected above threshold
        """
        if anomaly_info is None:
            return False

        return any(info['is_anomaly'] for info in anomaly_info.values())

    def get_anomaly_summary(self) -> Dict:
        """Get summary of anomaly detection statistics."""
        if not self.anomaly_explanations:
            return {'total_anomalies': 0, 'anomaly_types': {}, 'avg_severity': 0.0}

        total_anomalies = len(self.anomaly_explanations)
        type_counts = {}
        severities = []

        for explanation in self.anomaly_explanations:
            for anomaly_type in explanation['types']:
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            severities.extend(explanation['scores'].values())

        return {
            'total_anomalies': total_anomalies,
            'anomaly_types': type_counts,
            'avg_severity': np.mean(severities) if severities else 0.0
        }


# Global instance for easy access
_anomaly_controller: Optional[AnomalyAwareTrafficController] = None


def get_anomaly_controller() -> Optional[AnomalyAwareTrafficController]:
    """Get the global anomaly controller instance."""
    return _anomaly_controller


def init_anomaly_controller(
    model_path: str = "outputs/phase2/st_gnn_anomaly_detector.pt",
    **kwargs
) -> AnomalyAwareTrafficController:
    """
    Initialize the global anomaly controller.

    Args:
        model_path: Path to trained anomaly model
        **kwargs: Additional arguments for AnomalyAwareTrafficController

    Returns:
        Initialized controller
    """
    global _anomaly_controller
    _anomaly_controller = AnomalyAwareTrafficController(
        anomaly_model_path=model_path,
        **kwargs
    )
    return _anomaly_controller
