import json
import time
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

class TelemetryAggregator:
    """
    Backend service to aggregate real-time metrics from Perception, Control, 
    and Hardware layers for the frontend dashboard.
    """
    def __init__(self, log_dir: str = "results/telemetry"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_state = {}

    def update_perception_stats(self, intersection_id: str, vision_data: Any):
        """Mode 1 & 3: CV-derived metrics and Digital Twin states."""
        stats = {
            "intersection_id": intersection_id,
            "lane_queues": vision_data.lane_queues,
            "signal_phase": vision_data.current_signal_phase,
            "phase_time": vision_data.phase_elapsed_time,
            "timestamp": time.time()
        }
        self.current_state[f"perception_{intersection_id}"] = stats

    def update_control_diagnostics(self, anomaly_scores: Dict[str, float], attention_weights: np.ndarray):
        """Mode 2: GNN Attention and Autoencoder status."""
        self.current_state["diagnostics"] = {
            "anomaly_scores": anomaly_scores,
            "attention_map": attention_weights.tolist(), # For GAT visualization
            "timestamp": time.time()
        }

    def update_edge_telemetry(self, orin_stats: Dict[str, Any], latency_stack: Dict[str, float]):
        """Mode 3 & 4: Hardware and Network telemetry."""
        self.current_state["edge"] = {
            "hardware": orin_stats,
            "latency_breakdown": latency_stack,
            "timestamp": time.time()
        }

    def update_competitive_scoreboard(self, model_metrics: Dict[str, Dict[str, float]]):
        """Mode 4: Live Parallel-Runner comparison."""
        self.current_state["scoreboard"] = model_metrics

    def get_frontend_payload(self) -> str:
        """Returns the full state as a JSON string for the dashboard."""
        return json.dumps(self.current_state)

    def save_snapshot(self):
        """Saves current state for persistent dashboard sessions."""
        ts = int(time.time())
        with open(self.log_dir / f"state_{ts}.json", "w") as f:
            json.dump(self.current_state, f)

# Global aggregator instance
aggregator = TelemetryAggregator()
