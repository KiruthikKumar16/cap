"""
CV-to-RL Bridge: Perception Layer for Real-World Deployment

This module provides the interface between Computer Vision (CV) outputs 
(e.g., YOLOv10 + DeepSORT) and the MARL control logic. It transforms 
raw video-derived metrics into the 12-dimensional feature vector 
expected by the MAPPO-STGNN agent.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import json
from dataclasses import dataclass

@dataclass
class IntersectionVisionData:
    """
    Data structure representing raw detections from a single intersection camera.
    In a real deployment, this would be populated by a YOLOv10 + DeepSORT pipeline.
    """
    intersection_id: str
    lane_counts: Dict[str, int]         # vehicles per lane
    lane_queues: Dict[str, int]         # halting vehicles per lane
    lane_waiting_times: Dict[str, float] # cumulative waiting time per lane
    current_signal_phase: int           # Current phase index from the signal controller
    phase_elapsed_time: float           # Seconds since last phase change

class CVTrafficFeatureExtractor:
    """
    Extracts features from Computer Vision data instead of SUMO TraCI.
    This is the production-ready replacement for TrafficFeatureExtractor.
    """
    
    def __init__(self, intersections: List[str], max_queue: float = 100.0, max_waiting: float = 300.0):
        self.intersections = intersections
        self.max_queue = max_queue
        self.max_waiting = max_waiting
        self.num_phases = 4
        self.feature_dim = 12

    def transform_vision_to_features(self, vision_data_batch: List[IntersectionVisionData]) -> torch.Tensor:
        """
        Transforms a batch of vision-derived metrics into the RL feature tensor.
        
        Args:
            vision_data_batch: List of IntersectionVisionData objects from all cameras.
            
        Returns:
            torch.Tensor of shape [num_intersections, 12]
        """
        features = []
        
        # Sort data to match the order of intersections in the graph
        data_map = {d.intersection_id: d for d in vision_data_batch}
        
        for intersection_id in self.intersections:
            data = data_map.get(intersection_id)
            if data is None:
                # Fallback to zero if data for an intersection is missing (sensor failure)
                features.append(np.zeros(12, dtype=np.float32))
                continue
                
            features.append(self._process_single_intersection(data))
            
        return torch.tensor(np.array(features), dtype=torch.float32)

    def _process_single_intersection(self, data: IntersectionVisionData) -> np.ndarray:
        """
        Maps raw vision detections to the 12-dim RL state vector.
        """
        vec = np.zeros(12, dtype=np.float32)
        
        # 1-4: Signal Phase (One-Hot)
        # In production, this is synced with the Signal Controller (NTCIP 1202)
        phase_idx = data.current_signal_phase % self.num_phases
        vec[phase_idx] = 1.0
        
        # 5: Phase Duration (Normalized)
        vec[4] = min(data.phase_elapsed_time / 120.0, 1.0)
        
        # 6-7: Queue Stats (Sum/Max)
        queues = list(data.lane_queues.values())
        if queues:
            vec[5] = min(sum(queues) / self.max_queue, 1.0)
            vec[6] = min(max(queues) / self.max_queue, 1.0)
            
        # 8: Waiting Time (Normalized)
        wait_times = list(data.lane_waiting_times.values())
        if wait_times:
            vec[7] = min(sum(wait_times) / self.max_waiting, 1.0)
            
        # 9-12: Directional Vehicle Counts (North, East, South, West)
        # CV model provides counts per lane; we aggregate these into 4 cardinal directions
        # This mapping is site-specific during deployment calibration.
        directional_counts = self._aggregate_lane_counts(data.lane_counts)
        for i, count in enumerate(directional_counts):
            vec[8 + i] = min(count / 50.0, 1.0)
            
        return vec

    def _aggregate_lane_counts(self, lane_counts: Dict[str, int]) -> List[float]:
        """
        Example mapping of lane IDs to cardinal directions.
        In a real city, 'lane_0_0' might be 'Northbound Left Turn'.
        """
        counts = [0.0] * 4
        for i, (lane_id, count) in enumerate(lane_counts.items()):
            counts[i % 4] += count
        return counts

class CVDataSimulator:
    """
    Generates synthetic CV detections for testing the perception pipeline.
    Simulates YOLO bounding boxes converted to counts.
    """
    @staticmethod
    def generate_random_vision_data(intersection_ids: List[str]) -> List[IntersectionVisionData]:
        batch = []
        for iid in intersection_ids:
            # Simulate 4-8 lanes per intersection
            lanes = [f"lane_{iid}_{i}" for i in range(np.random.randint(4, 9))]
            batch.append(IntersectionVisionData(
                intersection_id=iid,
                lane_counts={l: np.random.randint(0, 15) for l in lanes},
                lane_queues={l: np.random.randint(0, 10) for l in lanes},
                lane_waiting_times={l: np.random.uniform(0, 50) for l in lanes},
                current_signal_phase=np.random.randint(0, 4),
                phase_elapsed_time=np.random.uniform(5, 60)
            ))
        return batch

if __name__ == "__main__":
    # Smoke test for the CV Bridge
    intersections = ["node_1", "node_2", "node_3"]
    extractor = CVTrafficFeatureExtractor(intersections)
    
    # Simulate real-world perception data
    sim_data = CVDataSimulator.generate_random_vision_data(intersections)
    
    # Transform to RL Features
    rl_features = extractor.transform_vision_to_features(sim_data)
    
    print(f"Vision-to-RL Feature Vector Shape: {rl_features.shape}")
    print(f"Sample Intersection Features:\n{rl_features[0]}")
    assert rl_features.shape == (3, 12)
    print("\n[OK] CV-to-RL Bridge validated.")
