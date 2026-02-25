"""
Feature Extraction Module for Traffic Network

Extracts real-time traffic features from SUMO simulation via TraCI API.
Features include signal phases, queue lengths, waiting times, vehicle counts, etc.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("Warning: traci not available. Install SUMO for full functionality.")


class TrafficFeatureExtractor:
    """
    Extracts traffic features from SUMO simulation.
    
    Features extracted per intersection:
        - Signal phase (one-hot encoded, 4 phases)
        - Phase duration (normalized)
        - Queue lengths (sum, max)
        - Waiting time (normalized)
        - Vehicle counts per direction (4 directions)
    
    Total: 12 features per intersection
    """
    
    def __init__(self, intersections: List[str], max_queue: float = 100.0, max_waiting: float = 300.0):
        """
        Initialize feature extractor.
        
        Args:
            intersections: List of intersection IDs (SUMO junction IDs)
            max_queue: Maximum queue length for normalization
            max_waiting: Maximum waiting time for normalization
        """
        self.intersections = intersections
        self.max_queue = max_queue
        self.max_waiting = max_waiting
        self.num_phases = 4  # Typically 4 phases per intersection
        
    def extract(self) -> torch.Tensor:
        """
        Extract features for all intersections.
        
        Returns:
            Feature tensor of shape [num_intersections, feature_dim]
            where feature_dim = 12
        """
        # Use placeholder mode if TraCI not available or not connected
        if not TRACI_AVAILABLE:
            return self._extract_placeholder()
        
        # Try to extract from SUMO, fallback to placeholder on error
        try:
            # Check if we can access TraCI (simple test)
            _ = traci.simulation.getTime()
        except (AttributeError, RuntimeError, Exception):
            # TraCI not connected or not available, use placeholder
            return self._extract_placeholder()
        
        features = []
        
        for intersection_id in self.intersections:
            intersection_features = self._extract_intersection_features(intersection_id)
            features.append(intersection_features)
        
        # Convert to numpy array first to avoid slow tensor conversion warning
        features_array = np.array(features, dtype=np.float32)
        features_tensor = torch.tensor(features_array, dtype=torch.float32)
        return features_tensor
    
    def _extract_intersection_features(self, intersection_id: str) -> np.ndarray:
        """
        Extract features for a single intersection.
        
        Args:
            intersection_id: SUMO junction/traffic light ID
            
        Returns:
            Feature vector of length 12
        """
        feature_vector = np.zeros(12, dtype=np.float32)
        
        try:
            # Get controlled lanes for this intersection
            controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
            
            if not controlled_lanes:
                return feature_vector
            
            # 1. Signal phase (one-hot encoded, indices 0-3)
            current_phase = traci.trafficlight.getPhase(intersection_id)
            phase_idx = current_phase % self.num_phases  # Ensure valid phase index
            feature_vector[phase_idx] = 1.0
            
            # 2. Phase duration (index 4)
            phase_duration = traci.trafficlight.getPhaseDuration(intersection_id)
            # Normalize: assume max duration is 120 seconds
            feature_vector[4] = min(phase_duration / 120.0, 1.0)
            
            # Initialize accumulators
            total_queue = 0.0
            max_queue = 0.0
            total_waiting = 0.0
            vehicle_counts = [0.0] * 4  # 4 directions
            
            # Extract features from each controlled lane
            for lane_idx, lane_id in enumerate(controlled_lanes):
                # Queue length (vehicles stopped)
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                total_queue += queue_length
                max_queue = max(max_queue, queue_length)
                
                # Waiting time
                waiting_time = traci.lane.getWaitingTime(lane_id)
                total_waiting += waiting_time
                
                # Vehicle count
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                direction_idx = lane_idx % 4  # Map to 4 directions
                vehicle_counts[direction_idx] += vehicle_count
            
            # 3. Queue lengths (indices 5-6)
            feature_vector[5] = min(total_queue / self.max_queue, 1.0)  # Sum
            feature_vector[6] = min(max_queue / self.max_queue, 1.0)  # Max
            
            # 4. Waiting time (index 7)
            feature_vector[7] = min(total_waiting / self.max_waiting, 1.0)
            
            # 5. Vehicle counts per direction (indices 8-11)
            # Normalize: assume max 50 vehicles per direction
            max_vehicles_per_direction = 50.0
            for i, count in enumerate(vehicle_counts):
                feature_vector[8 + i] = min(count / max_vehicles_per_direction, 1.0)
            
        except Exception as e:
            print(f"Error extracting features for {intersection_id}: {e}")
            # Return zero vector on error
        
        return feature_vector
    
    def _extract_placeholder(self) -> torch.Tensor:
        """
        Generate placeholder features for testing without SUMO.
        
        Returns:
            Random feature tensor with realistic values
        """
        num_intersections = len(self.intersections)
        features = np.zeros((num_intersections, 12), dtype=np.float32)
        
        # Generate realistic placeholder features
        for i in range(num_intersections):
            # Signal phase (one-hot encoded, indices 0-3)
            phase_idx = np.random.randint(0, 4)
            features[i, phase_idx] = 1.0
            
            # Phase duration (index 4) - normalized, typically 0.2-0.5
            features[i, 4] = np.random.uniform(0.2, 0.5)
            
            # Queue lengths (indices 5-6) - normalized, typically 0.1-0.6
            features[i, 5] = np.random.uniform(0.1, 0.6)  # Sum
            features[i, 6] = np.random.uniform(0.1, 0.5)  # Max
            
            # Waiting time (index 7) - normalized, typically 0.1-0.4
            features[i, 7] = np.random.uniform(0.1, 0.4)
            
            # Vehicle counts per direction (indices 8-11) - normalized, typically 0.2-0.7
            for j in range(8, 12):
                features[i, j] = np.random.uniform(0.2, 0.7)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to [0, 1] range (if not already normalized).
        
        Args:
            features: Feature tensor
            
        Returns:
            Normalized feature tensor
        """
        # Features should already be normalized, but this ensures it
        # Clamp to [0, 1]
        features = torch.clamp(features, 0.0, 1.0)
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features for documentation.
        
        Returns:
            List of feature names
        """
        return [
            "phase_0", "phase_1", "phase_2", "phase_3",  # Signal phases (one-hot)
            "phase_duration",  # Phase duration
            "queue_sum", "queue_max",  # Queue lengths
            "waiting_time",  # Total waiting time
            "vehicles_dir_0", "vehicles_dir_1", "vehicles_dir_2", "vehicles_dir_3"  # Vehicle counts
        ]


def extract_features_from_sumo(intersections: List[str]) -> torch.Tensor:
    """
    Convenience function to extract features from SUMO.
    
    Args:
        intersections: List of intersection IDs
        
    Returns:
        Feature tensor [num_intersections, feature_dim]
    """
    extractor = TrafficFeatureExtractor(intersections)
    features = extractor.extract()
    return features


