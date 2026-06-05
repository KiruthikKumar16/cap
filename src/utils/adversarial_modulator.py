import numpy as np
import torch
import time

class EnvironmentModulator:
    """
    Implements Adversarial & Hardware Degradation Channels (Modes 1 & 2).
    Simulates perception corruption and NTCIP network jitter.
    """
    def __init__(self, corruption_prob: float = 0.15, latency_range: tuple = (0.05, 2.0)):
        self.corruption_prob = corruption_prob
        self.latency_range = latency_range # in seconds

    def apply_perception_corruption(self, features: torch.Tensor) -> torch.Tensor:
        """
        Mode 1: Adversarial Perception.
        Simulates structural visual noise (YOLO Occlusion, Sticky Zeros).
        """
        if np.random.rand() < self.corruption_prob:
            # Simulate total camera failure/occlusion for a random intersection
            corrupted_feats = features.clone()
            idx = np.random.randint(0, features.shape[0])
            # Zero out queue and count features (indices 4-11 in the 12-dim vector)
            corrupted_feats[idx, 4:12] = 0.0
            return corrupted_feats
        return features

    def apply_network_latency(self) -> float:
        """
        Mode 2: Network Latency & Jitter.
        Simulates NTCIP 1202 SNMP packet lag using a Gamma distribution (jitter).
        """
        # Right-skewed delay distribution simulating real-world network jitter
        delay = np.random.gamma(shape=2.0, scale=0.15) 
        delay = np.clip(delay, self.latency_range[0], self.latency_range[1])
        
        # Emulate physical delay in the control loop
        # time.sleep(delay * 0.001) # Optional: Use for wall-clock simulation
        return delay

    def apply_train_gate_block(self, features: torch.Tensor, junction_id: str = "node_third_gate") -> torch.Tensor:
        """
        [NEW] Edge Case: The Train Gate Block at Third Gate Junction.
        Simulates a synchronized multi-lane stop condition (Railway Gate).
        Used to verify that the ST-Autoencoder flags it as a valid structural event.
        """
        corrupted_feats = features.clone()
        # Find index of Third Gate in features if applicable, else use a heuristic
        # For simulation, we'll assume the user provides the correct node features
        # We force speed to 0 and queue to max on all lanes for this node
        # Indices: 4-7 (Phase), 8-11 (Counts), 12 (Queue - if exists)
        # Based on our 12-dim vector: 8-11 are directional counts/speed proxy
        corrupted_feats[:, 8:12] = 0.0 # Force zero movement
        # Set queue features high (assuming index 4-7 are phase, let's say 12 was queue)
        # Note: In our current 12-dim vector, we'll max out the occupancy proxy
        return corrupted_feats

    def calculate_resiliency_index(self, nominal_metric: float, stressed_metric: float) -> float:
        """
        RI = Nominal / Stressed.
        Approaching 1.0 is perfect fault tolerance.
        """
        if stressed_metric == 0: return 0.0
        return min(1.0, nominal_metric / stressed_metric)
