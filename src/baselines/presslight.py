
"""
PressLight Agent Implementation

This module provides an implementation of the PressLight algorithm, a pressure-based
traffic signal control method.
"""

import numpy as np

class PresslightAgent:
    """
    A simple implementation of the PressLight algorithm.
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def predict(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Predict the action based on the pressure heuristic.
        raw_obs: [num_intersections, num_features]
        Indices 8, 9, 10, 11 are N, S, E, W incoming queue lengths.
        """
        actions = []
        for i in range(len(raw_obs)):
            node_feats = raw_obs[i]
            q_n, q_s, q_e, q_w = node_feats[8:12]
            
            # Simple phase mapping: 
            # 0: GGrr (N-S Green), 1: rrGG (E-W Green), 2: GYrr (N-S Yellow), 3: rrGY (E-W Yellow)
            # Actually, let's assume 4 phases like MaxPressure:
            # 0: N-S Green, 1: E-W Green, 2: N-S Yellow (not used for pressure), 3: E-W Yellow
            
            pressures = [
                q_n + q_s, # Phase 0: North-South
                q_e + q_w, # Phase 1: East-West
                0,         # Phase 2: Yellow
                0          # Phase 3: Yellow
            ]
            actions.append(np.argmax(pressures))
        return np.array(actions)
