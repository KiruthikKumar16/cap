"""
Functional Max Pressure (Greedy Queue) baseline agent.
Selects the phase that serves the most vehicles based on raw queue features.
"""

import numpy as np
import torch

class MaxPressureAgent:
    """
    Greedy MaxPressure proxy that selects signal phases based on 
    the direction with the highest traffic volume.
    """
    def __init__(self):
        pass

    def predict(self, observations, deterministic=True):
        """
        Input obs: [B, N, F] or [N, F]. 
        F=12 features from TrafficFeatureExtractor.
        Indices 8-11: vehicle counts in 4 directions.
        """
        if torch.is_tensor(observations):
            obs = observations.cpu().numpy()
        else:
            obs = np.array(observations)

        # Handle [B, N, F] vs [N, F]
        if len(obs.shape) == 3:
            # Batch mode from VecEnv
            batch_actions = []
            for b in range(obs.shape[0]):
                actions = self._get_actions_for_grid(obs[b])
                batch_actions.append(actions)
            return torch.tensor(np.array(batch_actions))
        else:
            # Single env [N, F]
            actions = self._get_actions_for_grid(obs)
            return torch.tensor(actions)

    def _get_actions_for_grid(self, grid_obs):
        """
        grid_obs: [N, 12]
        Returns: [N] actions
        """
        num_intersections = grid_obs.shape[0]
        actions = []
        for i in range(num_intersections):
            # Directions: 0=N, 1=S, 2=E, 3=W (Simplified mapping)
            # Typically Phase 0/2 serve pairs (N-S) and (E-W).
            counts = grid_obs[i, 8:12] # [dir0, dir1, dir2, dir3]
            ns_pressure = counts[0] + counts[1]
            ew_pressure = counts[2] + counts[3]
            
            # Choose phase based on highest pressure
            if ns_pressure > ew_pressure:
                # Phase 0 usually serves N-S in many SUMO grid defaults
                actions.append(0)
            else:
                # Phase 2 usually serves E-W
                actions.append(2)
        return np.array(actions, dtype=np.int32)
