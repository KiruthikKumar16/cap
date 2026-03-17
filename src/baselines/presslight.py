
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

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Predict the action based on the pressure observation.

        Args:
            obs: The observation, which is assumed to be the pressure at each intersection.

        Returns:
            The action to take for each intersection.
        """
        # The observation is the pressure for each intersection's possible phases.
        # PressLight simply chooses the action (phase) that minimizes the pressure.
        # The observation is expected to be of shape (num_intersections, num_phases)
        return np.argmin(obs, axis=1)
