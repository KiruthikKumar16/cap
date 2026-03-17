
"""
Multi-Agent Traffic Environment for SUMO

This environment provides a multi-agent reinforcement learning setup where each
intersection is controlled by an independent agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.phase1.traffic_env import SUMOTrafficEnv

class MARLTrafficEnv(gym.Env):
    """
    A multi-agent wrapper for the SUMO traffic environment.
    This wrapper provides a standard Gymnasium interface for PPO training.
    """
    def __init__(
        self,
        config: dict,
        model: any = None,
        reward_calculator: any = None
    ):
        super().__init__()
        sumo_cfg = config["sumo"]
        reward_cfg = config.get("reward", {})
        
        self.env = SUMOTrafficEnv(
            net_file=sumo_cfg["net_file"],
            route_file=sumo_cfg["route_file"],
            model=model,
            reward_calculator=reward_calculator,
            step_length=sumo_cfg.get("step_length", 1.0),
            max_steps=sumo_cfg.get("simulation_steps", 3600),
            use_gui=sumo_cfg.get("gui", False),
            time_penalty_per_step=reward_cfg.get("time_penalty_per_step", 0.0),
            enable_anomaly_awareness=config.get("phase3", {}).get("enable_anomaly_awareness", False)
        )
        self.num_agents = self.env.num_intersections
        
        # Use standard spaces for compatibility with SB3
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        # Use the cached forecast from the inner environment
        forecasted_state = self.env.last_forecast
        
        # Calculate risk penalty
        risk_penalty = self.env.reward_calculator.risk_model.calculate_risk(forecasted_state)
        
        # Subtract risk penalty from the base reward
        reward -= risk_penalty
        
        # Update info with risk metrics for tracking
        info["risk_penalty"] = risk_penalty
        
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
