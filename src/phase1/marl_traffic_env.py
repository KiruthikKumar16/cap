
"""
Multi-Agent Traffic Environment for SUMO

This environment provides a multi-agent reinforcement learning setup where each
intersection is controlled by an independent agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.phase1.traffic_env import SUMOTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL
import torch

from stable_baselines3.common.vec_env import VecEnv
from typing import List, Any, Dict, Optional, Tuple, Sequence

class MARLTrafficEnv(VecEnv):
    """
    A multi-agent vectorized environment for SUMO.
    Each intersection is treated as a separate parallel environment sharing the same policy.
    This enables Zero-Shot Generalization across different map sizes.
    """
    def __init__(
        self,
        config: dict,
        model: any = None,
        reward_calculator: any = None
    ):
        # Initialize internal environment
        sumo_cfg = config["sumo"]
        reward_cfg = config.get("reward", {})

        # `SUMOTrafficEnv` requires a `PredictiveGNNRL` model to build observations.
        # If the caller didn't provide one, construct it from the config.
        if model is None:
            model_cfg = config["model"]
            st_gnn_horizon = config.get("data", {}).get("window", {}).get("history", 3)

            model = PredictiveGNNRL(
                st_gnn_in_dim=model_cfg["feature_dim"],
                st_gnn_hidden_dim=model_cfg["hidden_dim"],
                st_gnn_heads=model_cfg.get("gat_heads", 2),
                st_gnn_layers=model_cfg["gnn_layers"],
                st_gnn_dropout=model_cfg["dropout"],
                st_gnn_horizon=st_gnn_horizon,
                rl_gnn_in_dim=model_cfg["feature_dim"],
                rl_gnn_hidden_dim=model_cfg["hidden_dim"],
                rl_gnn_embedding_dim=model_cfg["embedding_dim"],
                rl_gnn_layers=model_cfg["gnn_layers"],
                rl_gnn_type=model_cfg.get("gnn_type", "gat"),
                rl_gnn_heads=model_cfg.get("gat_heads", 2),
                rl_gnn_dropout=model_cfg["dropout"],
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
        
        self.env = SUMOTrafficEnv(
            net_file=sumo_cfg["net_file"],
            route_file=sumo_cfg["route_file"],
            model=model,
            config_file=sumo_cfg.get("config_file"),
            reward_calculator=reward_calculator,
            step_length=sumo_cfg.get("step_length", 1.0),
            max_steps=sumo_cfg.get("simulation_steps", 3600),
            use_gui=sumo_cfg.get("gui", False),
            traci_port=sumo_cfg.get("traci_port", 8813),
            sumo_binary=sumo_cfg.get("sumo_binary"),
            time_penalty_per_step=reward_cfg.get("time_penalty_per_step", 0.0),
            st_gnn_horizon=config.get("data", {}).get("window", {}).get("history", 3),
            enable_anomaly_awareness=config.get("phase3", {}).get("enable_anomaly_awareness", False),
            config=config,
        )
        
        num_agents = self.env.num_agents
        
        # Proposed: Support Map-Agnostic Zero-Shot Generalization
        # The observation space for each agent is already defined in self.env.observation_space (e.g., 192 dims)
        # We use this space for each parallel environment in the VecEnv.
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        
        # Initialize VecEnv with num_agents parallel environments.
        # This enables Shared-Weight Decentralized Execution.
        super().__init__(num_envs=num_agents, observation_space=observation_space, action_space=action_space)
        
        self.actions = None
        self.force_map_agnostic = config["model"].get("force_map_agnostic", False)

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        # obs is already (num_agents, obs_dim) from SUMOTrafficEnv
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs, reward, terminated, truncated, info = self.env.step(self.actions)
        
        # VecEnv expects 'done' (terminated | truncated)
        done = terminated | truncated
        
        # Handle reset if done (SB3 VecEnv automatically resets)
        if any(done):
            # For SUMO, if one is done, all are done since it's a shared simulation
            obs, _ = self.env.reset()
            
        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices: Optional[Sequence[int]] = None) -> List[Any]:
        val = getattr(self.env, attr_name)
        return [val for _ in range(self.num_envs)]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Sequence[int]] = None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: Optional[Sequence[int]] = None, **method_kwargs) -> List[Any]:
        method = getattr(self.env, method_name)
        val = method(*method_args, **method_kwargs)
        return [val for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class: Any, indices: Optional[Sequence[int]] = None) -> List[bool]:
        return [False for _ in range(self.num_envs)]

    def step(self, actions):
        # For compatibility if called directly
        self.step_async(actions)
        return self.step_wait()
