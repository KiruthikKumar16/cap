
"""
Multi-Agent Traffic Environment for SUMO

This environment provides a multi-agent reinforcement learning setup where each
intersection is controlled by an independent agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.phase1.traffic_env import SUMOTrafficEnv

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
        
        num_agents = self.env.num_agents
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        
        # Initialize VecEnv
        super().__init__(num_envs=num_agents, observation_space=observation_space, action_space=action_space)
        
        self.actions = None

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
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
