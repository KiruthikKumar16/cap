
"""
Multi-Agent PPO Training Script

This script trains a multi-agent system using Proximal Policy Optimization (PPO)
where each intersection is controlled by an independent PPO agent.
"""

import argparse
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.phase1.marl_traffic_env import MARLTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.phase1.reward_calculator import RewardCalculator
import numpy as np
import torch

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent PPO")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.total_timesteps:
        config["training"]["total_timesteps"] = args.total_timesteps
    model_cfg = config["model"]
    reward_cfg = config["reward"]

    # Create the Predictive GNN-RL model
    model = PredictiveGNNRL(
        st_gnn_in_dim=model_cfg["feature_dim"],
        st_gnn_hidden_dim=model_cfg["hidden_dim"],
        st_gnn_heads=model_cfg.get("gat_heads", 2),
        st_gnn_layers=model_cfg["gnn_layers"],
        st_gnn_dropout=model_cfg["dropout"],
        st_gnn_horizon=config.get("data", {}).get("window", {}).get("history", 3),
        rl_gnn_in_dim=model_cfg["feature_dim"],
        rl_gnn_hidden_dim=model_cfg["hidden_dim"],
        rl_gnn_embedding_dim=model_cfg["embedding_dim"],
        rl_gnn_layers=model_cfg["gnn_layers"],
        rl_gnn_type=model_cfg["gnn_type"],
        rl_gnn_heads=model_cfg.get("gat_heads", 2),
        rl_gnn_dropout=model_cfg["dropout"],
    )

    # Create reward calculator
    reward_calculator = RewardCalculator(
        waiting_time_weight=reward_cfg["waiting_time_weight"],
        queue_length_weight=reward_cfg["queue_length_weight"],
        pressure_weight=reward_cfg.get("pressure_weight", 0.0),
        speed_reward_weight=reward_cfg.get("speed_reward_weight", 0.0),
        normalize=reward_cfg.get("normalize", True),
    )

    # Create the multi-agent environment
    def make_env():
        return MARLTrafficEnv(config, model=model, reward_calculator=reward_calculator)
    
    # Create a vectorized environment for parallel processing
    vec_env = make_vec_env(make_env, n_envs=1) # Start with 1 for stability
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./marl_ppo_tensorboard/",
        **config.get("ppo", {}), # Add PPO specific params to config
    )

    model.learn(total_timesteps=config["training"]["total_timesteps"])

    model.save("marl_ppo_traffic")

if __name__ == "__main__":
    main()
