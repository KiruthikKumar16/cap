
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
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a pre-trained model")
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
        rl_gnn_type=model_cfg.get("gnn_type", "gat"),
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

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Move custom model to device
    model = model.to(device)

    # Create Environment (Directly as a VecEnv)
    print(f"Initializing MARL environment with grid size from {config['sumo']['net_file']}...")
    vec_env = MARLTrafficEnv(config, model=model, reward_calculator=reward_calculator)
    
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=vec_env,
            device=device,
            tensorboard_log="./marl_ppo_tensorboard/",
            custom_objects={"model": model} # Pass the GNN model
        )
        # Update PPO parameters from config if they changed
        for key, value in config.get("rl", {}).items():
            if key not in ["algorithm", "policy"]: # Skip non-PPO kwargs
                setattr(model, key, value)
    else:
        # Filter out non-PPO kwargs
        ppo_kwargs = {k: v for k, v in config.get("rl", {}).items() if k not in ["algorithm", "policy"]}
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=device,
            tensorboard_log="./marl_ppo_tensorboard/",
            **ppo_kwargs,
        )

    print("\n" + "="*60)
    print(f"Starting Training ({config['sumo'].get('net_file', 'unknown map')})")
    print(f"Total Timesteps: {config['training']['total_timesteps']}")
    print("="*60 + "\n")

    # Use SB3 progress bar for better visibility in Colab
    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            progress_bar=True
        )
        print("\n[OK] Training finished successfully")
    except Exception as e:
        print(f"\n[ERROR] Training interrupted: {e}")
    finally:
        # Save model
        model.save("marl_ppo_traffic")
        print("[OK] Model saved to marl_ppo_traffic.zip")
        
        # Explicitly close environment to prevent TraCI errors
        print("Closing environment...")
        vec_env.close()
        print("[OK] Environment closed")

if __name__ == "__main__":
    main()
