"""
Evaluation Script for MARL PPO Traffic Signal Control

This script loads a trained model and evaluates its performance on a 10x10 grid.
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO

from src.phase1.marl_traffic_env import MARLTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.phase1.reward_calculator import RewardCalculator

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_mean_reward(
    config_path: str,
    model_path: str,
    episodes: int = 5,
    require_cuda: bool = False,
    verbose: bool = False,
) -> float:
    """
    Run deterministic evaluation and return the mean per-episode average reward
    (mean of total_reward/steps over episodes). Used by curriculum_train gating.
    """
    config = load_config(config_path)
    model_cfg = config["model"]
    reward_cfg = config["reward"]

    gnn_model = PredictiveGNNRL(
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

    reward_calculator = RewardCalculator(
        waiting_time_weight=reward_cfg["waiting_time_weight"],
        queue_length_weight=reward_cfg["queue_length_weight"],
        pressure_weight=reward_cfg.get("pressure_weight", 0.0),
        speed_reward_weight=reward_cfg.get("speed_reward_weight", reward_cfg.get("speed_bonus_weight", 0.0)),
        normalize=reward_cfg.get("normalize", True),
        risk_density_threshold=reward_cfg.get("risk_density_threshold", 0.8),
        risk_penalty_factor=reward_cfg.get("risk_penalty_factor", 1.0),
        risk_sensitivity=reward_cfg.get("risk_sensitivity", 0.5),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and device != "cuda":
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")
    gnn_model = gnn_model.to(device)

    env = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calculator)
    model = PPO.load(
        model_path,
        env=env,
        device=device,
        custom_objects={"model": gnn_model},
    )

    ep_avg_rewards = []
    try:
        for ep in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            if verbose:
                print(f"\n--- Episode {ep + 1} ---")
            while not (isinstance(done, bool) and done) and not (
                isinstance(done, np.ndarray) and any(done)
            ):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += float(np.mean(reward))
                steps += 1
                if verbose and steps % 500 == 0:
                    print(f"Step {steps} | Mean Reward: {np.mean(reward):.4f}")
            if steps > 0:
                ep_avg_rewards.append(total_reward / steps)
            if verbose:
                print(
                    f"Episode {ep + 1} Finished | Total Steps: {steps} | Avg Reward: {ep_avg_rewards[-1]:.4f}"
                )
    finally:
        env.close()

    if not ep_avg_rewards:
        return 0.0
    return float(np.mean(ep_avg_rewards))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Agent PPO")
    parser.add_argument("--config", type=str, default="configs/phase2_10x10.yaml", help="Path to config file")
    parser.add_argument("--model-path", type=str, default="marl_ppo_traffic.zip", help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast if CUDA is not available")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gui:
        config["sumo"]["gui"] = True
    
    model_cfg = config["model"]
    reward_cfg = config["reward"]

    # Recreate the GNN architecture
    gnn_model = PredictiveGNNRL(
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

    reward_calculator = RewardCalculator(
        waiting_time_weight=reward_cfg["waiting_time_weight"],
        queue_length_weight=reward_cfg["queue_length_weight"],
        pressure_weight=reward_cfg.get("pressure_weight", 0.0),
        speed_reward_weight=reward_cfg.get("speed_reward_weight", reward_cfg.get("speed_bonus_weight", 0.0)),
        normalize=reward_cfg.get("normalize", True),
        risk_density_threshold=reward_cfg.get("risk_density_threshold", 0.8),
        risk_penalty_factor=reward_cfg.get("risk_penalty_factor", 1.0),
        risk_sensitivity=reward_cfg.get("risk_sensitivity", 0.5),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False. Activate venv_gpu / install CUDA torch.")
    print(f"Using device: {device}")
    gnn_model = gnn_model.to(device)

    # Create Env
    print(f"Initializing MARL environment for evaluation...")
    env = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calculator)

    # Load PPO model
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(
        args.model_path,
        env=env,
        device=device,
        custom_objects={"model": gnn_model}
    )

    # Evaluation loop
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_waiting_time = []
        episode_queue_length = []

        print(f"\n--- Episode {ep+1} ---")
        while not (isinstance(done, bool) and done) and not (isinstance(done, np.ndarray) and any(done)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += np.mean(reward)
            steps += 1
            
            # Capture stats from info (VecEnv returns list of dicts)
            if info and isinstance(info, list) and len(info) > 0:
                # Prefer step-level SUMO metrics when present
                if "step_total_waiting_time" in info[0]:
                    episode_waiting_time.append(info[0]["step_total_waiting_time"])
                elif "total_waiting_time" in info[0]:
                    episode_waiting_time.append(info[0]["total_waiting_time"])

                if "step_total_queue_length" in info[0]:
                    episode_queue_length.append(info[0]["step_total_queue_length"])
                elif "total_queue_length" in info[0]:
                    episode_queue_length.append(info[0]["total_queue_length"])
            
            if steps % 100 == 0:
                print(f"Step {steps} | Mean Reward: {np.mean(reward):.4f}")

        print(f"Episode {ep+1} Finished | Total Steps: {steps} | Avg Reward: {total_reward/steps:.4f}")
        if episode_waiting_time:
            print(f"Avg Waiting Time: {np.mean(episode_waiting_time):.2f}")
        if episode_queue_length:
            print(f"Avg Queue Length: {np.mean(episode_queue_length):.2f}")

    env.close()
    print("\n[OK] Evaluation completed.")

if __name__ == "__main__":
    main()
