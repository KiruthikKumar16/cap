"""
Training Script for GNN-RL Traffic Control

Main training script for Phase 1: GNN-enhanced DQN agent.
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.phase1.traffic_env import SUMOTrafficEnv
from src.phase1.gnn_encoder import TrafficGNNEncoder, MLPEncoder
from src.phase1.reward_calculator import RewardCalculator
from src.phase1.dqn_agent import create_dqn_agent, TrainingCallback


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(output_dir: Path) -> None:
    """Create output directories."""
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "best_models").mkdir(parents=True, exist_ok=True)


def create_environment(config: Dict[str, Any], traci_port: int = 8813) -> SUMOTrafficEnv:
    """Create SUMO traffic environment. Use different traci_port for train (8813) vs eval (8814)."""
    sumo_cfg = config["sumo"]
    model_cfg = config["model"]
    reward_cfg = config["reward"]
    
    # Create state encoder (GNN or MLP for ablation)
    use_gnn = model_cfg.get("use_gnn", True)
    if use_gnn:
        gnn_encoder = TrafficGNNEncoder(
            in_dim=model_cfg["feature_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            out_dim=model_cfg["embedding_dim"],
            num_layers=model_cfg["gnn_layers"],
            gnn_type=model_cfg["gnn_type"],
            gat_heads=model_cfg.get("gat_heads", 2),
            dropout=model_cfg["dropout"],
        )
    else:
        gnn_encoder = MLPEncoder(
            in_dim=model_cfg["feature_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            out_dim=model_cfg["embedding_dim"],
            num_layers=model_cfg["gnn_layers"],
            dropout=model_cfg["dropout"],
        )
    
    # Create reward calculator (multi-objective: waiting, queue, speed, pressure, throughput)
    reward_calculator = RewardCalculator(
        waiting_time_weight=reward_cfg["waiting_time_weight"],
        queue_length_weight=reward_cfg["queue_length_weight"],
        anomaly_weight=reward_cfg.get("anomaly_weight", 0.0),
        throughput_weight=reward_cfg.get("throughput_weight", 0.0),
        pressure_weight=reward_cfg.get("pressure_weight", 0.0),
        speed_reward_weight=reward_cfg.get("speed_reward_weight", 0.0),
        normalize=reward_cfg.get("normalize", True),
        max_throughput_per_step=reward_cfg.get("max_throughput_per_step", 20.0),
        max_speed=13.89,
    )
    
    # Create environment (traci_port isolates SUMO instances for train vs eval)
    env = SUMOTrafficEnv(
        net_file=sumo_cfg["net_file"],
        route_file=sumo_cfg["route_file"],
        config_file=sumo_cfg.get("config_file"),
        step_length=sumo_cfg["step_length"],
        max_steps=sumo_cfg["simulation_steps"],
        gnn_encoder=gnn_encoder,
        reward_calculator=reward_calculator,
        use_gui=sumo_cfg.get("gui", False),
        traci_port=traci_port,
        sumo_binary=sumo_cfg.get("sumo_binary"),
        time_penalty_per_step=reward_cfg.get("time_penalty_per_step", 0.0),
    )
    
    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN-RL traffic control agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"[OK] Configuration loaded from {args.config}")
    
    # Set random seeds
    seed = config["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directories
    output_dir = Path(config["experiment"]["output_dir"])
    create_output_dirs(output_dir)
    print(f"[OK] Output directories created: {output_dir}")
    
    # Create environment (single env; eval uses same env so only one TraCI connection)
    print("\nCreating environment...")
    env = create_environment(config, traci_port=8813)
    print(f"[OK] Environment created")
    print(f"   Intersections: {env.num_intersections}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Create DQN agent
    print("\nCreating DQN agent...")
    rl_cfg = config["rl"]
    model = create_dqn_agent(env, config=rl_cfg)
    print(f"[OK] DQN agent created")
    
    # Setup callbacks
    training_cfg = config["training"]
    output_cfg = config["output"]
    
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_cfg["save_freq"],
        save_path=output_cfg["checkpoint_dir"],
        name_prefix="dqn_traffic",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (same env; eval reset restarts SUMO, then training continues)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=output_cfg["best_model_dir"],
        log_path=output_cfg["log_dir"],
        eval_freq=training_cfg["eval_freq"],
        n_eval_episodes=training_cfg["eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)
    
    # Training callback
    training_callback = TrainingCallback(
        log_interval=training_cfg["log_interval"],
        verbose=1
    )
    callbacks.append(training_callback)
    
    # Train the model
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Total timesteps: {training_cfg['total_timesteps']}")
    print(f"Checkpoint frequency: {training_cfg['save_freq']}")
    print(f"Evaluation frequency: {training_cfg['eval_freq']}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=callbacks,
        log_interval=training_cfg["log_interval"],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = output_cfg["final_model_path"]
    model.save(final_model_path)
    print(f"\n[OK] Final model saved to: {final_model_path}")
    
    # Close environment
    env.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
