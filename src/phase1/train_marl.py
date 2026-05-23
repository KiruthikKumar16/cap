
"""
Multi-Agent PPO Training Script

This script trains a multi-agent system using Proximal Policy Optimization (PPO)
where each intersection is controlled by an independent PPO agent.
"""

import sys
import argparse
import yaml
from pathlib import Path

# Project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.phase1.marl_traffic_env import MARLTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.models.mappo_policy import MAPPOPolicy
from src.phase1.reward_calculator import RewardCalculator
from src.utils.model_metadata import build_metadata, save_metadata
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
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast if CUDA is not available")
    parser.add_argument("--maps_dir", type=str, default=None, help="Path to directory containing procedural maps")
    parser.add_argument("--use_regional_critics", type=str, default="True", help="Enable Hierarchical Regional Critics (True/False)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.total_timesteps:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.maps_dir:
        config["sumo"]["net_file"] = args.maps_dir
    
    use_regional_critics = args.use_regional_critics.lower() == "true"
    
    model_cfg = config["model"]
    reward_cfg = config["reward"]

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False. Activate venv_gpu / install CUDA torch.")
    print(f"Using device: {device}")

    st_gnn_horizon = config.get("data", {}).get("window", {}).get("history", 3)
    # Create a stable reference to the GNN model and pass it to the environment.
    # NOTE: SB3's feature-extractor mechanism calls the class with (observation_space, ...),
    # which doesn't match `PredictiveGNNRL.__init__`, so we instantiate it manually here.
    gnn_model = PredictiveGNNRL(
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
    ).to(device)

    # Create reward calculator
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
    
    # Create Environment (observations are produced using the GNN model)
    print(f"Initializing MARL environment with grid size from {config['sumo']['net_file']}...")
    vec_env = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calculator)

    # MAPPOPolicy doesn't use SB3's feature extractor output (it defines its own heads),
    # so we keep policy_kwargs empty to avoid SB3 trying to instantiate PredictiveGNNRL.
    policy_kwargs = {}

    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}...")
        # When loading, SB3 automatically rebuilds the policy and feature extractor
        # with the provided policy_kwargs.
        ppo_model = PPO.load(
            args.load_model,
            env=vec_env,
            device=device,
            tensorboard_log="./marl_ppo_tensorboard/",
            policy_kwargs=policy_kwargs
        )
    else:
        # Filter out non-PPO kwargs
        ppo_kwargs = {k: v for k, v in config.get("rl", {}).items() if k not in ["algorithm", "policy"]}
        
        # SOTA: Adjust n_steps to respect total_timesteps and avoid massive overshooting.
        # SB3 PPO collects n_steps * num_envs per update, and won't stop until rollout is finished.
        num_envs = vec_env.num_envs
        total_timesteps = config["training"]["total_timesteps"]
        if "n_steps" in ppo_kwargs:
            rollout_size = ppo_kwargs["n_steps"] * num_envs
            if rollout_size > total_timesteps:
                new_n_steps = max(1, total_timesteps // num_envs)
                print(f"  [Config] Clipping n_steps: {ppo_kwargs['n_steps']} -> {new_n_steps} (total_timesteps={total_timesteps}, agents={num_envs})")
                ppo_kwargs["n_steps"] = new_n_steps

        ppo_model = PPO(
            MAPPOPolicy, # Use custom MAPPO policy
            vec_env,
            verbose=1,
            device=device,
            tensorboard_log="./marl_ppo_tensorboard/",
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
        )

    # Optimizer for forecasting loss, using the stable GNN reference.
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-4)

    # Custom Callback for Forecasting Loss
    from stable_baselines3.common.callbacks import BaseCallback
    class ForecastingLossCallback(BaseCallback):
        def __init__(self, gnn_model, optimizer, verbose=0):
            super().__init__(verbose)
            self.gnn_model = gnn_model
            self.optimizer = optimizer

        def _on_step(self) -> bool:
            # Run forecasting update every 100 steps
            if self.n_calls % 100 == 0:
                # 1. Fetch training data via standard SB3 VecEnv methods
                # This bypasses all wrapper layers safely
                state_histories = self.training_env.get_attr("state_history")
                edge_indices = self.training_env.get_attr("edge_index")
                
                # We only need data from the first parallel env for the GNN update
                history = state_histories[0]
                edge_index = edge_indices[0].to(device)
                
                if len(history) >= history.maxlen:
                    # Prepare input sequence: [B, H, N, F]
                    x_seq = torch.stack(list(history), dim=0).unsqueeze(0).to(device)
                    
                    # 2. Get latent forecast from model
                    _, _, mean_forecast, _ = self.gnn_model(x_seq, edge_index)
                    
                    # 3. Get ground truth from environment
                    actual_current_list = self.training_env.env_method("_get_raw_observation")
                    actual_current = actual_current_list[0].unsqueeze(0).to(device)
                    
                    # 4. Calculate loss (internally decodes latent 256 -> physical 12)
                    loss = self.gnn_model.compute_forecasting_loss(mean_forecast, actual_current)
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if self.verbose > 0:
                        print(f"  [GNN] Step {self.n_calls} | Forecast Loss: {loss.item():.6f}")
            return True
    
    print("\n" + "="*60)
    print(f"Starting Training ({config['sumo'].get('net_file', 'unknown map')})")
    print(f"Total Timesteps: {config['training']['total_timesteps']}")
    print("="*60 + "\n")

    # Finalize the callback with the correct model reference
    forecasting_callback = ForecastingLossCallback(
        gnn_model, # Use the GNN model directly
        gnn_optimizer, 
        verbose=1
    )

    # Use SB3 progress bar for better visibility in Colab
    try:
        ppo_model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            progress_bar=True,
            callback=forecasting_callback
        )
        print("\n[OK] Training finished successfully")
    except Exception as e:
        print(f"\n[ERROR] Training interrupted: {e}")
    finally:
        # Save model
        ppo_model.save("marl_ppo_traffic")
        print("[OK] Model saved to marl_ppo_traffic.zip")
        metadata = build_metadata(
            algorithm="PPO",
            checkpoint_path="marl_ppo_traffic.zip",
            config=config,
            observation_space_repr=str(vec_env.observation_space),
            action_space_repr=str(vec_env.action_space),
        )
        meta_path = save_metadata(metadata, Path("marl_ppo_traffic.zip"))
        print(f"[OK] Metadata saved to {meta_path}")
        
        # Explicitly close environment to prevent TraCI errors
        print("Closing environment...")
        vec_env.close()
        print("[OK] Environment closed")

if __name__ == "__main__":
    main()
