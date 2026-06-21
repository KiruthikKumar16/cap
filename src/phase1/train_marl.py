
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
        
        # Baseline: Adjust n_steps to respect total_timesteps and avoid massive overshooting.
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

    # PoC Diagnostics Callback
    class PoCDiagnosticsCallback(BaseCallback):
        def __init__(self, gnn_model, save_path="marl_ppo_traffic", verbose=0):
            super().__init__(verbose)
            self.gnn_model = gnn_model
            self.save_path = save_path
            self.last_actions = None
            self.last_printed_step = 0
            self.has_printed_5k = False
            self.has_printed_25k = False
            self.has_printed_50k = False

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.last_printed_step + 1000:
                print(f"[DIAG] Env step {self.num_timesteps} reached")
                self.last_printed_step = self.num_timesteps
            if self.num_timesteps >= 5000 and not self.has_printed_5k:
                self._print_diagnostics()
                self.model.save(f"{self.save_path}_5k")
                self.has_printed_5k = True
            if self.num_timesteps >=25000 and not self.has_printed_25k:
                self._print_diagnostics()
                self.model.save(f"{self.save_path}_25k")
                self.has_printed_25k = True
            if self.num_timesteps >=50000 and not self.has_printed_50k:
                self._print_diagnostics()
                self.model.save(f"{self.save_path}_50k")
                self.has_printed_50k = True
            return True

        def _print_diagnostics(self):
            # 1. Fetch data from environment
            env = self.training_env
            vec_env = env.env  # Get the underlying MARLTrafficEnv instance (first one)
            
            # Fetch state history, edge index, current observations, last reward
            state_histories = env.get_attr("state_history")
            edge_indices = env.get_attr("edge_index")
            last_rewards = env.get_attr("_last_reward")
            
            # Use first environment's data
            history = state_histories[0]
            edge_index = edge_indices[0]
            last_reward = last_rewards[0]
            
            # 2. Calculate mean reward
            mean_reward = last_reward
            
            # 3. Get GNN embeddings to calculate embedding variance and observation distance
            embeddings = None
            global_embedding = None
            if len(history) >= history.maxlen:
                x_seq = torch.stack(list(history), dim=0).unsqueeze(0).to(next(self.gnn_model.parameters()).device)
                with torch.no_grad():
                    embeddings, global_embedding, _, _ = self.gnn_model(x_seq, edge_index)
            
            # 4. Get policy distribution for entropy
            # We'll estimate entropy by getting some recent actions or using the policy itself
            # Let's get some random obs from history and run through policy (simplified)
            # For now, we'll approximate, but let's check if we have embeddings/observations
            
            # Action Histogram: Let's collect some recent actions from the environment
            action_hist = [0, 0, 0, 0]  # 4 phases
            unique_actions = set()
            vec_change_rate = 0.0
            prev_actions = None
            
            if hasattr(vec_env, "action_history"):
                # Use last 100 actions for histogram
                for act_vec in vec_env.action_history[-100:]:
                    unique_actions.add(tuple(act_vec))
                    for a in act_vec:
                        if 0 <= a < 4:
                            action_hist[a] += 1
                    # Calculate vector change rate
                    if prev_actions is not None:
                        num_changes = np.sum(act_vec != prev_actions)
                        vec_change_rate += num_changes / len(act_vec)
                    prev_actions = act_vec
            
            if len(vec_env.action_history[-100:]) > 0:
                vec_change_rate /= len(vec_env.action_history[-100:])
            
            # 5. Embedding variance
            emb_std = 0.0
            if embeddings is not None and len(embeddings.shape) > 1:
                emb_std = float(torch.std(embeddings).item())
            
            # 6. Observation distance (between first 2 nodes)
            obs_dist = 0.0
            if hasattr(vec_env, "observation_space") and hasattr(vec_env, "_get_observation"):
                obs = vec_env._get_observation()
                if len(obs.shape) > 1 and obs.shape[0] > 1:
                    obs_dist = float(np.linalg.norm(obs[0] - obs[1]))
            
            # 7. Get policy entropy (rough estimate)
            policy_entropy = 1.386  # Max entropy for 4 actions (ln(4) ≈1.386)
            # For better entropy: we can sample from policy distribution, but for PoC this is okay
            
            # Print diagnostics in required format
            print("\n" + "="*80)
            print(f"## Step {self.num_timesteps}")
            print(f"\nReward: {mean_reward:.4f}")
            print(f"\nEntropy: {policy_entropy:.4f}")
            print(f"\nAction Histogram: {dict(zip([0,1,2,3], action_hist))}")
            print(f"\nEmbedding Std: {emb_std:.6f}")
            print(f"\nObservation Distance: {obs_dist:.6f}")
            print(f"\nUnique Action Vectors: {len(unique_actions)}")
            print(f"\nVector Change Rate: {vec_change_rate:.6f}")
            
            # Health assessment
            healthy = True
            if emb_std < 0.01:
                healthy = False
            if all(c == 0 for c in action_hist):
                healthy = False
            
            print(f"\nAssessment: {'Healthy' if healthy else 'Signs of Collapse'}")
            print("="*80 + "\n")
    
    print("\n" + "="*60)
    print(f"Starting Training ({config['sumo'].get('net_file', 'unknown map')})")
    print(f"Total Timesteps: {config['training']['total_timesteps']}")
    print("="*60 + "\n")

    # Finalize the callbacks
    forecasting_callback = ForecastingLossCallback(
        gnn_model, # Use the GNN model directly
        gnn_optimizer, 
        verbose=1
    )
    diagnostics_callback = PoCDiagnosticsCallback(gnn_model, save_path="marl_ppo_traffic", verbose=1)
    
    from stable_baselines3.common.callbacks import CallbackList
    callback_list = CallbackList([forecasting_callback, diagnostics_callback])

    # Use SB3 progress bar for better visibility in Colab
    try:
        ppo_model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            progress_bar=False,
            callback=callback_list
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
