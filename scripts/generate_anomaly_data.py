"""
Phase 2 Data Generation Script

Runs the pre-trained MAPPO agent inside the live SUMO simulation to gather
a massive dataset of spatial-temporal matrices [H+1, N, F] for the Anomaly Autoencoder.
Randomly injects "incidents" (lane closures, speed halving) to teach the autoencoder
what catastrophic queues look like in heavily congested geometric environments.
"""

import os
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
import sys

# Ensure root is mapped
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from src.phase1.evaluate import create_environment
import traci

def inject_sumo_anomaly(env, probability=0.01):
    """
    Randomly select an edge and artificially force vehicles to stop
    or drastically reduce speed limit to simulate a crash/lane closure.
    """
    try:
        # 1% chance per step to trigger an anomaly somewhere
        if np.random.rand() < probability:
            edges = traci.edge.getIDList()
            internal_edges = [e for e in edges if not e.startswith(":")]
            target_edge = np.random.choice(internal_edges)
            
            # Halve the speed limit on this edge for massive congestion
            current_speed = traci.edge.getMaxSpeed(target_edge)
            traci.edge.setMaxSpeed(target_edge, max(1.0, current_speed * 0.1))
            print(f"[ANOMALY] Severe accident injected on edge {target_edge}")
    except Exception as e:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default="best_model_stage_2.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--output_file", type=str, default="data/processed/sumo_anomaly_dataset.pt")
    parser.add_argument("--anomaly_prob", type=float, default=0.02, help="Probability of accident per step")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading MAPPO Agent from {args.checkpoint}...")
    env_vectorized = create_environment(config)
    model = PPO.load(args.checkpoint, env=env_vectorized)

    # Extract the internal base environment where history is stored
    base_env = env_vectorized
    while hasattr(base_env, "envs") or hasattr(base_env, "env") or hasattr(base_env, "unwrapped"):
        if hasattr(base_env, "envs"):
            base_env = base_env.envs[0]
        elif hasattr(base_env, "unwrapped") and base_env.unwrapped is not base_env:
            base_env = base_env.unwrapped
        elif hasattr(base_env, "env") and base_env.env is not base_env:
            base_env = base_env.env
        else:
            break

    horizon = 3 # Matches Phase 2 trainer H=3
    dataset_sequences = []

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    for ep in range(args.episodes):
        obs = env_vectorized.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        print(f"--- Generating Episode {ep+1}/{args.episodes} ---")
        
        # Keep a rolling raw features log to build [H+1] sequences
        raw_feature_log = []
        
        for step in range(args.max_steps):
            # Normal action execution
            action, _ = model.predict(obs, deterministic=True)
            step_out = env_vectorized.step(action)
            
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                obs, reward, terminated, info = step_out[0], step_out[1], step_out[2], step_out[3]
                truncated = np.array([False])
            
            # Use raw unflattened node features directly [num_nodes, features]
            raw_node_tensor = base_env._get_raw_observation()
            raw_feature_log.append(raw_node_tensor.clone())
            
            if len(raw_feature_log) >= (horizon + 1):
                # We have enough history for an H+1 sequence
                # sequence shape: [H+1, N, F]
                seq = torch.stack(raw_feature_log[-(horizon+1):], dim=0)
                dataset_sequences.append(seq)
                
            # Randomly trigger TraCI crashes
            inject_sumo_anomaly(env_vectorized, probability=args.anomaly_prob)

            if np.any(terminated) or np.any(truncated):
                break
                
    env_vectorized.close()
    
    if len(dataset_sequences) == 0:
        print("Failed to generate data!")
        return

    # Stack into [B, H+1, N, F]
    final_tensor = torch.stack(dataset_sequences, dim=0)
    print(f"Successfully generated real SUMO dataset: {final_tensor.shape}")
    torch.save(final_tensor, args.output_file)
    print(f"Data saved to -> {args.output_file}")

if __name__ == "__main__":
    main()
