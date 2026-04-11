
import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.phase1.train_rl import create_environment
from stable_baselines3 import PPO

def collect_data(config_path, checkpoint_path, output_file, episodes=2):
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating environment...")
    env = create_environment(config)
    
    print(f"Loading model from {checkpoint_path}...")
    model = PPO.load(checkpoint_path, env=env)
    
    all_sequences = []
    horizon = config.get("data", {}).get("window", {}).get("history", 3)
    
    print(f"Collecting data for {episodes} episodes...")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < 3600:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Extract raw node features from environment
            # Robustly reach into SUMOTrafficEnv
            inner_env = env
            if hasattr(inner_env, "env"): # For MARLTrafficEnv
                inner_env = inner_env.env
            while hasattr(inner_env, "unwrapped") and inner_env.unwrapped is not inner_env:
                inner_env = inner_env.unwrapped
            
            raw_obs = inner_env._get_raw_observation() # [N, F]
            all_sequences.append(raw_obs.clone())
            
            step += 1
            done = np.any(done)
            if step % 100 == 0:
                print(f"  Episode {ep} | Step {step}")
                
    # Stack into sequences of length H+1 for Phase 2 training
    # Actually, the anomaly trainer expects [B, H+1, N, F]
    # We'll just save the raw node features [T, N, F] and let a utility script slice them
    
    final_data = torch.stack(all_sequences) # [T, N, F]
    
    # Slice into [B, H+1, N, F]
    B = len(final_data) - (horizon + 1)
    if B <= 0:
        print("Not enough data collected!")
        return
        
    training_samples = []
    for i in range(B):
        training_samples.append(final_data[i : i + horizon + 1])
    
    training_data = torch.stack(training_samples)
    print(f"Shape of collected training data: {training_data.shape}")
    
    torch.save(training_data, output_file)
    print(f"[OK] Saved real traffic data to {output_file}")
    env.close()

if __name__ == "__main__":
    collect_data(
        config_path="configs/phase1.yaml",
        checkpoint_path="marl_ppo_traffic.zip",
        output_file="data/raw/real_traffic_trajectories.pt",
        episodes=1 # 1 episode = 3600 steps = plenty of samples
    )
