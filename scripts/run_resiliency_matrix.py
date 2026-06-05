import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.phase1.traffic_env import SUMOTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.utils.adversarial_modulator import EnvironmentModulator
from stable_baselines3 import PPO

def run_resiliency_test():
    # 1. Setup paths and config
    config_path = "configs/phase1.yaml"
    checkpoint_path = "checkpoints/marl_ppo_traffic.zip"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Initialize Model
    # Note: Using the same architecture as training
    model_gnn = PredictiveGNNRL(
        in_dim=12,
        hidden_dim=64,
        out_dim=64,
        num_layers=2
    )
    
    # 3. Test Matrix Runner
    modes = {
        0: "Nominal",
        1: "Adversarial Perception",
        2: "Network Latency",
        3: "CMU Safety Stress"
    }
    
    results = []
    modulator = EnvironmentModulator()
    
    print("Starting Resiliency Testing Matrix...")
    
    # Run Mode 0 first to get Nominal baseline for RI calculation
    nominal_tt = 0
    
    for mode_id, mode_name in modes.items():
        print(f"\n>>> Mode {mode_id}: {mode_name}")
        
        # Update config for the specific mode
        test_config = config.copy()
        test_config["test_mode"] = mode_id
        test_config["sumo"]["gui"] = False
        
        # Initialize Env
        env = SUMOTrafficEnv(
            net_file=config["sumo"]["net_file"],
            route_file=config["sumo"]["route_file"],
            model=model_gnn,
            config=test_config,
            max_steps=1000 # Shortened for benchmark speed
        )
        
        # Load PPO Weights
        ppo_agent = PPO.load(checkpoint_path, env=env)
        
        obs, info = env.reset()
        done = False
        total_tt = []
        
        while not done:
            action, _ = ppo_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Use info[0] since it's vectorized
            if "step_travel_time" in info[0]:
                total_tt.append(info[0]["step_travel_time"])
                
            done = terminated[0] or truncated[0]
            
        env.close()
        
        # Compile Metrics
        final_info = info[0]
        mean_tt = float(final_info.get("episode_avg_travel_time", np.mean(total_tt)))
        
        if mode_id == 0:
            nominal_tt = mean_tt
            
        ri = modulator.calculate_resiliency_index(nominal_tt, mean_tt)
        arr = float(final_info.get('action_rejection_rate', 0.0))
        
        results.append({
            "Mode": mode_name,
            "Mean Travel Time": f"{mean_tt:.2f}s",
            "Resiliency Index (RI)": f"{ri:.4f}",
            "Action Rejection Rate": f"{arr:.4f}"
        })

    # 4. Export Results
    df = pd.DataFrame(results)
    print("\n--- FINAL RESILIENCY MATRIX ---")
    # FIX: Ensure we print clean floats, not [object Object]
    for _, row in df.iterrows():
        print(f"Mode: {row['Mode']:25} | RI: {row['Resiliency Index (RI)']} | ARR: {row['Action Rejection Rate']}")
    
    df.to_csv("results/resiliency_matrix_report.csv", index=False)
    print("\n[OK] Resiliency Matrix saved to results/resiliency_matrix_report.csv")

if __name__ == "__main__":
    run_resiliency_test()
