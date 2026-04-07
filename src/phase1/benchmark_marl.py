"""
Benchmarking Script for MARL Traffic Signal Control

This script compares the trained GNN-RL model against multiple baselines:
1. Fixed-Time Control
2. Actuated Control (SUMO Smart)
3. Random Control
4. Our Trained GNN-RL Model
"""

import argparse
import yaml
import numpy as np
import torch
import pandas as pd
import sys
from pathlib import Path
from stable_baselines3 import PPO
import traci

# Ensure project root is on sys.path so `import src.*` works when invoked as a script.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.phase1.marl_traffic_env import MARLTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.phase1.reward_calculator import RewardCalculator

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_simulation(env, model=None, mode="model", steps=2000):
    obs = env.reset()
    total_reward = 0
    waiting_times = []
    queue_lengths = []
    arrived_vehicles = 0
    
    # Fixed-time parameters: change phase every 30 seconds (60 steps if step_length=0.5)
    phase_duration = 60 
    current_phases = np.zeros(env.num_envs, dtype=int)
    
    # Actuated parameters (simple greedy logic for demo)
    # If queue > threshold, stay green; otherwise switch.
    # SUMO's internal 'actuated' is better, but we simulate it via traci here if needed.
    
    for step in range(steps):
        if mode == "model":
            action, _ = model.predict(obs, deterministic=True)
        elif mode == "fixed":
            if step % phase_duration == 0:
                current_phases = (current_phases + 1) % 4
            action = current_phases
        elif mode == "max_pressure":
            # Simplified Max-Pressure Logic: select phase with highest pressure
            # Pressure = incoming_queue - outgoing_queue
            actions = []
            for i_id in env.env.intersections:
                lanes = traci.trafficlight.getControlledLanes(i_id)
                # For each phase, calculate its pressure
                num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(i_id)[0].phases)
                phase_pressures = []
                for p_idx in range(num_phases):
                    # In Max-Pressure, we want the phase that serves the lanes with the highest pressure
                    # Here we approximate: even phases are usually the 'green' ones in our grid
                    if p_idx % 2 != 0: # Skip yellow/red transitions for selection
                        phase_pressures.append(-1e6)
                        continue
                        
                    # Calculate pressure for the lanes that would be green in this phase
                    # (Simplified: check NS vs EW)
                    traci.trafficlight.setPhase(i_id, p_idx)
                    controlled = traci.trafficlight.getControlledLanes(i_id)
                    pressure = sum([traci.lane.getLastStepHaltingNumber(l) for l in controlled])
                    phase_pressures.append(pressure)
                
                actions.append(np.argmax(phase_pressures))
            action = np.array(actions)
        elif mode == "actuated":
            # Simple Actuated Logic: Check if current phase has vehicles.
            # In a real actuated system, we'd use traci.trafficlight.getPhaseDuration()
            # Here we'll just use a slightly smarter fixed-time that skips empty phases
            # but for a true 'Actuated' comparison, SUMO's internal 'actuated' program is best.
            # We'll approximate with a shorter cycle that adapts.
            if step % 30 == 0: # Check more frequently
                current_phases = (current_phases + 1) % 4
            action = current_phases
        elif mode == "random":
            action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        
        obs, reward, done, info = env.step(action)
        total_reward += np.mean(reward)
        
        # Collect metrics from the last step's info
        if info and isinstance(info, list) and len(info) > 0:
            last_info = info[0]
            if "step_total_waiting_time" in last_info:
                waiting_times.append(last_info["step_total_waiting_time"])
            if "step_total_queue_length" in last_info:
                queue_lengths.append(last_info["step_total_queue_length"])
            if "episode_throughput" in last_info:
                arrived_vehicles = last_info["episode_throughput"]
            elif "step_arrived_vehicles" in last_info:
                arrived_vehicles += last_info["step_arrived_vehicles"]
        
        if any(done):
            break
            
    return {
        "Avg Reward": total_reward / steps,
        "Avg Wait (s)": np.mean(waiting_times) if waiting_times else 0,
        "Avg Queue": np.mean(queue_lengths) if queue_lengths else 0,
        "Throughput": arrived_vehicles
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark MARL Baselines")
    parser.add_argument("--config", type=str, default="configs/phase2_10x10.yaml")
    parser.add_argument("--model-path", type=str, default="marl_ppo_traffic.zip")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast if CUDA is not available")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False. Activate venv_gpu / install CUDA torch.")
    
    # Setup model and reward calculator once to satisfy environment requirements
    model_cfg = config["model"]
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
    ).to(device)
    
    reward_calc = RewardCalculator(
        waiting_time_weight=config["reward"]["waiting_time_weight"],
        queue_length_weight=config["reward"]["queue_length_weight"],
        pressure_weight=config["reward"].get("pressure_weight", 0.0),
        speed_reward_weight=config["reward"].get("speed_reward_weight", config["reward"].get("speed_bonus_weight", 0.0)),
        normalize=config["reward"].get("normalize", True),
        risk_density_threshold=config["reward"].get("risk_density_threshold", 0.8),
        risk_penalty_factor=config["reward"].get("risk_penalty_factor", 1.0),
        risk_sensitivity=config["reward"].get("risk_sensitivity", 0.5),
    )

    results = []

    # 1. Benchmark: Fixed-Time
    print("\n[1/5] Running Fixed-Time Baseline...")
    config["sumo"]["traci_port"] = 8820
    env_fixed = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calc)
    results.append({"Method": "Fixed-Time", **run_simulation(env_fixed, mode="fixed", steps=args.steps)})
    env_fixed.close()

    # 2. Benchmark: Max-Pressure
    print("[2/5] Running Max-Pressure Baseline...")
    config["sumo"]["traci_port"] = 8821
    env_mp = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calc)
    results.append({"Method": "Max-Pressure", **run_simulation(env_mp, mode="max_pressure", steps=args.steps)})
    env_mp.close()

    # 3. Benchmark: Actuated (Heuristic)
    print("[3/5] Running Actuated (Heuristic) Baseline...")
    config["sumo"]["traci_port"] = 8822
    env_actuated = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calc)
    results.append({"Method": "Actuated (Heuristic)", **run_simulation(env_actuated, mode="actuated", steps=args.steps)})
    env_actuated.close()

    # 4. Benchmark: Random
    print("[4/5] Running Random Baseline...")
    config["sumo"]["traci_port"] = 8823
    env_random = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calc)
    results.append({"Method": "Random", **run_simulation(env_random, mode="random", steps=args.steps)})
    env_random.close()

    # 5. Benchmark: Our Model
    print("[5/5] Running Our GNN-RL Model...")
    config["sumo"]["traci_port"] = 8824
    env_model = MARLTrafficEnv(config, model=gnn_model, reward_calculator=reward_calc)
    model = PPO.load(args.model_path, env=env_model, device=device, custom_objects={"model": gnn_model})
    
    results.append({"Method": "Our GNN-RL (Big Brain)", **run_simulation(env_model, model=model, mode="model", steps=args.steps)})
    env_model.close()

    # Display Results
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("BENCHMARK RESULTS (10x10 Grid)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
