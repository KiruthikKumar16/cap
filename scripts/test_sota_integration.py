import torch
import numpy as np
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.phase1.traffic_env import SUMOTrafficEnv
from src.models.predictive_gnn_rl import PredictiveGNNRL

def test_integration():
    print("--- Starting Integration Test ---")
    
    # 1. Setup a clean config using existing files
    config = {
        'sumo': {
            'net_file': 'data/raw/grid_3x3.net.xml',
            'route_file': 'data/raw/grid_3x3.rou.xml',
            'gui': False,
            'step_length': 1.0,
            'simulation_steps': 100
        },
        'model': {
            'feature_dim': 12,
            'hidden_dim': 32, 
            'st_gnn': {'heads': 1, 'layers': 1, 'dropout': 0.1, 'horizon': 5},
            'rl_gnn': {'layers': 2, 'embedding_dim': 32, 'type': 'GCN', 'heads': 1, 'dropout': 0.1}
        }
    }
    
    # 2. Initialize Unified Model
    # ST-GNN: in_dim=12 -> mean_head outputs feature_dim=12
    # Controller: in_dim=12 (because it receives predicted_state from ST-GNN)
    # The error "mat1 and mat2 shapes cannot be multiplied (9x32 and 12x32)"
    # suggested the GCN was expecting 12 but getting 32, or vice versa.
    # If hidden_dim=32, the first GCN layer (in_dim, hidden_dim) = (12, 32).
    # The predicted_state shape is (9, 12). (9, 12) * (12, 32) -> (9, 32).
    # The next layer (current_dim, out_dim) = (32, 32).
    
    print("[TEST] Initializing Unified Model...")
    model = PredictiveGNNRL(
        st_gnn_in_dim=12,
        st_gnn_hidden_dim=32,
        st_gnn_heads=1,
        st_gnn_layers=1,
        st_gnn_dropout=0.1,
        st_gnn_horizon=5,
        rl_gnn_in_dim=12, # MUST match the feature_dim of the predicted_state (12)
        rl_gnn_hidden_dim=32,
        rl_gnn_embedding_dim=32,
        rl_gnn_layers=2, # Using 2 layers to match the GCN layer dimension transition
        rl_gnn_type='GCN',
        rl_gnn_heads=1,
        rl_gnn_dropout=0.1
    )
    
    # 3. Initialize Environment
    print("[TEST] Initializing Environment...")
    env = SUMOTrafficEnv(
        net_file=config['sumo']['net_file'],
        route_file=config['sumo']['route_file'],
        model=model,
        step_length=config['sumo']['step_length'],
        max_steps=config['sumo']['simulation_steps'],
        use_gui=config['sumo']['gui']
    )
    
    # 4. Test Reset
    print("[TEST] Testing reset()...")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    # embedding_dim = 32. neighbors = 4. total = 32 * (1+4) = 160.
    assert obs.shape[1] == 192, f"Incorrect observation feature dimension: {obs.shape[1]}"
    
    # 5. Test Step & Reward Logic
    print("[TEST] Testing step() and reward logic...")
    num_intersections = getattr(env, "num_intersections", getattr(env, "num_envs", 0))
    actions = np.zeros(num_intersections, dtype=int) 
    obs, reward, terminated, truncated, info = env.step(actions)
    
    print(f"Step Reward: {reward}")
    
    # Check if forecasts were stored in env
    assert hasattr(env, "last_mean_forecast"), "Missing mean forecast storage"
    assert hasattr(env, "last_variance_forecast"), "Missing variance forecast storage"
    
    # Check if metrics are being calculated
    assert "step_total_waiting_time" in info, "Missing waiting time metric"
    assert "step_total_queue_length" in info, "Missing queue length metric"
    
    print("\n[SUCCESS] SOTA Integration Test Passed!")
    env.close()

if __name__ == "__main__":
    try:
        test_integration()
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
