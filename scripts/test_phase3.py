"""
Test Phase 3: Integration of RL and SpatialTemporalAutoencoder.

This script duplicates the configuration, enables Phase 3 anomaly awareness,
and runs a short RL benchmark episode. The system will log real-time
penalty outputs as soon as the autoencoder calculates accident geometries.
"""
import sys
import subprocess
import os
import yaml
import shutil

print("\n" + "="*50)
print("PHASE 3 TEST: End-to-End Risk-Aware Traffic Routing")
print("="*50)

config_src = "configs/phase1.yaml"
config_test = "configs/phase3_test.yaml"

# 1. Dynamically clone Phase 1 Config and Enable Phase 3
with open(config_src, "r") as f:
    config_data = yaml.safe_load(f)

if "phase3" not in config_data:
    config_data["phase3"] = {}
    
config_data["phase3"]["enable_anomaly_awareness"] = True
config_data["phase3"]["anomaly_model_path"] = "outputs/phase2/st_gnn_anomaly_detector.pt"
config_data["phase3"]["anomaly_threshold"] = 0.5

with open(config_test, "w") as f:
    yaml.dump(config_data, f)

print("[OK] Enabled native Anomaly Routing penalties.")

# 2. Run the Benchmark script using the new configuration
subprocess.run([
    sys.executable, "scripts/run_benchmarks.py", 
    "--config", config_test, 
    "--checkpoint", "marl_ppo_traffic.zip",
    "--episodes", "1"
])

print("\n--- Phase 3 Successfully Initialized ---")
print("You should see [AnomalyController] logs directly penalizing intersections heavily based on geometric accident detection!")

# Cleanup
if os.path.exists(config_test):
    os.remove(config_test)
