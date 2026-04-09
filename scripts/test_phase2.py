"""
Test Phase 2: Formally evaluates the SpatialTemporalAutoencoder on tracking geometric crashes.
"""
import sys
import subprocess
import os

print("\n" + "="*50)
print("PHASE 2 TEST: Traffic Anomaly Detection (ST-GNN)")
print("="*50)

# Temporarily inject PYTHONPATH
env_copy = os.environ.copy()
env_copy["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Excute the evaluation metric extractor
subprocess.run([
    sys.executable, "src/phase2/evaluate_anomaly.py", 
    "--model", "outputs/phase2/st_gnn_anomaly_detector.pt",
    "--samples", "200"
], env=env_copy)

print("\n[OK] Phase 2 Execution Completed. Check outputs/phase2/anomaly_eval_summary.json for F1 confidence bounds.")
