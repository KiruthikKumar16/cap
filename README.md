# Risk-Aware MAPPO Traffic Signal Control

A comprehensive Capstone Project implementing a Multi-Agent Proximal Policy Optimization (MAPPO) architecture bundled with a Spatial-Temporal Graph Neural Network (ST-GNN) Autoencoder.

This system dynamically detects geometrically clustered traffic crashes and adaptively penalizes the Reinforcement Learning agent to prevent grid-lock.

## 1. System Requirements & Installation

You must install **Python 3.9+** and **Eclipse SUMO** (Simulation of Urban MObility) to run the simulation environments.

### Clone the Repository
```powershell
git clone https://github.com/KiruthikKumar16/cap.git
cd cap
```

### Install Python Dependencies
It is highly recommended to use a virtual environment (`venv`).
```powershell
python -m venv venv_gpu
.\venv_gpu\Scripts\activate

# Install requirements (PyTorch, Stable-Baselines3, SUMO-RL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3[extra] sumolib traci torch-geometric pyyaml matplotlib
```

### Configure SUMO Variables
Ensure SUMO is installed on your Windows/Linux machine and mapped to your system `PATH`.
```powershell
# Set SUMO_HOME environment variable (Required for TraCI)
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

---

## 2. Execution & Testing

The project is broken into three phases. You can test each phase independently using the exact wrappers provided in the `scripts/` folder.

### Phase 1: Test Baseline MAPPO vs SOTA Models
Evaluates the baseline MAPPO policy throughput against CoLight and PressLight baseline algorithms using pure JSON benchmarking.
```powershell
python scripts/test_phase1.py
```

### Phase 2: Test Spatial-Temporal Autoencoder
Isolates the trained ST-GNN Autoencoder (`st_gnn_anomaly_detector.pt`) and maps complex collision geometries to generate Precision and F1 anomaly metrics.
```powershell
python scripts/test_phase2.py
```

### Phase 3: Test Dynamic Anomaly Integration
Boots the RL agent with the PyTorch anomaly detector mounted dynamically in parallel. You will directly observe the integration `[AnomalyController]` penalizing traffic routing into geometric accidents in real-time.
```powershell
python scripts/test_phase3.py
```

## Outputs
- Tabular SOTA physical metrics map to `outputs/benchmark_results.json`.
- Visual matplotlib graphs are dynamically generated into `outputs/plots/`.
