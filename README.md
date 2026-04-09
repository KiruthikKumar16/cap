<div align="center">

# 🚦 Traffic Resilience Engine
### Risk-Aware Multi-Agent Signal Control via Spatio-Temporal Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![SUMO](https://img.shields.io/badge/Eclipse-SUMO-orange.svg)
![Baseline](https://img.shields.io/badge/Baseline-NSTLight_2025-purple.svg)
![Status](https://img.shields.io/badge/Status-SOTA_Research-brightgreen.svg)

</div>

---

## 📖 Abstract

Urban traffic control breaks when assumptions of stationarity fail. This Capstone reframes signal optimization as a **traffic resilience** problem: policies must sustain performance through accidents, demand shifts, and partial sensor outages rather than only maximize throughput in clean conditions.

The **Traffic Resilience Engine** combines **MAPPO** with a **Spatio-Temporal GNN Autoencoder** and risk-aware reward shaping to model explicitly **non-stationary environments**. The result is a controller that adapts to shock events, preserves flow under adversarial disturbances, and remains competitive on unseen city topology (Bengaluru OSM) without retraining.

---

## 🏗️ Three-Phase Architecture

```
Phase 1 ── ST-GNN Encoder + MAPPO ──▶ Joint Traffic Policy (10×10 Grid)
Phase 2 ── Spatio-Temporal Autoencoder ──▶ Anomaly / Crash Detection
Phase 3 ── Risk-Aware Reward Shaping ──▶ Resilience Under Uncertainty
```

| Component | Role |
|-----------|------|
| `GATv2Conv` Graph Encoder | Maps road topology → dense intersection embeddings |
| ST-GNN Autoencoder | Detects congestion anomalies without human labelling |
| MAPPO Policy Network | Decentralised joint-action optimisation across all agents |
| Risk Penalty Term | Penalises actions that propagate congestion to neighbours |
| NSTLight (2025) | Primary SOTA baseline for degradation benchmarking |

---

## 📁 Repository Structure

```text
📦 cap
 ┣ 📂 configs/             # YAML hyperparameters for all 3 phases
 ┣ 📂 data/raw/            # SUMO grid networks (3×3, 5×5, 6×6, 10×10)
 ┣ 📂 scripts/
 ┃  ┣ accident_injection.py        # Adversarial ghost-vehicle crash simulator
 ┃  ┣ evaluate_generalization.py   # Zero-shot Bengaluru OSM validation
 ┃  ┣ sota_visualizations.py       # Heatmaps, t-SNE, convergence plots
 ┃  ┣ latency_benchmark.py         # CUDA/CPU inference latency (ms/step)
 ┃  ┗ phase1_generate_figures.py   # Full publication figure suite
 ┣ 📂 src/
 ┃  ┣ 📂 baselines/        # NSTLight (2025) baseline package
 ┃  ┣ 📂 models/           # ST-GNN + MAPPO PyTorch modules
 ┃  ┣ 📂 phase1/           # SUMO-TraCI RL environment & training
 ┃  ┣ 📂 phase2/           # Autoencoder anomaly detector
 ┃  ┗ 📂 phase3/           # Risk-aware reward integration
 ┣ 📂 outputs/             # Metrics, checkpoints, visualisation PNGs
 ┣ 📜 SOTA_PROGRESS_REPORT.md
 ┗ 📜 README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```powershell
git clone https://github.com/KiruthikKumar16/cap.git
cd cap
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set SUMO Path
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PYTHONPATH = "C:\path\to\cap"
```

### 3. Train the Model
```powershell
python src/phase1/train_rl.py --config configs/phase1.yaml
```

### 4. Run Adversarial Stress Test
```powershell
# Ghost-vehicle accident injection + Sensor noise benchmarking
python scripts/accident_injection.py
```

### 5. Generate All SOTA Visualisations
```powershell
python scripts/sota_visualizations.py
```

### 6. Run Latency Benchmark (GPU)
```powershell
python scripts/latency_benchmark.py --gpu
```

### 7. Zero-Shot Bengaluru Generalisation Test
```powershell
python scripts/evaluate_generalization.py
```

---

## 📊 SOTA Benchmark Results

> Evaluated on 10×10 synthetic SUMO grid, 10 episodes each.

| Model | Throughput ↑ | Waiting Time ↓ | Under Accident ↑ | Under Noise ↑ |
|---|---|---|---|---|
| **MAPPO + ST-GNN (Ours)** | **847 veh/ep** | **31.4 s** | **83.5%** | **91.2%** |
| NSTLight 2025 | 763 veh/ep | 44.2 s | 55.4% | 72.1% |
| Fixed-Time | 612 veh/ep | 68.7 s | 40.1% | 58.3% |

> *"Performance Retained (%)" = metric vs normal baseline under adversarial conditions.*

---

## 🔬 Adversarial Resilience Framework

Our Phase 3 **Stress-Test Suite** validates robustness where other methods fail:

### 🚗 Accident Injection
```python
# scripts/accident_injection.py
# Freezes 5 vehicles at central junctions at step 500
traci.vehicle.setSpeed(vid, 0.0)
```

### 📡 Sensor Failure Simulation
```python
# 10% of observations randomly zeroed (sensor blackout)
mask = np.random.rand(*obs.shape) < 0.10
obs[mask] = 0.0
```

### 🗺️ Zero-Shot Generalisation
```python
# scripts/evaluate_generalization.py
# Runs the trained model on unseen Bengaluru OSM network
# No retraining — tests weight transfer capability
```

---

## 🖥️ Inference Latency

| Model | Mean (ms/step) | p95 (ms/step) | Real-Time Ready |
|---|---|---|---|
| **MAPPO + ST-GNN (Ours)** | **~2.1 ms** | **~3.4 ms** | ✅ Yes |
| NSTLight 2025 | ~1.8 ms | ~2.9 ms | ✅ Yes |
| Fixed-Time | ~0.1 ms | ~0.2 ms | ✅ Yes |

> Run `python scripts/latency_benchmark.py --gpu` to reproduce on your hardware.

---

## 📈 Key Visualisations

| Chart | Script |
|---|---|
| Congestion Propagation Heatmap | `generate_plots.py` |
| ST-GNN Latent Space (t-SNE) | `generate_plots.py` |
| Reward Convergence Comparison | `generate_plots.py` |
| SOTA Benchmark Dashboard | `sota_visualizations.py` |
| Architecture Flowchart | `phase1_generate_figures.py` |
| Anomaly Detection Metrics | `generate_plots.py` |

---

## 📚 References

- [NSTLight (ACM 2025)](https://dl.acm.org/doi/10.1145/3705754.3705770) — Non-Stationary Traffic Light Control
- [MAPPO](https://arxiv.org/abs/2103.01955) — Multi-Agent PPO
- [GATv2](https://arxiv.org/abs/2105.14491) — Graph Attention Networks v2
- [Eclipse SUMO](https://www.eclipse.org/sumo/) — Simulation of Urban MObility

---

<div align="center">
<sub>Built for the Final Capstone Presentation · Dept. of Computer Science & Engineering · 2026</sub>
</div>
