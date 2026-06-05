# Robust Multi-Agent Traffic Control Under Non-Stationarity (MAPPO-STGNN) - R&D Stage

[![Status](https://img.shields.io/badge/Status-R%26D_Active-orange)](https://github.com/kk/cap)

This repository contains the ongoing R&D for a **Spatial-Temporal Graph Neural Network (ST-GNN) augmented Multi-Agent Reinforcement Learning (MARL)** system for urban traffic control. The project focuses on handling **Non-Stationarity** (accidents, sensor failures, and demand shifts) through an anomaly-aware proactive adaptation framework.

## 🚧 R&D Development Pipeline

The project is currently in the technical implementation and verification phase:

### **Phase 1: Foundation & Baselines**
- **Diverse Topologies**: Support for grid networks and real-world OSM data (Thoothukudi).
- **Map-Agnostic Inference**: Decentralized, shared-weight MAPPO policy for zero-shot generalization.
- **MARL Training**: Implementation of MAPPO with ST-GNN state forecasting.

### **Phase 2: Anomaly Intelligence (Active R&D)**
- **ST-GNN Detector**: Developing a spatial-temporal autoencoder to learn normal traffic patterns.
- **Incident Injection**: Tools to simulate accidents and sensor corruption for robust testing.

### **Phase 3: Real-World Perception & Resiliency**
- **CV Integration**: YOLOv10-X + BoT-SORT tracker for high-fidelity vehicle detection and tracking.
- **Hardware Safety**: Conflict Monitor Unit (CMU) enforcing NEMA TS2 standards for production deployment.
- **Resiliency Matrix**: Exhaustive testing across Adversarial Perception, Network Latency, and Edge Constraints.

## 🛠️ Setup & Development

### **1. Environment Setup**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/test_setup.py
```

### **2. Research & Production Commands**
```bash
# 🧪 Run the Multi-Agent Resiliency Testing Matrix (Production Audit)
python3 scripts/run_resiliency_matrix.py

# 🛰️ Verify SUMO Connectivity (Thoothukudi Map)
python3 scripts/check_sumo.py --config configs/thoothukudi_verify.yaml

# 🧠 Evaluate Zero-Shot Generalization
python3 scripts/evaluate_generalization.py --checkpoint checkpoints/marl_ppo_traffic.zip

# 👁️ Smoke Test CV-to-RL Bridge (HIL Mode)
python3 src/perception/cv_bridge.py
```

## 📊 Resiliency Testing Matrix

The project implements a **Multi-Agent Evaluation & Resiliency Testing Matrix** to bridge the gap between simulation and the real world:

| Mode | Title | Audit Focus |
| :--- | :--- | :--- |
| **Mode 1** | **Adversarial Perception** | YOLO Occlusion, Sticky Zeros, and Train Gate Blocks. |
| **Mode 2** | **Network Latency** | NTCIP 1202 SNMP Jitter (50ms - 2000ms packet lag). |
| **Mode 3** | **Hardware Safety** | Conflict Monitor Unit (CMU) enforcement of min-green/clearance. |
| **Mode 4** | **Edge Resource** | Jetson AGX Orin VRAM (4.5GB cap) and FPS profiling. |

## 🏗️ Repository Structure

- `src/`: Core implementation logic (Environments, Models, Dashboard).
- `src/perception/`: Real-world perception layer (YOLOv10, BoT-SORT, and CV Bridge).
- `scripts/`: Development, training, and verification scripts.
- `configs/`: Experiment and environment configuration files.
- `data/`: 
  - `data/maps/`: OSM and SUMO network files (Thoothukudi, Grid).
  - `data/signals/`: Traffic signal timing and logic configurations.
- `checkpoints/`: Trained model weights (.zip, .pt) and metadata.
- `docs/`: Technical documentation and system diagrams.
- `papers/`: Reference research papers and background literature.
- `results/`: Runtime-generated metrics, logs, and evaluation charts.
- `rubrics/`: Evaluation rubrics and project templates.

## 📜 Internal Documentation

- [Thoothukudi Map Setup](file:///home/kk/cap/docs/THOOTHUKUDI_SETUP.md)
- [Architecture System Diagram](file:///home/kk/cap/docs/ARCHITECTURE_SYSTEM_DIAGRAM.md)
- [Algorithmic Architecture](file:///home/kk/cap/docs/algorithmic_architecture.png)
