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

### **Phase 3: Real-World Perception (Active R&D)**
- **CV Integration**: YOLOv10-X + BoT-SORT tracker for high-fidelity vehicle detection and tracking.
- **Spatial Metrics**: Perspective transformation logic for converting pixel coordinates to world metrics.

## 🛠️ Setup & Development

### **1. Environment Setup**
```bash
source venv/bin/activate
pip install -r requirements.txt
python scripts/test_setup.py
```

### **2. Development Commands**
```bash
# Verify SUMO Connectivity (Thoothukudi Map)
./venv/bin/python scripts/check_sumo.py --config configs/thoothukudi_verify.yaml

# Evaluate Generalization (Zero-Shot)
./venv/bin/python scripts/evaluate_generalization.py --checkpoint marl_ppo_traffic.zip --episodes 1

# Train Anomaly Detector (Experimental)
python scripts/generate_anomaly_data.py --episodes 3
python src/phase2/anomaly_trainer.py --epochs 10
```

## 🏗️ Repository Structure

- `src/`: Core logic (Environments, Models, Dashboard).
- `src/perception/`: Real-world perception layer (YOLOv10 & CV logic).
- `scripts/`: Development and verification scripts.
- `configs/`: Experiment configurations.
- `data/`: Raw and processed simulation data.

## 📜 Internal Documentation

- [Thoothukudi Map Setup](file:///home/kk/cap/docs/THOOTHUKUDI_SETUP.md)
- [Architecture System Diagram](file:///home/kk/cap/docs/ARCHITECTURE_SYSTEM_DIAGRAM.md)
- [Algorithmic Architecture](file:///home/kk/cap/docs/algorithmic_architecture.png)
