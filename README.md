# Smart Traffic Management System - Unified GNN-RL Framework

A comprehensive, 100% implemented intelligent traffic management system using Graph Neural Networks (GNNs), Spatio-Temporal AI, and Reinforcement Learning. This project integrates adaptive control with proactive anomaly detection and hierarchical coordination.

## 🚀 Project Status: Implementation complete; training/evaluation ongoing

- **Phase 1**: Traffic Control (GNN + DQN/PPO) — implemented; results depend on training convergence
- **Phase 2**: Anomaly Detection (ST-GNN + Bayesian Uncertainty) — implemented; thresholding/evaluation configurable
- **Phase 3**: Proactive Integration (Self-Adaptive Rewards + Risk hooks) — implemented; can be enabled via config

## 🧠 Core Novelties (Patent-Ready)

- **Self-Adaptive Reward Function**: Dynamically adjusts optimization goals based on traffic density, anomaly severity, and time-of-day.
- **Spatio-Temporal Congestion Wave Forecasting**: Predicts future bottleneck zones 5-10 steps ahead using graph-based propagation modeling.
- **Hierarchical Multi-Agent Coordination**: Two-tier control system with local intersection agents and regional zone-level coordinators.
- **Uncertainty-Aware Anomaly Detection**: Uses Bayesian GNNs (Monte Carlo Dropout) to distinguish between real traffic incidents and sensor noise.
- **Risk-Aware Decision Making**: Real-time calculation of Spillback Probability and Accident Likelihood integrated into the RL policy.

## 🛠️ Quick Start

### 1. Environment Setup
Ensure you have **SUMO** installed and `SUMO_HOME` set.
```bash
pip install -r requirements.txt
python scripts/setup_environment.py
python scripts/test_setup.py
```

### 2. Run Integrated Training (Phase 3)
Trains the full proactive system with anomaly-aware rewards.
```bash
python -m src.training.train --config configs/default.yaml
```

### 3. Run Generalization Test (Zero-Shot)
Train on 5x5 and test on 10x10 + Real Bengaluru Map.
```bash
python scripts/run_generalization_test.py
```

### 4. Launch Real-Time Dashboard
Visualize anomalies and control decisions in a web UI.
```bash
streamlit run src/dashboard/app.py -- --config configs/default.yaml --checkpoint outputs/checkpoints/latest.ckpt
```

## 📁 Repo Layout

- `src/phase1/` – Adaptive control logic (GNN + RL)
- `src/phase2/` – Anomaly detection pipeline (ST-GNN)
- `src/phase3/` – Proactive integration, coordination, and risk models
- `src/models/` – Unified GNN architectures (ST-GNN, Predictive-RL)
- `src/data/` – SUMO simulation wrappers and graph builders
- `src/dashboard/` – Streamlit visualization interface
- `scripts/` – Workflow automation (Generalization, Ablation, Benchmarks)
- `configs/` – YAML configurations for different scenarios (5x5, 10x10, Bengaluru)

## 📊 Scientific Validation

The system includes built-in scripts for:
- **Ablation Studies**: Proving the impact of GNN vs. MLP and Anomaly-Awareness.
- **SOTA Benchmarking**: Comparison against **CoLight** and **PressLight**.
- **Performance Figures**: Automated generation of throughput, waiting time, and emission charts.

## ⚖️ License
This project is developed as a Capstone Project. See `PATENT_ANALYSIS.md` for novelty claims.
