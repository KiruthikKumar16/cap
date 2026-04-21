# Capstone Project: Master Technical Content
## Project Title: Multi-Agent Reinforcement Learning with Spatial-Temporal GNNs for Non-Stationary Traffic Signal Control

---

## 1. Introduction
### 1.1 Background
Urbanization has led to a critical surge in traffic density, rendering traditional fixed-time or simple actuated traffic signal controllers (TSCs) obsolete. Inefficient traffic management contributes to billion-dollar economic losses annually due to wasted fuel, increased carbon emissions, and lost productivity.

### 1.2 Motivation
Traditional RL for traffic control often treats intersections as isolated agents. However, traffic is inherently spatial (intersections affect neighbors) and temporal (past trends predict future surges). **Multi-Agent Reinforcement Learning (MARL)**, specifically **MAPPO (Multi-Agent Proximal Policy Optimization)**, allows for coordinated control across decentralized agents while maintaining global policy coherence.

### 1.3 Problem Scope
This project designs and evaluates an **ST-GNN (Spatial-Temporal Graph Neural Network)** backbone for MAPPO. The scope covers:
- Training on structured 5x5 and 6x6 grid networks.
- **Zero-Shot evaluation** on real-world Bengaluru OpenStreetMap (OSM) topologies.
- Resilience testing under **Non-Stationary Stress** (Simulated accidents and sensor noise).

---

## 2. Project Description and Goals
### 2.1 Literature Review
- **Legacy Baselines**: Classical methods like MaxPressure (2013) provide theoretical stability but fail under high-volatility demand.
- **Deep RL (2018-2020)**: **CoLight** (2019) introduced Graph Attention (GAT) to capture spatial correlations.
- **SOTA (2024-2025)**: **NSTLight** (2025) targets non-stationary environments using advanced differencing features.

### 2.2 Research Gap
Most existing models assume a stationary data distribution. This project addresses the gap in **Zero-Shot Generalization**—where a model trained on a grid must immediately control a complex real-world map (Bengaluru) without retraining.

### 2.3 Objectives
- **Objective 1**: Implement a unified Spatial-Temporal Autoencoder for traffic forecasting.
- **Objective 2**: Integrate the forecaster with a MAPPO controller for risk-aware signal timing.
- **Objective 3**: Benchmark against 2019 and 2025 SOTA systems using 300 DPI professional visualization suite.

---

## 3. Technical Specification
### 3.1 Hardware Requirements
- **Processor**: Intel Core i7 / AMD Ryzen 7 (8+ cores).
- **Memory**: 16GB - 32GB RAM.
- **GPU**: NVIDIA GeForce RTX 30-series or higher (8GB+ VRAM, CUDA 11.8+).
- **Storage**: 500MB for simulation binaries; 2GB+ for training logs and checkpoints.

### 3.2 Software Stack
- **OS**: Windows 10/11 or Ubuntu 20.04+.
- **Simulation**: SUMO (Simulation of Urban MObility) 1.18.0+.
- **Interface**: TraCI (Traffic Control Interface).
- **Deep Learning**: PyTorch 2.1.0+, PyTorch Geometric (PyG).
- **Environment**: Gymnasium / PettingZoo.

---

## 4. System Design & Architecture
### 4.1 Architectural Blueprint
The system follows a **Spatial-Temporal Control Loop**:
1. **Feature Extractor**: Processes raw SUMO telemetry (Queue length, Lane density, Waiting Time).
2. **ST-GNN Encoder**:
   - **GAT Layers**: Captured spatial dependencies between $N$ nodes.
   - **GRU/Transformer**: Encodes temporal history ($\tau = 12$ steps).
3. **Latent Bridge**: Projects 256-dimensional embeddings to the RL controller.
4. **MAPPO Head**: Outputs Multi-Discrete actions representing the optimal green phase.

### 4.2 State Space (Dec-POMDP)
For each agent $i$, the state $s_i$ is defined as:
$s_i = \{ q_{i}, w_{i}, p_{i}, \text{enc}(N_i) \}$
where $q$ is queue length, $w$ is waiting time, $p$ is the current phase, and $\text{enc}(N_i)$ is the graph embedding of the local neighborhood.

### 4.3 Reward Function
The project utilizes a **Self-Adaptive Sigmoid Reward**:
$R = \omega_s \cdot \hat{v} - \sigma(q) \cdot (\omega_q \cdot \hat{q} + \omega_w \cdot \hat{w})$
- $\hat{v}$: Normalized mean speed.
- $\sigma(q)$: Sigmoid density factor (emphasizes penalties as queues grow).
- $\omega$: Configurable weights ($\alpha_1=0.1, \alpha_2=0.05$).

---

## 5. Methodology
### 5.1 Training Strategy: CTDE
The agent follows **Centralized Training with Decentralized Execution**.
- **Training**: Global critic sees the full network state.
- **Execution**: Local actors see only their neighborhood embedding, ensuring scalability.

### 5.2 Adversarial Setup
To simulate real-world imperfections:
- **Sensor Noise**: Gaussian noise $\mathcal{N}(0, \sigma^2)$ injected into queue length observations (5-10% intensity).
- **Accident Injection**: Randomly stopping 5+ vehicles at step 500 to observe gridlock recovery.

---

## 6. Project Implementation
### 6.1 Hyperparameter Table
| Parameter | Value | Description |
| :--- | :--- | :--- |
| Learning Rate | 1.5e-3 | Adam optimizer primary rate |
| PPO Clip | 0.2 | Standard Proximal Policy clipping |
| Hidden Dim | 256 | Latent dimension of ST-GNN |
| GAT Heads | 5 | Multi-head attention count |
| Horizon | 3 | Forecasting steps (3 seconds ahead) |
| Batch Size | 16 | Trajectory batch size |

---

## 7. Results and Discussion
### 7.1 Performance Benchmarks
Based on 100-episode stress testing:
- **MAPPO-STGNN (Proposed)**: ~42,200s total waiting time (Peak Efficiency).
- **NSTLight (2025 SOTA)**: ~71,800s (Solid but lacks global coordination).
- **CoLight (2019 Legacy)**: ~96,500s (Susceptible to congestion waves).

### 7.2 Results Interpretation
- **Heatmap Analysis**: In `heatmap_mappo.png`, the congestion density is uniformly distributed, avoiding "Hotspots" seen in the baselines.
- **Convergence**: `convergence_avg_waiting_time.png` confirms that the proposed model reaches stable efficiency 20% faster than baselines.

---

## 8. Conclusion
### 8.1 Summary
The integration of **Spatial-Temporal Graph Neural Networks** with **MAPPO** provides a robust solution for traffic control. The model's ability to "Forecast" congestion before it arrives allows for preemptive signal clearing, significantly reducing tail-end latency.

### 8.2 Future Work
- **Macroscopic Fundamental Diagrams (MFD)**: Integrating city-wide flow constraints.
- **Hardware-in-the-loop (HIL)**: Deploying the controller on Raspberry Pi / NVIDIA Jetson edge devices for physical lane control.
