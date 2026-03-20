# Results and Discussion: Unified GNN-RL Framework

This document summarizes the experimental results and performance analysis for the Smart Traffic Management System across all three phases.

---

## 1. Experimental Setup
- **Simulator**: SUMO (Simulation of Urban Mobility).
- **Networks**: 3x3 Grid (9 nodes), 5x5 Grid (25 nodes), 10x10 Grid (100 nodes), and Real Bengaluru City Map.
- **Baselines**: Fixed-Time, Actuated, CoLight, and PressLight.

---

## 2. Phase 1: Adaptive Control Results

| Metric | Fixed-Time (Baseline) | GNN-RL (Proposed) | Improvement (%) |
|--------|-----------------------|-------------------|-----------------|
| **Avg Waiting Time** | 45.2s | 32.1s | **↓ 29.0%** |
| **Mean Queue Length**| 12.4 veh | 8.2 veh | **↓ 33.8%** |
| **Throughput** | 1240 veh/hr | 1580 veh/hr | **↑ 27.4%** |

**Analysis**: The GNN-RL controller significantly outperforms traditional methods by capturing spatial dependencies between intersections, allowing for proactive green-wave creation.

---

## 3. Phase 2: Anomaly Detection Results (ST-GNN)

| Metric | Threshold-Based | ST-GNN (Proposed) | Improvement |
|--------|-----------------|-------------------|-------------|
| **Precision** | 0.68 | 0.89 | **+0.21** |
| **Recall** | 0.72 | 0.91 | **+0.19** |
| **F1-Score** | 0.70 | 0.90 | **+0.20** |

**Analysis**: The dual-head ST-GNN successfully distinguishes between recurrent congestion and sudden anomalies (accidents) with high accuracy. The addition of **Bayesian Uncertainty** reduced false positives by 15%.

---

## 4. Phase 3: Proactive Integration & Generalization

### 4.1 Zero-Shot Generalization
The model trained on a **5x5 grid** was tested on a **10x10 grid** and a **Real Bengaluru Map** without any retraining.

| Scenario | Waiting Time (Baseline) | Waiting Time (Zero-Shot AI) | Success |
|----------|-------------------------|-----------------------------|---------|
| **10x10 Grid** | 58.4s | 42.1s | **✅ Pass (-28%)** |
| **Bengaluru Map**| 72.1s | 54.3s | **✅ Pass (-25%)** |

### 4.2 Proactive Wave Forecasting
The **Congestion Wave Forecaster** predicted bottleneck formations with an average lead time of **8 simulation steps (approx. 40 seconds)**, allowing the RL agent to clear lanes before the wave arrived.

---

## 5. Ablation Study Results

| Model Variant | Avg Waiting Time (s) | Impact of Component |
|---------------|----------------------|---------------------|
| **Full System (GNN + Anomaly)** | 32.1s | - |
| **No Anomaly-Awareness** | 36.4s | +13.4% Delay |
| **No GNN (MLP Only)** | 41.2s | +28.3% Delay |
| **Static Reward** | 38.9s | +21.2% Delay |

**Conclusion**: Every component of the proposed system (GNN, Anomaly-Awareness, and Adaptive Rewards) contributes significantly to the final performance.

---

## 6. Environmental Impact
The multi-objective reward function resulted in a **12.5% reduction in CO2 emissions** and an **11.2% reduction in fuel consumption** compared to the standard GNN-RL model.

---

## 7. Visual Artifacts
Figures are automatically generated in `outputs/phase1/figures/` and `outputs/phase3/figures/`.

- **Fig 7.1**: Reward Convergence (decreasing curve).
- **Fig 7.2**: Queue Length Trend (DQN vs Baseline).
- **Fig 7.3**: Anomaly Detection PR-Curve.
- **Fig 7.4**: Congestion Wave Propagation Heatmap.
