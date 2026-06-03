# System Implementation Guide: Smart Traffic Management

This guide provides a comprehensive technical overview of the implementation logic across all three phases of the Smart Traffic Management System.

---

## 🏗️ Phase 1: Adaptive Traffic Control (GNN + RL)

### 1.1 Graph Construction (`src/phase1/graph_builder.py`)
- **Logic**: Parses SUMO `.net.xml` files using `sumolib`.
- **Node Filtering**: Identifies signalized junctions as nodes and filters out internal junctions (e.g., `:0_0`).
- **Graph Object**: Creates a `networkx.DiGraph` and converts it to `torch_geometric.data.Data`.

### 1.2 Feature Extraction (`src/phase1/feature_extractor.py`)
- **Real-time Extraction**: Pulls 12 features per node via TraCI:
  - Signal phase (4-dim one-hot)
  - Phase duration (1-dim normalized)
  - Queue lengths (Sum/Max)
  - Waiting time (1-dim normalized)
  - Vehicle counts (4 directions)

### 1.3 GNN Encoding (`src/phase1/gnn_encoder.py`)
- **Architecture**: Implements Graph Attention Networks (GAT) to capture spatial state influence between neighboring intersections.
- **Output**: Produces latent embeddings that serve as the input state for the RL policy.

---

## 🕵️ Phase 2: Anomaly Detection (ST-GNN)

### 2.1 Spatio-Temporal Modeling (`src/models/st_gnn.py`)
- **Architecture**: Combines GAT spatial layers with a GRU temporal encoder.
- **Dual-Head Output**: 
  - `recon_head`: Reconstructs the current traffic state.
  - `mean_head`: Forecasts the future traffic sequence (horizon: 3-5 steps).
- **Uncertainty Logic**: Implements Monte Carlo Dropout (`mc_dropout_predict`) to estimate epistemic and aleatoric uncertainty.

### 2.2 Anomaly Scoring (`src/phase2/anomaly_scorer.py`)
- **Metrics**: Calculates MSE between reconstructed/forecasted values and actual traffic.
- **Classification**: Uses quantile-based adaptive thresholding to flag incidents (Accidents, Surges).

---

## 🧠 Phase 3: Proactive Integration & Coordination

### 3.1 Proactive Integration (`src/phase3/integration.py`)
- **Controller**: Maintains a rolling window of traffic history to feed the ST-GNN.
- **Penalty Logic**: Translates anomaly severity into a reward penalty for the RL agent.

### 3.2 Self-Adaptive Reward Shaping (`src/phase1/reward_calculator.py`)
- **Dynamic Weighting**: Automatically shifts optimization goals based on:
  - **Density**: Prioritizes queue reduction when density > 0.7.
  - **Severity**: Multiplies anomaly penalty when severity > 0.5.
  - **Peak Hproposed**: Adjusts waiting time importance during rush hproposed.

### 3.3 Multi-Agent Coordination (`src/phase3/multi_agent_coordination.py`)
- **Regional Controller**: Aggregates local states into a regional embedding to provide zone-level guidance.
- **Consensus Action**: Intersections exchange messages to coordinate flow during detected anomalies.

### 3.4 Risk & Wave Forecasting (`src/phase3/risk_model.py` & `predictive_control.py`)
- **Spillback Prob**: Probabilistic calculation of queue overflow.
- **Accident Likelihood**: Risk estimation based on vehicle speed variance.
- **Wave Propagation**: Simulates the spread of congestion waves across the graph to predict bottlenecks.

---

## 📊 Evaluation & Visualization

### 3.1 Benchmarking (`scripts/run_benchmarks.py`)
- Compares the integrated model against **Fixed-Time**, **Actuated**, **CoLight**, and **PressLight**.

### 3.2 Visualization
- **Dashboard**: `src/dashboard/app.py` (Streamlit).
- **Figures**: `scripts/phase1_generate_figures.py` (Matplotlib/Seaborn).

---

## 🛠️ Testing & Validation Checklist
- [x] Environment Verification (`scripts/setup_environment.py`)
- [x] Zero-Shot Generalization (5x5 → 10x10)
- [x] Ablation Study (GNN vs. No-GNN)
- [x] Multi-Objective Optimization (CO2/Fuel)
