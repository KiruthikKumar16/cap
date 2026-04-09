<div align="center">

# Risk-Aware MAPPO Traffic Signal Control 🚦
**An Intelligent Flow Optimization Engine using Spatial-Temporal Graph Neural Networks**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![StableBaselines3](https://img.shields.io/badge/Stable_Baselines_3-RL-success.svg)
![SUMO](https://img.shields.io/badge/Eclipse-SUMO-orange.svg)

</div>

---

## 📖 Abstract
Current traffic optimization frameworks rely heavily on fixed-time heuristics or localized multi-agent systems that blindly force throughput at the expense of adjacent gridlock. This Capstone introduces an **End-to-End Risk-Aware Multi-Agent Reinforcement Learning (MAPPO)** controller natively integrated with a predictive **Spatial-Temporal Autoencoder (ST-GNN)**. 

By autonomously detecting massive flow deviations (e.g., accidents or lane permutations) in real-time without explicit human labeling, the RL agents organically re-route vehicle clusters preventing massive downstream congestion waves.

---

## 🏗️ Architectural Overview
This system dynamically optimizes the phases of a complex $10 \times 10$ city topology.
1. **Geometric Encoding (ST-GNN):** Isolates flow matrices dynamically mapped using a `GATv2Conv` Graph Attention pipeline.
2. **Predictive Routing (MAPPO):** Computes robust joint-action policies using Proximal Policy Optimization logic on top of the autoencoder's embeddings.
3. **Hardware Acceleration:** Native PyTorch bindings ensure lightning-fast parallel emulation sweeps across standard NVIDIA Cuda instances.

---

## 📁 Repository Structure
```text
📦 cap
 ┣ 📂 configs/            # Hyperparameter mapping & Phase triggers (YAML)
 ┣ 📂 data/               # Processed spatial geometry collision buffers (.pt)
 ┣ 📂 scripts/            # Core CLI entry points for Phase testing
 ┣ 📂 src/
 ┃  ┣ 📂 baselines/       # SOTA Algorithms for comparison (PressLight, CoLight)
 ┃  ┣ 📂 models/          # ST-GNN and MAPPO pure Python/PyTorch logic
 ┃  ┣ 📂 phase1/          # RL Training environment and rewards (SUMO-RL)
 ┃  ┣ 📂 phase2/          # Anomaly Sequence PyTorch Autoencoder Model
 ┃  ┗ 📂 phase3/          # Dynamic Risk-Aware Integration Binders
 ┣ 📂 outputs/            # Extracted SOTA Metrics, Checkpoints, and Visual PNG Plots
 ┗ 📜 README.md           # You are here
```

---

## 🚀 Installation & Build Guide

The framework requires **Python 3.9+** and a strict path mapping to **Eclipse SUMO** (Simulation of Urban MObility).

### 1. Repository Setup
```powershell
git clone https://github.com/KiruthikKumar16/cap.git
cd cap
```

### 2. Environment Configuration
We highly advise partitioning your dependencies natively into a GPU-enabled Virtual Environment.
```powershell
python -m venv venv_gpu
.\venv_gpu\Scripts\activate
```

### 3. Dependency Injection
*(Verify CUDA build compatibility natively if using NVIDIA GPU)*
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3[extra] sumolib traci torch-geometric pyyaml matplotlib
```

### 4. SUMO Bridging
Ensure SUMO is installed on your Windows/Linux machine and append the exact TraCI executable path to your runtime. On Windows:
```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

---

## 🔬 Execution Workflows

To simplify grading and demonstration, the project pipelines have been decoupled into 3 executable testing wrappers.

### Phase 1: Pure MARL Benchmarking
Evaluates and contrasts the pure baseline throughput of our core MAPPO algorithm natively against graph-based CoLight and standard PressLight models.
```powershell
python scripts/test_phase1.py
```
> **Output:** Populates `outputs/benchmark_results.json` containing numeric flow delays.

### Phase 2: Traffic Anomaly Pipeline
Extracts latent geometric trajectories directly from the generated simulation buffers and tests the ST-GNN Autoencoder for false-positive detection stability via dense accident reconstruction.
```powershell
python scripts/test_phase2.py
```
> **Output:** Exports exact F1 / Precision limits mapping the Autoencoder prediction bound accuracy.

### Phase 3: The Risk-Aware Integration Loop
Mounts Phase 2 directly into the memory block of Phase 1. As the RL model attempts to actuate the simulation, the Anomaly detector acts as an active **Penalty Overload**; artificially crashing the rewards explicitly routing into accident nodes to invoke "Risk-Aware Avoidance" within the central agent.
```powershell
python scripts/test_phase3.py
```

---

## 📊 Result Summaries
Visual interpretations mapping throughput volume reductions alongside execution latency gradients are automatically parsed into Graph Data inside `outputs/plots/`.

- MAPPO natively achieves an **estimated ~23% reduction** in aggregate queue trailing distances.
- Integration mapping correctly forces non-deterministic phase shifting over heavy lane blockades natively.

---

> This repository serves as the Final Deliverable output. Code logic implemented securely on local runtime architecture with 100% Native Code Extraction mappings.
