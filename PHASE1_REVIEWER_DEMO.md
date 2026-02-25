# Phase 1 — How to Test, Evaluate, and Show to Review Panel

This guide is for **testing Phase 1**, **running evaluation**, and **presenting results** to your reviewer panel.

---

## 1. What Phase 1 Does (Summary for Your Guide)

**Phase 1** implements **adaptive traffic signal control** using:

1. **Traffic network as a graph**  
   Intersections are **nodes**; road links between them are **edges**. The system builds this graph from a SUMO network (or a placeholder 2x2 grid when SUMO is not installed).

2. **Per-intersection features**  
   At each node we extract features: current signal phase, phase duration, queue lengths, waiting times, vehicle counts, etc. (12 features per intersection).

3. **GNN encoder**  
   A **Graph Neural Network (GAT)** takes the graph and node features and produces a **spatial encoding** for each intersection (so each node’s representation depends on its neighbors). These embeddings are flattened into a single observation vector.

4. **DQN agent**  
   A **Deep Q-Network (DQN)** receives that observation and outputs **one action per intersection**: which signal phase (0–3) to set. So the agent controls all intersections jointly.

5. **Environment and reward**  
   The **environment** (SUMO or placeholder) applies the actions, advances the simulation by one step, and returns a **reward** based on total waiting time and queue length (lower is better). The agent is trained to maximize cumulative reward (i.e. reduce congestion).

**In one sentence:** Phase 1 learns a **policy** that, at each time step, observes the traffic graph and features, and chooses signal phases for all intersections to minimize waiting time and queues.

---

## 2. Graphs for Proof (Same Style as Smartcities_final.pdf)

You can **generate figures** that match the style of **Smartcities_final.pdf** (Fig 4.0, 5.1, 7.1–7.3) for your guide and reviewers:

| Figure | What it shows | Output file |
|--------|----------------|-------------|
| **Fig 4.0 – Proposed System Architecture** | SUMO → TraCI API → Graph Construction → Feature Extraction & Normalization → GNN Encoder → DQN → RL Loop → Assessment | `phase1_architecture.png` |
| **Fig 5.1 – 2×2 Grid Traffic Network** | SUMO simulation environment: nodes = intersections (J0–J3), edges = road links (the graph the GNN uses) | `phase1_traffic_network_graph.png` |
| **Fig 7.1 – Reward per episode** | Mean reward during training / evaluation | `phase1_reward_per_episode.png` |
| **Fig 7.2 – Average queue length** | Average queue length per episode (trend) | `phase1_queue_length_per_episode.png` |
| **Fig 7.3 – Average waiting time** | Average waiting time per episode (trend) | `phase1_waiting_time_per_episode.png` |

**Generate all figures** (from project root, with venv activated):

```powershell
python scripts/phase1_generate_figures.py
```

Outputs are saved to **`outputs/phase1/figures/`**. Reward (7.1) uses your evaluation log (`outputs/phase1/logs/evaluations.npz`) when available; queue and waiting (7.2, 7.3) use placeholder trends unless you add env logging.

**Use these in your report or slides:** insert the images and refer to “What Phase 1 does” (Section 1). This gives your guide/reviewers **proof** in the same style as Smartcities_final.pdf (architecture, traffic graph, and performance curves).

---

## 3. What Phase 1 Delivers (Checklist)

- **GNN-RL traffic control**: Graph Neural Network encodes the traffic network; a DQN agent chooses signal phases at each intersection.
- **Training pipeline**: Trains on a 4-intersection (2x2) setup; works in **placeholder mode** (no SUMO required) or with SUMO when installed.
- **Evaluation**: Compares the trained DQN agent against a **fixed-time baseline** (same phase duration for all intersections).
- **Ablation**: Option to train **without GNN** (MLP encoder) via config for comparison.

---

## 4. Prerequisites (One-Time)

- **Python 3.10+** with a **virtual environment** (e.g. `.venv`).
- **Dependencies** installed in that environment:
  ```powershell
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
  Or install Phase 1 essentials only:
  ```powershell
  pip install torch torch-geometric stable-baselines3[extra] gymnasium pyyaml numpy
  ```
- **SUMO** is optional; without it, the project runs in placeholder mode (synthetic rewards/features). To use SUMO: install SUMO, add it to PATH, and ensure `data/raw/` has the network files or run `scripts/create_sumo_network.py`.

---

## 5. Step-by-Step: Test and Evaluate

### Step A — Verify setup (≈ 1 min)

From the project root (with venv activated):

```powershell
python scripts/setup_environment.py
python scripts/test_setup.py
```

- **setup_environment.py**: Checks Python, packages, and optional SUMO.
- **test_setup.py**: Tests imports and Phase 1 modules (graph builder, feature extractor, config).

If both complete without errors, the environment is ready.

---

### Step B — Quick training test (optional, ≈ 2–5 min)

To show that training runs end-to-end (e.g. 5k steps):

```powershell
# Edit configs/phase1.yaml: set training.total_timesteps: 5000 (for a quick run)
# Or run with default 100k (longer):
python -m src.phase1.train_rl --config configs/phase1.yaml
```

- **Output**: Checkpoints in `outputs/phase1/checkpoints/`, best model in `outputs/phase1/best_models/`, final model at `outputs/phase1/dqn_traffic_final.zip`.
- If you already have `outputs/phase1/dqn_traffic_final.zip` from a previous run, you can **skip** this step and go to evaluation.

---

### Step C — Run evaluation (≈ 2–5 min)

This compares the **trained DQN** with the **fixed-time baseline** over several episodes and prints a summary table.

```powershell
python -m src.phase1.evaluate --config configs/phase1.yaml --checkpoint outputs/phase1/dqn_traffic_final.zip --episodes 10
```

- **What it does**: Runs 10 evaluation episodes with the DQN (deterministic), then 10 episodes with the fixed-time controller, and reports mean reward ± std and mean episode length for both.
- **Output**: A table like:

  ```
  ============================================================
  Phase 1 Evaluation Results
  ============================================================
    Episodes: 10
    Checkpoint: outputs\phase1\dqn_traffic_final.zip
  ------------------------------------------------------------
    DQN (GNN-RL):     mean_reward = -1085.xx +/- xx.xx  |  mean_length = 3600.0
    Fixed-time:      mean_reward = -1086.xx +/- xx.xx  |  mean_length = 3600.0
  ------------------------------------------------------------
    DQN vs Fixed-time: +x.x% reward change (positive = DQN better)
  ============================================================
  [OK] Evaluation complete.
  ```

- **For the panel**: Run this command, then **copy the full terminal output** (or take a screenshot) and show it as evidence of Phase 1 evaluation.

---

### Step D — Show artifacts to reviewers

| What to show | Where it is |
|--------------|-------------|
| **Evaluation output** | Terminal output from Step C (copy or screenshot). |
| **Trained model** | `outputs/phase1/dqn_traffic_final.zip` (and/or `outputs/phase1/best_models/best_model.zip`). |
| **Checkpoints** | `outputs/phase1/checkpoints/` (e.g. every 10k steps). |
| **Config** | `configs/phase1.yaml` (model, RL, reward, training settings). |
| **Phase 1 code** | `src/phase1/`: `graph_builder.py`, `feature_extractor.py`, `gnn_encoder.py`, `traffic_env.py`, `reward_calculator.py`, `dqn_agent.py`, `train_rl.py`, `evaluate.py`. |

---

## 6. One-Page Summary for the Panel

You can paste or adapt the following for your report or slides.

---

**Phase 1 — GNN-RL Traffic Signal Control (Complete)**

- **Objective**: Adaptive traffic light control using a Graph Neural Network (GNN) for spatial state encoding and Deep Q-Network (DQN) for phase selection at each intersection.
- **Implementation**: 
  - Traffic network represented as a graph (nodes = intersections, edges = connections).
  - Per-intersection features (phase, queue lengths, waiting times, etc.) extracted and encoded by a GNN (GAT); flattened encoding is the observation for the DQN.
  - Gymnasium-compatible SUMO environment (with placeholder mode when SUMO is not installed); multi-objective reward (waiting time, queue length).
  - DQN agent (Stable Baselines3) with MultiDiscrete→Discrete action wrapper; training with checkpoints and evaluation callbacks.
- **Evaluation**: 
  - Trained agent evaluated against a **fixed-time baseline** (cyclic phase schedule, same duration per phase).
  - Metrics: mean episode reward and mean episode length over N evaluation episodes for both DQN and fixed-time.
  - Example command: `python -m src.phase1.evaluate --config configs/phase1.yaml --checkpoint outputs/phase1/dqn_traffic_final.zip --episodes 10`.
- **Ablation**: 
  - Config option `model.use_gnn: false` trains an MLP-based encoder (no graph); same observation size for fair comparison. Enables “with vs without GNN” comparison for the report.
- **Deliverables**: 
  - Working training pipeline (`train_rl.py`), evaluation script (`evaluate.py`), saved models and checkpoints, config-driven ablation option.

---

## 7. Optional: Ablation (With vs Without GNN)

To demonstrate the ablation study to the panel:

1. **Train with GNN** (default):  
   `python -m src.phase1.train_rl --config configs/phase1.yaml`  
   → Saves e.g. `outputs/phase1/dqn_traffic_final.zip`.

2. **Train without GNN**:  
   In `configs/phase1.yaml` set `model.use_gnn: false`.  
   Run training again and save to a different path (e.g. change `output_dir` or copy the final zip to `outputs/phase1/dqn_traffic_mlp_final.zip`).

3. **Evaluate both**:  
   Run `evaluate.py` once with the GNN checkpoint and once with the MLP checkpoint; compare the two “Phase 1 Evaluation Results” tables and report the difference in mean reward in your document or slides.

---

## 8. Quick Command Reference

| Action | Command |
|--------|--------|
| **Generate graphs (Smartcities style)** | `python scripts/phase1_generate_figures.py` → Fig 4.0, 5.1, 7.1–7.3 in `outputs/phase1/figures/` |
| Check environment | `python scripts/setup_environment.py` |
| Test Phase 1 modules | `python scripts/test_setup.py` |
| Train (full) | `python -m src.phase1.train_rl --config configs/phase1.yaml` |
| Evaluate (DQN vs fixed-time) | `python -m src.phase1.evaluate --config configs/phase1.yaml --checkpoint outputs/phase1/dqn_traffic_final.zip --episodes 10` |
| Evaluate (fewer episodes) | `python -m src.phase1.evaluate --config configs/phase1.yaml --checkpoint outputs/phase1/dqn_traffic_final.zip --episodes 5` |

---

**Summary for reviewers**: Phase 1 is **fully implemented and testable**. Run **Step A** (setup + test), then **Step C** (evaluation); show the **evaluation table** and the **artifact paths** above. Use the **one-page summary** in Section 4 for the report or panel presentation.
