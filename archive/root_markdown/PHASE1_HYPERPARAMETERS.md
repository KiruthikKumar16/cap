# Phase 1 — Hyperparameter Table (SOTA)

Use this table in your report or README for reproducibility. All values are from `configs/phase1.yaml` unless noted.

---

## Experiment

| Parameter | Value |
|-----------|--------|
| Name | gnn_rl_traffic_control |
| Seed | 42 |
| Output directory | outputs/phase1 |

---

## SUMO

| Parameter | Value |
|-----------|--------|
| Net file | data/raw/grid_2x2.net.xml |
| Route file | data/raw/grid_2x2.rou.xml |
| Config file | data/raw/grid_2x2.sumocfg |
| Step length (s) | 1.0 |
| Simulation steps per episode | 3600 |
| GUI | false |

---

## Model (GNN / encoder)

| Parameter | Value |
|-----------|--------|
| Use GNN | true (false for MLP ablation) |
| Feature dimension | 12 |
| Hidden dimension | 64 |
| Embedding dimension | 32 |
| GNN layers | 2 |
| GNN type | gat (options: gcn, gat) |
| GAT heads | 2 |
| Dropout | 0.1 |

---

## Reinforcement learning (DQN)

| Parameter | Value |
|-----------|--------|
| Algorithm | DQN |
| Learning rate | 0.001 |
| Buffer size | 50000 |
| Batch size | 32 |
| Gamma | 0.99 |
| Tau (soft update) | 1.0 (hard update) |
| Target update interval | 1000 |
| Exploration initial ε | 1.0 |
| Exploration final ε | 0.05 |
| Exploration fraction | 0.1 |
| Learning starts | 1000 |
| Train frequency | 4 |
| Gradient steps | 1 |
| **Double DQN** | **true** |
| **Dueling** | **true** |

---

## Reward

| Parameter | Value |
|-----------|--------|
| Waiting time weight (α₁) | 0.1 |
| Queue length weight (α₂) | 0.05 |
| Anomaly weight (α₃) | 0.0 |
| Throughput weight (α₄) | 0.0 |
| **Pressure weight (PressLight-style)** | **0.0** |
| Normalize | true |
| Max throughput per step (norm) | 20.0 |

---

## Training

| Parameter | Value |
|-----------|--------|
| Total timesteps | 100000 |
| Eval frequency | 5000 |
| Eval episodes | 10 |
| Save frequency | 10000 |
| Log interval | 10 |
| Device | auto |

---

## Evaluation

| Parameter | Value |
|-----------|--------|
| Num episodes | 100 |
| Deterministic | true |
| Render | false |
| Seeds (for mean ± std) | [42, 43, 44, 45, 46] |

---

## Output paths

| Parameter | Value |
|-----------|--------|
| Checkpoint dir | outputs/phase1/checkpoints |
| Log dir | outputs/phase1/logs |
| Best model dir | outputs/phase1/best_models |
| Final model path | outputs/phase1/dqn_traffic_final.zip |
