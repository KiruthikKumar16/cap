# Phase 1 — What’s Left to Reach State-of-the-Art Level

This document lists what Phase 1 **already has** and what is **left** to bring it to **state-of-the-art** level (comparable to CoLight, PressLight, and similar GNN-RL traffic signal control work).

---

## 1. What Phase 1 Already Has (Current State)

| Component | Status | Notes |
|-----------|--------|------|
| **Graph** | Done | Intersections = nodes, road links = edges; PyG `edge_index`; placeholder when no SUMO |
| **Features** | Done | 12 per node: phase one-hot, duration, queue sum/max, waiting, vehicle counts |
| **GNN encoder** | Done | GCN or GAT (config); node embeddings; MLP ablation (`use_gnn: false`) |
| **State** | Done | Flattened GNN embeddings as observation |
| **Action space** | Done | MultiDiscrete (4 phases × N intersections); wrapped to Discrete for SB3 |
| **Reward** | Done | R = −α·waiting − β·queue + optional throughput; `get_reward_components()` |
| **RL algorithm** | Done | DQN (SB3): replay buffer, target network, epsilon-greedy |
| **Environment** | Done | Gymnasium wrapper; SUMO + TraCI when available; placeholder mode |
| **Training** | Done | `train_rl.py`; 100k steps; checkpoints; eval callback |
| **Evaluation** | Done | DQN vs **fixed-time**; mean reward, episode length, throughput (when SUMO) |
| **Config** | Done | Single YAML (model, reward, RL, SUMO); reproducible |
| **Figures** | Done | Architecture, data flow, use case, class, sequence, network graph, Fig 7.1–7.3 |

---

## 2. What’s Left for State-of-the-Art Level

State-of-the-art traffic signal control (e.g. CoLight, PressLight, MPLight) typically adds the following. Items are ordered by impact and feasibility.

### 2.1 Evaluation & Baselines (High impact, expected in SOTA)

| Item | Status | What to do |
|------|--------|------------|
| **Actuated baseline** | Not done | Add an actuated controller (e.g. switch phase when queue on current phase exceeds threshold or max green) and compare DQN vs fixed-time vs actuated in `evaluate.py`. |
| **Multiple seeds** | Not done | Run training/eval with 3–5 seeds; report mean ± std (or confidence interval) for reward, travel time, throughput. |
| **Statistical significance** | Not done | When comparing DQN vs baselines, run a simple test (e.g. paired t-test or Wilcoxon) and report p-value. |
| **CoLight / PressLight comparison** | Not done | Either (a) implement simplified CoLight/PressLight baselines (graph attention / pressure-based reward) or (b) cite their reported numbers and compare your setup (same network, same metrics) so reviewers see you’re in the same league. |

### 2.2 RL Algorithm Upgrades (Medium impact, common in SOTA)

| Item | Status | What to do |
|------|--------|------------|
| **Double DQN** | Not done | SB3 DQN supports `use_double_dqn=True`; add to config and enable to reduce overestimation. |
| **Dueling DQN** | Not done | SB3 supports `policy_kwargs=dict(dueling=True)`; add config option for dueling architecture. |
| **n-step returns** | Not done | Consider using SB3’s `NStepReplayBuffer` or switch to Rainbow-style n-step; improves credit assignment. |
| **Prioritized replay** | Optional | Not in standard SB3 DQN; would require custom buffer; lower priority unless you aim for a full Rainbow-style setup. |

### 2.3 Reward & State (Medium impact)

| Item | Status | What to do |
|------|--------|------------|
| **Pressure (PressLight-style)** | Not done | Add optional reward or feature: pressure = queue difference (incoming − outgoing) per movement/lane; often improves coordination. |
| **Travel time** | Partial | If SUMO available, log travel time (e.g. from `traci.simulation.getArrivedIDList` + travel time subscription); report “average travel time” in evaluation table. |

### 2.4 Scalability & Benchmarks (Medium impact for “SOTA” story)

| Item | Status | What to do |
|------|--------|------------|
| **Larger networks** | Not done | Add 4×4 or 6×6 SUMO grids (or use existing scripts); train and evaluate on 2×2, 4×4, (6×6) to show scalability. |
| **SUMO runs** | Placeholder only | Install SUMO, add to PATH, create/use `.net.xml` and `.rou.xml`; run training and evaluation with real SUMO so reported numbers are from simulation, not placeholder. |

### 2.5 Reproducibility & Reporting (Needed for SOTA)

| Item | Status | What to do |
|------|--------|------------|
| **Seeds in config** | Done | Already in `phase1.yaml`. |
| **Hyperparameter table** | Partial | Add a “Phase 1 hyperparameters” table to the report (or README) listing: lr, buffer size, gamma, exploration, GNN type/layers, reward weights, etc. |
| **Metrics table** | Partial | In report: table with DQN vs fixed-time (and actuated if added) for reward, average travel time, throughput, queue/waiting (when available); mean ± std over seeds. |

### 2.6 Optional (Nice-to-have)

| Item | Status | What to do |
|------|--------|------------|
| **Lane-level graph** | Not done | Use lanes as nodes (instead of intersections) for finer-grained control; more complex, often in CoLight/PressLight variants. |
| **Edge features** | Not done | Add edge attributes (e.g. distance, capacity) to the graph and GNN. |
| **Learning curve from real training** | Synthetic figures | When SUMO training runs, log eval reward (and optionally queue/wait) per eval; use these for Fig 7.1–7.3 instead of synthetic curves. |

---

## 3. Prioritized Checklist (State-of-the-Art Level)

Use this as a concise “what’s left” list. Order is by impact for a SOTA narrative.

- [ ] **SUMO runs** — Install SUMO, run training and evaluation with real simulation (not only placeholder).
- [ ] **Actuated baseline** — Implement actuated controller; add DQN vs fixed-time vs actuated to `evaluate.py`.
- [ ] **Multiple seeds** — Run 3–5 seeds for training/eval; report mean ± std for reward, travel time, throughput.
- [ ] **Double DQN** — Enable in config (`use_double_dqn: true`) and document.
- [ ] **Dueling DQN** — Add config option and enable via `policy_kwargs` (dueling).
- [ ] **Pressure reward/feature** — Add optional PressLight-style pressure (e.g. queue diff) to reward or features.
- [ ] **Travel time metric** — Log/report average travel time when SUMO is used.
- [ ] **Statistical test** — Report p-value (e.g. t-test) for DQN vs baseline(s).
- [ ] **CoLight/PressLight comparison** — Implement simplified baselines or align setup and cite their numbers with same metrics.
- [ ] **Larger networks** — 4×4 (and optionally 6×6) grid; report results to show scalability.
- [ ] **Hyperparameter table** — One table in report/README with all Phase 1 hyperparameters.
- [ ] **Real learning curves** — Use logged eval data for Fig 7.1–7.3 when available.

---

## 4. Summary

- **Already at “paper-ready” level:** Graph, GNN (GAT/GCN), DQN, multi-objective reward, fixed-time baseline, config-driven pipeline, placeholder mode, figures, and documentation.
- **To reach state-of-the-art level:** Add (1) real SUMO runs, (2) actuated baseline and multi-seed evaluation with stats, (3) Double/Dueling DQN and optional pressure, (4) travel time and throughput in the metrics table, (5) comparison with CoLight/PressLight (implementation or cited numbers), and (6) scalability on 4×4 (and optionally 6×6) and a clear hyperparameter table. The checklist in Section 3 is your “what’s left” list for Phase 1 at SOTA level.
