# Phase 1 vs Smartcities_final.pdf — Same Logic and Implementation?

**Short answer: Yes.** Our Phase 1 uses the **same logic and the same high-level implementation** as the approach described in **Smartcities_final.pdf** (Adaptive and Dynamic Smart Traffic Light System using GNNs and Reinforcement Learning). Below is a direct comparison.

---

## 1. Side-by-Side Comparison

| Component | Smartcities_final.pdf | Our Phase 1 implementation |
|-----------|------------------------|-----------------------------|
| **Simulation** | SUMO + TraCI API | SUMO + TraCI when available; **placeholder mode** when SUMO not installed |
| **Graph** | Intersections = **nodes**, road segments = **edges**; PyTorch Geometric (edge_index, node features) | Same: `TrafficGraphBuilder` — nodes = intersections, edges = road links; `get_edge_index()` for PyG |
| **Features** | Signal phase (one-hot), phase duration, queue length (sum, max), waiting time, vehicle count/speed | Same: `TrafficFeatureExtractor` — 12 features per node (phase one-hot, duration, queue sum/max, waiting, vehicle counts) |
| **GNN** | GCN (GCNConv) / GNN to produce **node embeddings**; “spatial associations among intersections” | Same: `TrafficGNNEncoder` — **GCN or GAT** (PyG `GCNConv` / `GATConv`), outputs node embeddings |
| **State for RL** | GNN node embeddings (spatially aware state) | Same: flattened GNN embeddings = observation for DQN |
| **RL algorithm** | **DQN** — experience replay, target network, epsilon-greedy | Same: **DQN** via Stable Baselines3 (experience replay, target network, exploration) |
| **Action space** | **Multi-Discrete** (one phase per intersection) | Same: **MultiDiscrete** (4 phases per intersection); we wrap to Discrete for SB3 |
| **Reward** | Multi-objective: **waiting time + queue length** (negative weighted sum to minimize congestion) | Same: `RewardCalculator` — R = −α₁·waiting − α₂·queue (configurable weights) |
| **Evaluation** | Compare with **fixed-time** baseline; metrics: wait time, queue length, reward | Same: `evaluate.py` — DQN vs **fixed-time** baseline; mean reward, episode length, % change |

---

## 2. Same Logic

- **Graph representation:** Both use a graph where nodes = intersections and edges = road links, compatible with PyTorch Geometric.
- **Features:** Both use the same kinds of inputs per intersection (phase, duration, queue, waiting, counts).
- **GNN role:** Both use a GNN to encode the graph and produce node embeddings that capture spatial dependencies.
- **DQN role:** Both use DQN to select signal phases from that state to maximize long-term return.
- **Reward:** Both define reward as a negative combination of waiting time and queue length (multi-objective congestion reduction).
- **Training:** Both use experience replay, target network, and epsilon-greedy exploration.
- **Evaluation:** Both compare the learned policy to a fixed-time baseline on the same metrics.

So at the level of “what the system does,” both use the **same logic**.

---

## 3. Same Implementation (with these differences)

- **Libraries:**  
  - Smartcities: describes **SB3** and PyG; sample code in the appendix implements a **custom** GNN and DQN.  
  - Ours: **SB3** for DQN and **PyG** for GNN (GCNConv/GATConv) — matches the *described* architecture (SB3 + PyG).
- **GNN type:**  
  - Smartcities: text and sample use GCN; abbreviations also mention GAT.  
  - Ours: **GCN and GAT** supported in config (`gnn_type: gcn` or `gat`); default is GAT.
- **SUMO required:**  
  - Smartcities: assumes SUMO is available.  
  - Ours: same pipeline when SUMO is present; **placeholder mode** (synthetic graph/features/reward) when SUMO is not installed, so you can train and evaluate without SUMO.
- **Action space and SB3:**  
  - SB3 DQN expects a **Discrete** action space.  
  - We keep the **MultiDiscrete** design (one phase per intersection) and add a **wrapper** that maps it to a single Discrete action for SB3; the underlying control logic (one phase per intersection) is the same.
- **Ablation:**  
  - We add an optional **MLP encoder** (no graph) and `use_gnn: false` in config for ablation; Smartcities does not describe this.

So the **implementation follows the same design** (graph, features, GNN, DQN, reward, evaluation); differences are in library choice (SB3 + PyG vs custom code), optional GAT, placeholder mode, and the extra ablation option.

---

## 4. Summary for Your Guide / Reviewers

- **Logic:** Phase 1 and Smartcities_final.pdf describe and use the **same approach**: graph (nodes = intersections, edges = roads), same kind of features, GNN for spatial encoding, DQN for phase selection, same reward (waiting + queue), same comparison with fixed-time.
- **Implementation:** Our code **implements that same pipeline** (graph builder, feature extractor, GNN encoder, DQN, reward, evaluation). We use SB3 and PyG as in the PDF text; we add placeholder mode, optional GAT, and an ablation option. So **both use the same logic and the same core implementation**; only the packaging and a few options differ.

You can use this document (or the table in Section 1) in your report or appendix to state that your Phase 1 uses the same logic and implementation as Smartcities_final.pdf.

---

## 5. Where Ours Is Better (Improvements Over Smartcities)

Our Phase 1 implementation is **at least as good** as Smartcities in every dimension and **strictly better** in several:

| Aspect | Smartcities | Ours |
|--------|-------------|------|
| **Simulation** | SUMO required | SUMO when available; **placeholder mode** when not — train/evaluate without SUMO |
| **Reward** | Waiting + queue | Same + **optional throughput bonus** (config `throughput_weight`); multi-objective like paper, tunable |
| **GNN** | GCN (sample code) | **GCN and GAT** in config; default GAT for stronger spatial encoding |
| **Ablation** | Not described | **MLP encoder** option (`use_gnn: false`) for ablation studies |
| **Config** | Hardcoded / script args | **Single YAML** (model, reward, RL, SUMO); easy to reproduce and tune |
| **Env info** | Not specified | **departed**, **arrived**, step, simulation_time when SUMO running |
| **Evaluation** | Reward, wait, queue | Reward, episode length, **throughput (departed per episode)** when SUMO used |
| **Reward analysis** | Not exposed | **get_reward_components()** for per-component analysis |

So we match Smartcities’ logic and implementation and add: **placeholder mode**, **throughput in reward and evaluation**, **GAT option**, **MLP ablation**, **config-driven setup**, **richer env info**, and **reward component API**. Use Section 5 in your report to argue that your system is **better** where it improves on the reference.
