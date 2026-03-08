# cap

# Smart Traffic Management System - 3-Phase Project

A comprehensive intelligent traffic management system using Graph Neural Networks (GNNs) and AI techniques, organized into three phases:

- **Phase 1**: Traffic Prediction & Adaptive Control (GNN + DQN) — **100%** (eval + fixed-time comparison + ablation ready)
- **Phase 2**: Anomaly Detection (ST-GNN) — **~55%** (trainer + scorer done; threshold/eval pending)
- **Phase 3**: Integration (anomaly-aware rewards) — **~5%** (placeholder)

See [PROGRESS_UPDATE.md](PROGRESS_UPDATE.md) for full progress and to-dos.

## Project Overview

This project implements an adaptive traffic management system that:
- Uses GNNs to model spatial relationships between intersections
- Employs Reinforcement Learning (DQN) for adaptive traffic light control
- Detects anomalies and incidents using self-supervised ST-GNNs
- Provides advanced routing and shortest-time path calculations

For detailed project planning, see [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Quick Start

### Phase 2: Anomaly Detection (Current Implementation)

1. Create a Python 3.10+ environment with CUDA if available.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the synthetic data + training example:
   ```bash
   python -m src.training.train --config configs/default.yaml
   ```
4. Launch the dashboard (after training produces a checkpoint):
   ```bash
   streamlit run src/dashboard/app.py -- --config configs/default.yaml --checkpoint outputs/checkpoints/latest.ckpt
   ```

### Phase 1: Traffic Control (100% — Test & Show to Reviewers)

**To test, evaluate, and present Phase 1 to your review panel**, use: **[`PHASE1_REVIEWER_DEMO.md`](PHASE1_REVIEWER_DEMO.md)**.

**Quick commands (with venv activated):**
```powershell
# 1. Verify setup
python scripts/setup_environment.py
python scripts/test_setup.py

# 2. Evaluate DQN vs fixed-time (use existing checkpoint)
python -m src.phase1.evaluate --config configs/phase1.yaml --checkpoint outputs/phase1/dqn_traffic_final.zip --episodes 10
```
Show the evaluation table output and artifacts in `outputs/phase1/` to reviewers.

**Optional:** See [`PHASE1_IMPLEMENTATION_GUIDE.md`](PHASE1_IMPLEMENTATION_GUIDE.md) for implementation details. **SUMO is mandatory** (3×3 minimum, test on 6×6 for scalability).

### Quick Reference

See [`QUICK_START_CHECKLIST.md`](QUICK_START_CHECKLIST.md) for a complete checklist and common commands.

## Repo Layout

### Phase 1 (To Implement)
- `src/phase1/` – Traffic control using GNN + RL
  - `graph_builder.py` – Build traffic network graph
  - `feature_extractor.py` – Extract node/edge features
  - `gnn_encoder.py` – GNN spatial encoder
  - `traffic_env.py` – SUMO environment wrapper
  - `dqn_agent.py` – DQN agent setup
  - `train_rl.py` – Training script

### Phase 2 (Current)
- `src/models/` – Spatio-temporal GNN architectures (ST-GNN)
- `src/training/` – Training loop, datasets, and CLI entrypoint
- `src/utils/` – Metrics and helpers
- `src/dashboard/` – Streamlit alert visualization
- `src/data/` – SUMO/OSM-driven simulation utilities and graph builders

### Phase 3 (Planned)
- `src/phase3/` – Advanced features (routing, shortest time, etc.)

### Other
- `configs/` – YAML configs for data, model, and training hyperparameters
- `data/` – Raw and processed artifacts (gitignored except placeholders)
- `notebooks/` – Exploration notebooks
- `docs/` – Documentation and literature review

## Documentation

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** – Complete 3-phase project plan with detailed specifications
- **[PHASE1_IMPLEMENTATION_GUIDE.md](PHASE1_IMPLEMENTATION_GUIDE.md)** – Step-by-step guide for implementing Phase 1
- **[QUICK_START_CHECKLIST.md](QUICK_START_CHECKLIST.md)** – Quick reference checklist and common commands
- **[docs/literature_review.md](docs/literature_review.md)** – IEEE-style survey of ST-GNN traffic forecasting and anomaly detection

## Notes

- **Phase 1 (MANDATORY SUMO)**: Requires SUMO installed and on PATH. No placeholder fallback. Uses 3×3 for research baseline, 6×6 for scalability tests.
- **Phase 2**: Can use synthetic data or real SUMO runs for training ST-GNN anomaly detector.
- OSMnx requires `GEOS`/`GDAL` stack; prefer Colab/Conda for fewer issues.
- Adjust `configs/default.yaml` and network configs for dataset paths, model depth, and thresholds.
- Removed: 2×2 grid and all placeholder/synthetic fallback modes (use 3×3 minimum for research credibility).

## References

- **Smartcities_final.pdf** – Reference paper for Phase 1 approach (GNN + RL for adaptive traffic control)
- Current codebase implements Phase 2 (anomaly detection) as foundation
