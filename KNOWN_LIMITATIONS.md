# Known Limitations & Technical Warnings

## 🚦 Phase 1: Foundation & MARL
- **Smoke Test vs. Production**: Phase 1 smoke tests use a short SUMO horizon (e.g., 360 steps) and are not intended for publication-grade benchmarks.
  - *Mitigation*: Run `scripts/run_benchmarks.py` with the `--production` flag to execute full 3600s horizons.
- **Checkpoint Compatibility**: The current compatible PPO checkpoint is `checkpoints/marl_ppo_traffic.zip`. Older checkpoints (e.g., `dqn_traffic_final.zip`) may be incompatible.
  - *Mitigation*: Use the `ModelMigrationTool` (to be implemented) to map weights across observation space versions.
- **Action Sequence**: The system assumes a standard 4-phase sequence.
  - *Mitigation*: **[SOLVED]** Implemented Dynamic Phase Skipping logic in `traffic_env.py` (`_apply_phase_skipping`). It cycles signals early if zero-demand is detected on the chosen phase.

## ⚠️ Phase 2: Anomaly Intelligence
- **Data Labeling**: Anomaly metrics are primarily computed from synthetic data injections.
  - *Mitigation*: Implement "Semi-Supervised Active Learning" to label real-world edge cases from the CV stream.
- **Evidence Gaps**: Full ablation and scalability tables are evidence gaps.
  - *Mitigation*: Execute `scripts/run_publication_suite.py` to regenerate all evidence tables.

## 👁️ Phase 3: Real-World Perception & Loopback
- **Calibration Overhead**: Manual homography calibration (4 points) is required per camera.
  - *Mitigation*: **[SOLVED]** Implemented `auto_calibrate` in `PerspectiveTransformer` using Hough Transform for lane detection and vanishing point estimation.
- **Environmental Robustness**: YOLOv10 sensitivity to rain/fog/low-light.
  - *Mitigation*: **[SOLVED]** Implemented Adversarial Noise Augmentation in `traffic_env.py` (`_get_raw_observation`) to simulate sensor variance and failure during training.
- **ROI Dependency**: Manual lane ROI definition.
  - *Mitigation*: Automated ROI generation using semantic segmentation (SAM) to detect road boundaries.

## 💻 Hardware & Performance
- **Inference Latency**: High-end hardware required for 30 FPS.
  - *Mitigation*: Use `model.export(format="engine")` for TensorRT optimization and implement "Frame Skipping" (Perception at 10Hz, Control at 1Hz).
- **GPU Saturation**: Memory spikes during concurrent tasks.
  - *Mitigation*: Implement a "Decentralized Perception" architecture where video processing is offloaded to a separate process or edge node.

## 📂 Project & Legacy
- **Sim-to-Real Gap**: SUMO models lack non-standard real-world behaviors.
  - *Mitigation*: **[SOLVED]** Implemented Heterogeneous Traffic Injection in `create_sumo_scenario.py` (`_inject_mixed_traffic`) to include rickshaws and bicycles.
- **SUMO Port Conflicts**: Runtime-sensitive port failures.
  - *Mitigation*: Automated port management and "Zombie Process" cleanup in `traffic_env.py`.
