# Known Limitations & Technical Warnings

## 🚦 Phase 1: Foundation & MARL
- **Smoke Test vs. Production**: Phase 1 smoke tests use a short SUMO horizon (e.g., 360 steps) and are not intended for publication-grade benchmarks.
- **Checkpoint Compatibility**: The current compatible PPO checkpoint is `checkpoints/marl_ppo_traffic.zip`. Older checkpoints (e.g., `dqn_traffic_final.zip`) may be incompatible with the current `configs/phase1.yaml` due to observation space shifts.
- **Action Sequence**: The system assumes a standard 4-phase sequence. It does not yet support dynamic phase skipping based on zero-demand detection from the CV layer.

## ⚠️ Phase 2: Anomaly Intelligence
- **Data Labeling**: Phase 2 anomaly metrics are primarily computed from synthetic data injections (e.g., lane closures/crashes in SUMO) unless a real-world labeled traffic incident dataset is provided.
- **Evidence Gaps**: Full ablation, scalability, and zero-shot generalization tables are evidence gaps unless regenerated from the latest system configuration.

## 👁️ Phase 3: Real-World Perception & Loopback
- **Calibration Overhead**: The [PerspectiveTransformer](file:///home/kk/cap/src/perception/yolo_inference.py) requires manual calibration of 4 reference points (homography) per camera view to map pixels to real-world meters.
- **Environmental Robustness**: High-fidelity perception (YOLOv10) is sensitive to extreme environmental conditions (heavy rain, dense fog, or low light), which may introduce noise into the RL feature vector.
- **ROI Dependency**: Vehicle counting and queue length measurement depend on manually defined Regions of Interest (ROIs) for specific lanes.

## 💻 Hardware & Performance
- **Inference Latency**: Real-time control requires low latency (<30ms). Running the full stack (YOLOv10-X + BoT-SORT + ST-GNN) at 30 FPS requires high-end edge hardware (e.g., NVIDIA Jetson Orin).
- **Fallback Modes**: On lower-end hardware, the system must fall back to YOLOv10-S/N, which reduces detection range and accuracy for distant vehicles.
- **GPU Saturation**: Simultaneous training and high-resolution video inference can saturate GPU memory, leading to frame drops in the perception-control loop.

## 📂 Project & Legacy
- **Legacy Material**: Some archived files in the `archive/` branch contain outdated claims regarding patent readiness or leaderboard rankings. These should be treated as legacy draft material.
- **SUMO Port Conflicts**: SUMO/TraCI tests are port-sensitive and will fail if a previous process is not terminated cleanly or if another application holds the target port.
- **Sim-to-Real Gap**: Driver behavior models in SUMO (IDM/Krauss) do not capture non-standard real-world behaviors such as lane splitting, illegal U-turns, or mixed vehicle types (e.g., rickshaws).
