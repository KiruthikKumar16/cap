# Baseline Enhancement Progress Report (Phase 4)
**Date:** April 9, 2026
**Status:** 85% Complete (Phase 4 Benchmarking Active)

## 1. Accomplishments (Completed)
We have successfully evolved the project into a state-of-the-art research framework.

*   **Baseline Upgrade (NSTLight)**:
    *   Integrated **NSTLight (2025/2026 Standard)** as our primary competitor.
    *   Phased out legacy PressLight/CoLight benchmarks to meet high-impact presentation standards.
*   **Adversarial Resilience (Stress Test)**:
    *   Developed `scripts/accident_injection.py`: Simulates a central gridlock crash by artificially halting 5 vehicles at step 500.
    *   Implemented **Sensor Failure Simulation**: Added a 10% Gaussian/Masking noise wrapper to the MAPPO observation space to test recovery under uncertainty.
*   **Zero-Shot Bengaluru Generalization**:
    *   Created `scripts/evaluate_generalization.py`: Validates performance transfer from Synthetic Grid (Training) to Unseen Bengaluru OSM Map (Testing) without retraining.
*   **Architecture Refactor**:
    *   Patched `src/phase1/traffic_env.py` and `evaluate.py` to fix configuration attribute scope issues, ensuring stable multi-threaded evaluation.

## 2. Technical Debt & To-Do (Remaining)
*   **Modern Visualizations**: 
    *   [ ] Generate spatio-temporal congestion heatmaps showing "Congestion Propagation Waves".
    *   [ ] Render t-SNE clusters of the ST-GNN Autoencoder's latent space to visualize "Crisis Latents".
*   **Hardware Benchmarking**:
    *   [ ] Log inference latency (ms/step) on CUDA to prove real-time viability.
*   **Documentation Polish**:
    *   [ ] Rewrite README.md Abstract to emphasize "Traffic Resilience" and "Risk-Aware Multi-Agent Control".

## 3. Git Parity Status
All files are staged for `git push`.
*   Modified: `src/phase1/traffic_env.py`, `src/phase1/evaluate.py`, `src/phase1/train_rl.py`
*   New: `src/baselines/nstlight.py`, `scripts/accident_injection.py`, `scripts/evaluate_generalization.py`
