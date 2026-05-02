# Known Limitations

- Phase 1 smoke tests use a short SUMO horizon and are not publication-grade benchmarks.
- The compatible PPO checkpoint is `marl_ppo_traffic.zip`; `outputs/phase1/dqn_traffic_final.zip` is currently incompatible with `configs/phase1.yaml`.
- Phase 2 anomaly metrics are computed from synthetic data unless a real labeled dataset is provided.
- Full ablation, scalability, and zero-shot generalization tables are still evidence gaps unless regenerated from completed runs.
- Some archived markdown files contain old claims such as complete status, patent readiness, and leaderboard language. Treat `archive/root_markdown/` as legacy draft material.
- SUMO/TraCI tests are runtime-sensitive and can fail if another SUMO process holds a port.
