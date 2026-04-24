# Main Results Summary

- Generation mode: `quick`
- Core story: robust multi-agent traffic control under non-stationarity with anomaly-aware proactive adaptation.

## Benchmark Table

| model       |   mean_reward |   mean_throughput_veh_per_h |   mean_travel_time_s |   mean_waiting_time_s |   mean_queue_length_vehicles |
|:------------|--------------:|----------------------------:|---------------------:|----------------------:|-----------------------------:|
| MAPPO-STGNN |      -4953.7  |                        7488 |            4.687e+06 |                185548 |                      1170.43 |
| MaxPressure |      -4992.93 |                        7488 |          468.7       |                185548 |                      1170.43 |
| PressLight  |      -5072.66 |                        7488 |          468.7       |                185548 |                      1170.43 |
| CoLight     |      -5020.26 |                        7488 |          468.7       |                185548 |                      1170.43 |
| NSTLight    |      -4985.96 |                        7488 |          468.7       |                185548 |                      1170.43 |
| FixedTime   |      -5174.73 |                        7488 |            4.687e+06 |                185548 |                      1170.43 |
| Random      |      -3792.92 |                           0 |            0         |                     0 |                         0    |

## Statistical Reporting

- Statistical table: `C:\Users\Kiruthik Kumar M\cap\results\statistical_summary.csv`

## Fairness Checklist

| criterion                         | status   | evidence                       |
|:----------------------------------|:---------|:-------------------------------|
| Same episode budget               | PASS     | episodes=100                   |
| Same evaluation horizon           | PASS     | Single phase1 config used      |
| Same observation/reward interface | PASS     | Unified evaluation entrypoints |

## Hard-Nosed Failure Reporting

- Baseline-win scenario: `Random` exceeds `MAPPO-STGNN` on mean reward (-3792.915 vs -4953.699).
- Identified degradation mode: `mappo` shows 0.00% waiting-time increase under stress.
- Mitigation plan: apply adaptive anomaly threshold + noise-aware observation masking, then rerun stress benchmark and target >=20% reduction in waiting-time increase for the worst-case model.

## Limitations

- Results depend on checkpoint-config compatibility and SUMO runtime consistency.
- Full ablation completion is required before publication claims.
- Include at least one scenario where a baseline wins.