# Main Results Summary

- Generation mode: `quick`
- Core story: robust multi-agent traffic control under non-stationarity with anomaly-aware proactive adaptation.

## Benchmark Table

      model  mean_reward  mean_throughput_veh_per_h  mean_travel_time_s  mean_waiting_time_s  mean_queue_length_vehicles
MAPPO-STGNN  -939.229770                     2108.0           50.720114         44721.148333                  316.338333
MaxPressure  -953.907471                     2108.0           50.720114         44721.148333                  316.338333
 PressLight  -995.012024                     2108.0           50.720114         44721.148333                  316.338333
    CoLight  -930.293579                     2108.0           50.720114         44721.148333                  316.338333
   NSTLight  -971.757385                     2108.0           50.720114         44721.148333                  316.338333
  FixedTime  -975.661658                     2108.0           50.720114         44721.148333                  316.338333
     Random  -941.172385                        0.0            0.000000             0.000000                    0.000000

## Statistical Reporting

- Statistical table: `C:\Users\Kiruthik Kumar M\cap\results\statistical_summary.csv`

## Fairness Checklist

                        criterion status                       evidence
              Same episode budget   PASS                    episodes=10
          Same evaluation horizon   PASS      Single phase1 config used
Same observation/reward interface   PASS Unified evaluation entrypoints

## Hard-Nosed Failure Reporting

- Baseline-win scenario: `CoLight` exceeds `MAPPO-STGNN` on mean reward (-930.294 vs -939.230).
- Identified degradation mode: `mappo` shows 27.49% waiting-time increase under stress.
- Mitigation plan: apply adaptive anomaly threshold + noise-aware observation masking, then rerun stress benchmark and target >=20% reduction in waiting-time increase for the worst-case model.

## Limitations

- Results depend on checkpoint-config compatibility and SUMO runtime consistency.
- Full ablation completion is required before publication claims.
- Include at least one scenario where a baseline wins.