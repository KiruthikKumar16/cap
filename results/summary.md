# Main Results Summary

- Generation mode: `quick`
- Status: generated from currently available outputs; missing experiments are reported as gaps.

## Benchmark Table

No benchmark data available.

## Statistical Reporting

- Statistical table: `C:\Users\Kiruthik Kumar M\cap\results\statistical_summary.csv`

## Fairness Checklist

                        criterion status                  evidence
              Same episode budget   PASS                episodes=1
          Same evaluation horizon   PASS Single phase1 config used
Same observation/reward interface  CHECK   benchmark table missing

## Hard-Nosed Failure Reporting

- Baseline-win scenario: not available yet.
- Identified degradation mode: `mappo` shows 0.00% waiting-time increase under stress.
- Mitigation plan: rerun the full stress benchmark, then evaluate whether adaptive anomaly thresholding or noise-aware observation masking improves the measured worst case.

## Limitations

- Results depend on checkpoint-config compatibility and SUMO runtime consistency.
- Full ablation completion is required before publication claims.
- Include at least one scenario where a baseline wins.