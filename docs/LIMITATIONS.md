# Limitations and Failure Cases

## Current Limitations

- Performance depends on checkpoint/config compatibility (now guarded by metadata-based validation in training/evaluation/dashboard prechecks).
- Some stress and ablation runs are still template-driven and require full execution to finalize publication claims.
- Cross-city and topology-transfer evidence is still limited to available scenarios.

## Hard-Nosed Failure Requirements

Publication version must include:
- At least one scenario where a baseline outperforms the primary method.
- One identified degradation mode with quantified impact.
- One mitigation plan tied to measurable follow-up experiment.

## Practical Failure Modes to Report

- Observation-space mismatch between checkpoint and environment.
- Performance drop under severe sensor corruption.
- Stability loss under sudden demand shocks.
- Communication overhead growth in larger CTDE setups.
