# Project Verification & SOTA Status Report

This report provides a critical analysis of the current project state, output legitimacy, and the authenticity of the "NSTLight" baseline.

## 🔍 Executive Summary

> [!WARNING]
> **Project Status: INCOMPLETE / INACCURATE BENCHMARKS**
> While the code framework is functional, the current benchmark results are **invalid** due to zeroed-out metrics for the primary MAPPO model and the use of a non-functional dummy baseline for NSTLight.

---

## 1. NSTLight Baseline Legitimacy
The user's suspicion that the NSTLight used is not legit is **CORRECT**.

- **Code Inspection**: File `src/baselines/nstlight.py` is explicitly commented as a "dummy baseline agent."
- **Internal Logic**: It uses a standard GAT encoder but uses an **untrained Linear layer** for its action head. It is essentially a random policy with a slight pressure bias.
- **Real-World Context**: A real "NSTLight" (Non-Stationary Traffic Light) research paper was indeed published in January 2025. However, the implementation in this repository **does not match** the research; it is a placeholder.

---

## 2. Output Legitimacy Verification
The outputs in `outputs/` are currently **not research-legit**.

### Benchmark Data Analysis (`benchmark_results.json`)
| Model | Throughput | Waiting Time | Queue Length | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **MAPPO-STGNN (Ours)** | **0.0** | **0.0** | **0.0** | ❌ **FAILED** |
| **NSTLight (Dummy)** | 354.0 | 80304.7 | 500.1 | ⚠️ Low Performance |
| **Fixed-Time** | 354.0 | 80304.7 | 500.1 | ⚠️ Generic Baseline |

### Why are there Zeros?
The `MAPPO-STGNN` model shows zeros because:
1. The `evaluate_sb3_agent` function in `src/phase1/evaluate.py` may be failing to extract metrics from the `MARLTrafficEnv` wrapper.
2. The system may have defaulted to a "placeholder mode" during evaluation when it couldn't connect to a real SUMO instance for that specific agent.

---

## 3. Comparison with Actual SOTA Models
Since our current baseline (NSTLight Dummy) is untrained, the "Ours is better" claim in the generated charts is currently unsubstantiated.

### Better SOTA Models to Consider (Actual 2025/2026 Models):
1. **GTLight (2024/2025)**: Uses Graph Transformers for network-wide coordination. Generally outperforms basic GNN-based MAPPO.
2. **TransferLight (2025)**: Focused on zero-shot generalization across different cities using meta-reinforcement learning.
3. **MPLight (SOTA Baseline)**: A classic strong baseline using Max-Pressure with RL. Our current model should be compared against a *real* MPLight implementation.
4. **ResilienceNet (2025)**: Specifically designed for traffic resilience under sensor failures (directly competing with our ST-GNN objective).

---

## 4. Completion Status Checklist

- [x] **Infrastructure**: 100% (Scripts, training loops, and environment wrappers are all present).
- [ ] **Model Performance**: 40% (Model trains, but metrics are not being recorded/captured correctly in evaluation).
- [ ] **Validation Legitimacy**: 10% (Baselines are dummies; comparisons are currently "WOW" but not statistically valid).

## 💡 Recommended Next Steps (If permitted)
1. **Fix Metric Collection**: Update `evaluate.py` to correctly pull lane-level metrics from the MARL wrapper to eliminate the 0.0 values.
2. **Implement Real Baselines**: Replace the dummy NSTLight with a functional Actuated controller or a pre-trained SOTA checkpoint.
3. **Re-run Valid Benchmarks**: Execute a clean evaluation run to get real numbers for the presentation.

---
**Conclusion**: The project demonstrates a high-quality "Shell" and "Visual Dashboard", but the underlying data for the Capstone is currently based on broken metrics and placeholder comparisons.
