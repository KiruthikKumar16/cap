# Activity Chart & Gantt Chart
## Detailed Timeline with Milestones and Deliverables

---

## Visual Gantt Chart (Text-Based)

```
Week:    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
         │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
Phase 1: ████████████████████████████████
         │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
         M1   M2   │    │    │    │    │    │    │    │    │    │    │    │    │    │
         │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
Phase 2:         ████████████████████████████████████████████████████████████████████
                 │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
                 │    │    │    │    M3   │    │    │    │    │    │    │    │    │    │
                 │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
Phase 3:                 ████████████████████████████████████████████████████████████████
                         │    │    │    │    │    │    │    │    │    │    │    │    │    │
                         │    │    │    │    │    │    │    M4   │    │    │    │    │    │
                         │    │    │    │    │    │    │    │    │    │    │    │    │    │
Evaluation:                      ████████████████████████████████████████████████████████
                                  │    │    │    │    │    │    │    │    │    │    │    │    │
                                  │    │    │    │    │    │    │    │    │    │    M5   │    │
                                  │    │    │    │    │    │    │    │    │    │    │    │    │
Documentation:                                        ████████████████████████████████████████
                                                      │    │    │    │    │    │    │    │    │    │
                                                      │    │    │    │    │    │    │    │    M6   │
                                                      │    │    │    │    │    │    │    │    │    │
Patent:                                               ████████████████████████████████████████
                                                      │    │    │    │    │    │    │    │    │    │
                                                      │    │    │    │    │    │    │    M7   │    │
                                                      │    │    │    │    │    │    │    │    │    │
Final Submission:                                                          ████████████████████████
                                                                           │    │    │    │    │    │
                                                                           │    │    │    │    │    M8
```

**Legend**:
- M1-M8: Milestones
- ████: Activity duration
- │: Week markers

---

## Detailed Activity Breakdown

### Phase 1: Foundation & Literature Review (Weeks 1-2)

#### Week 1: Literature Review & Planning
**Activities**:
- [x] Literature review (>50 papers)
- [x] Gap identification (6 gaps)
- [x] Research questions formulation
- [x] Project plan creation
- [x] Requirements specification

**Deliverables**:
- ✅ Literature review document (`docs/literature_review.md`)
- ✅ Gap analysis document (`RESEARCH_PAPER_PLAN.md` Section 1.2)
- ✅ Project plan (`PROJECT_PLAN.md`)
- ✅ Requirements specification (`CAPSTONE_EVALUATION_CHECKLIST.md` Section 5)

**Milestone M1**: Literature Review Complete ✅

---

#### Week 2: Environment Setup & Graph Construction
**Activities**:
- [ ] SUMO installation and configuration
- [ ] Python environment setup (PyTorch, PyG, SB3)
- [ ] Graph construction module implementation
- [ ] Feature extraction module implementation
- [ ] Initial testing on 2×2 grid

**Deliverables**:
- Code: `src/phase1/graph_builder.py`
- Code: `src/phase1/feature_extractor.py`
- Test results: Graph construction validation
- Documentation: Setup guide

**Milestone M2**: Environment Setup Complete

---

### Phase 2: GNN-RL Implementation (Weeks 3-5)

#### Week 3: GNN Encoder & DQN Setup
**Activities**:
- [ ] GNN encoder implementation (GAT/GCN)
- [ ] DQN agent setup (Stable Baselines3)
- [ ] Baseline reward function implementation
- [ ] Environment wrapper (SUMO + TraCI)
- [ ] Initial training pipeline

**Deliverables**:
- Code: `src/phase1/gnn_encoder.py`
- Code: `src/phase1/dqn_agent.py`
- Code: `src/phase1/traffic_env.py`
- Code: `src/phase1/train_rl.py`
- Documentation: Architecture design

---

#### Week 4: Training & Baseline Evaluation
**Activities**:
- [ ] Training pipeline completion
- [ ] Hyperparameter tuning
- [ ] Baseline evaluation (fixed-time controller)
- [ ] Performance metrics collection
- [ ] Initial results analysis

**Deliverables**:
- Trained model: `outputs/phase1/baseline_model.zip`
- Results: Baseline comparison metrics
- Documentation: Training guide
- Report: Phase 1 initial results

---

#### Week 5: Comparison & Ablation Studies
**Activities**:
- [ ] Comparison with baseline DQN (without GNN)
- [ ] Comparison with actuated controller
- [ ] Ablation study: without GNN (MLP-based)
- [ ] Performance optimization
- [ ] Mid-term review preparation

**Deliverables**:
- Results: Comparison tables
- Results: Ablation study results
- Report: Phase 1 complete results
- Presentation: Mid-term review slides

**Milestone M3**: GNN-RL Implementation Complete

---

### Phase 3: Anomaly Detection Integration (Weeks 6-8)

#### Week 6: ST-GNN Anomaly Detection
**Activities**:
- [ ] ST-GNN model implementation (dual-head)
- [ ] Self-supervised training pipeline
- [ ] Anomaly scoring implementation
- [ ] Threshold selection methods
- [ ] Anomaly detection evaluation

**Deliverables**:
- Code: Enhanced `src/models/st_gnn.py`
- Code: `src/phase2/anomaly_trainer.py`
- Trained model: `outputs/phase2/anomaly_model.ckpt`
- Results: Anomaly detection metrics (F1, precision, recall)

---

#### Week 7: Integration & Anomaly-Aware Rewards
**Activities**:
- [ ] Integration of Phase 1 and Phase 2
- [ ] Anomaly-aware reward function implementation
- [ ] End-to-end system testing
- [ ] Integration testing
- [ ] Bug fixes and optimization

**Deliverables**:
- Code: `src/phase3/integration.py`
- Code: Enhanced `src/phase1/reward_calculator.py`
- Test results: Integration test results
- Documentation: Integration guide

---

#### Week 8: Combined System Training
**Activities**:
- [ ] Combined system training
- [ ] Performance evaluation
- [ ] Comparison with separate systems
- [ ] Error analysis
- [ ] System optimization

**Deliverables**:
- Trained model: `outputs/phase3/integrated_model.zip`
- Results: Integrated system performance
- Report: Integration results
- Documentation: System architecture

**Milestone M4**: Anomaly Detection Integrated

---

### Phase 4: Comprehensive Evaluation (Weeks 9-11)

#### Week 9: Multiple Baseline Comparisons
**Activities**:
- [ ] Comparison with CoLight (multi-agent RL)
- [ ] Comparison with PressLight (max-pressure)
- [ ] Comparison with other SOTA methods
- [ ] Statistical significance testing
- [ ] Performance metrics collection

**Deliverables**:
- Results: Comprehensive comparison tables
- Charts: Performance comparison charts
- Report: Baseline comparison report
- Statistical analysis: Significance test results

---

#### Week 10: Scalability & Real-World Testing
**Activities**:
- [ ] Scalability testing (4×4, 6×6, larger grids)
- [ ] Real-world data testing (if available)
- [ ] Computational performance analysis
- [ ] Edge case testing
- [ ] Stress testing

**Deliverables**:
- Results: Scalability analysis
- Results: Performance benchmarks
- Report: Scalability report
- Documentation: Deployment guide

---

#### Week 11: Results Analysis & Visualization
**Activities**:
- [ ] Results analysis and interpretation
- [ ] Case studies
- [ ] Visualization creation (charts, graphs)
- [ ] Error analysis
- [ ] Final performance optimization

**Deliverables**:
- Visualizations: Performance charts
- Report: Comprehensive results report
- Case studies: Detailed case study reports
- Documentation: Results interpretation

**Milestone M5**: Comprehensive Evaluation Complete

---

### Phase 5: Documentation & Submission (Weeks 12-16)

#### Week 12: Research Paper Writing (Draft)
**Activities**:
- [ ] Abstract writing
- [ ] Introduction section
- [ ] Related work section
- [ ] Methodology section
- [ ] Figures and tables preparation

**Deliverables**:
- Paper: Draft v1.0
- Figures: All paper figures
- Tables: All paper tables
- References: Complete bibliography

---

#### Week 13: Paper Completion & Review
**Activities**:
- [ ] Results section writing
- [ ] Discussion and analysis
- [ ] Conclusion and future work
- [ ] Paper review and revision
- [ ] Formatting (IEEE format)

**Deliverables**:
- Paper: Draft v2.0 (complete)
- Review: Peer review comments
- Revisions: Revised sections

**Milestone M6**: Paper Draft Complete

---

#### Week 14: Patent Application & Code Documentation
**Activities**:
- [ ] Patent application preparation
- [ ] Provisional patent filing
- [ ] Code documentation (docstrings)
- [ ] Open-source repository setup
- [ ] README and user guides

**Deliverables**:
- Patent: Provisional patent application
- Code: Fully documented codebase
- Repository: GitHub repository
- Documentation: User guides

**Milestone M7**: Patent Filed

---

#### Week 15: Final Revisions & Submission
**Activities**:
- [ ] Final paper revisions
- [ ] Submission to target venue
- [ ] Presentation preparation
- [ ] Project documentation completion
- [ ] Final code cleanup

**Deliverables**:
- Paper: Final version (submitted)
- Presentation: Project presentation slides
- Documentation: Complete project documentation
- Code: Final codebase version

---

#### Week 16: Final Review & Delivery
**Activities**:
- [ ] Final review and corrections
- [ ] Project submission
- [ ] Presentation delivery
- [ ] Post-submission follow-up
- [ ] Project closure

**Deliverables**:
- Project: Complete project submission
- Presentation: Delivered presentation
- Report: Final project report
- Archive: Project archive

**Milestone M8**: Final Submission Complete

---

## Milestone Summary Table

| Milestone | Week | Deliverable | Status | Dependencies |
|-----------|------|-------------|--------|--------------|
| **M1**: Literature Review | 1 | Review document | ✅ Done | None |
| **M2**: Environment Setup | 2 | Setup complete | ⏳ Planned | M1 |
| **M3**: GNN-RL Complete | 5 | Working system | ⏳ Planned | M2 |
| **M4**: Anomaly Integrated | 8 | Integrated system | ⏳ Planned | M3 |
| **M5**: Evaluation Complete | 11 | Evaluation results | ⏳ Planned | M4 |
| **M6**: Paper Draft | 13 | Research paper | ⏳ Planned | M5 |
| **M7**: Patent Filed | 14 | Patent application | ⏳ Planned | M6 |
| **M8**: Final Submission | 16 | Complete project | ⏳ Planned | M7 |

---

## Critical Path Analysis

**Critical Path**: M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8

**Critical Activities** (on critical path):
1. Literature review (Week 1)
2. Graph construction (Week 2)
3. GNN-RL implementation (Weeks 3-5)
4. Anomaly integration (Weeks 6-8)
5. Comprehensive evaluation (Weeks 9-11)
6. Paper writing (Weeks 12-13)
7. Patent filing (Week 14)
8. Final submission (Week 16)

**Non-Critical Activities** (have slack):
- Dashboard improvements (can be done in parallel)
- Additional features (can be deferred)
- Extended testing (can be done in parallel)

---

## Risk Mitigation Timeline

**Week 1-2**: Identify risks early
- Technical feasibility validation
- Resource availability check
- Data availability confirmation

**Week 3-5**: Monitor implementation risks
- Training stability issues
- Performance bottlenecks
- Integration challenges

**Week 6-8**: Address integration risks
- Component compatibility
- Performance degradation
- Error handling

**Week 9-11**: Mitigate evaluation risks
- Baseline comparison issues
- Statistical significance concerns
- Scalability problems

**Week 12-16**: Manage documentation risks
- Paper quality concerns
- Patent filing deadlines
- Submission requirements

---

## Resource Allocation

### Human Resources
- **Week 1-5**: Full-time development (40 hours/week)
- **Week 6-11**: Full-time development + evaluation (40 hours/week)
- **Week 12-16**: Full-time documentation (40 hours/week)

### Computational Resources
- **Week 1-2**: CPU only (setup, literature review)
- **Week 3-8**: GPU required (training)
- **Week 9-11**: GPU + CPU (evaluation)
- **Week 12-16**: CPU only (documentation)

### Budget Estimate
- **GPU Cloud**: $500-1000 (for training)
- **Software**: $0 (open-source)
- **Patent**: $2000-3000 (provisional)
- **Total**: $2500-4000

---

## Quality Checkpoints

**Checkpoint 1** (Week 2): Environment setup validated
**Checkpoint 2** (Week 5): GNN-RL working correctly
**Checkpoint 3** (Week 8): Integration successful
**Checkpoint 4** (Week 11): Evaluation complete
**Checkpoint 5** (Week 13): Paper quality review
**Checkpoint 6** (Week 15): Final submission review

---

## Success Criteria per Phase

### Phase 1 Success Criteria
- ✅ Literature review complete (>50 papers)
- ✅ 6 gaps identified
- ✅ Graph construction working
- ✅ GNN encoder implemented

### Phase 2 Success Criteria
- ✅ GNN-RL system training successfully
- ✅ 10%+ improvement over baseline
- ✅ Baseline comparisons complete

### Phase 3 Success Criteria
- ✅ Anomaly detection integrated
- ✅ Anomaly-aware rewards working
- ✅ End-to-end system functional

### Phase 4 Success Criteria
- ✅ Comprehensive evaluation complete
- ✅ 15-25% improvement demonstrated
- ✅ Statistical significance confirmed

### Phase 5 Success Criteria
- ✅ Research paper submitted
- ✅ Patent application filed
- ✅ Project documentation complete

---

**Last Updated**: [Current Date]
**Status**: Ready for implementation
