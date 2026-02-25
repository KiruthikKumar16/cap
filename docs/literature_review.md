# Literature Review (IEEE-style numbered citations)

This section summarizes recent (last ~5–7 years) work on spatio-temporal graph neural networks (ST-GNNs) for traffic, congestion/incident detection, and SUMO-based simulation, and positions the proposed project.

## Core ST-GNN traffic forecasting
- STGCN introduces graph convolutions plus temporal convolutions for speed/flow forecasting, modeling road networks as graphs [3][4].
- STG4Traffic survey/benchmark shows ST-GNN dominance for traffic prediction and categorizes CNN-, RNN-, and attention-based temporal modules [1].
- BigST targets linear-complexity ST-GNNs for large, long sequences [5].
- SimST questions the necessity of GNNs, proposing a non-GNN spatio-temporal model with competitive accuracy [6].
- STCGCN fuses distance, correlation, and “comfort” matrices for flow prediction [7].
**Takeaway:** These focus on supervised forecasting under normal conditions, not anomaly/incident detection.

## Broader ST-GNN traffic applications
- Congestion DL survey highlights gaps in proactive, network-level congestion management and limited self-/unsupervised approaches [2].
- UAV-based ST-GNN forecasting and pretraining-improved ST-GNNs extend forecasting to new modalities and longer horizons [8][9].
**Takeaway:** ST-GNNs are widely used for flow prediction; anomaly/incident tasks remain less mainstream.

## Incident, congestion, and anomaly-focused works
- Congestion/incident surveys note many CNN/RNN/heuristic methods, few full ST-GNNs over road graphs [2].
- SUMO-based incident propagation prediction builds datasets and models congestion spread after incidents [10].
- Accident-driven congestion with Bayesian Networks + SUMO shows incident-aware modeling but not GNN-based [11]; accident-aware traffic management surveys stress early detection [12].
**Takeaway:** SUMO incidents are studied, but often with non-GNN or supervised models; network spillover is important.

## Self-supervised / anomaly detection gap
- Graph anomaly detection literature covers reconstruction/self-supervision mostly outside traffic [1][2].
- Traffic works often treat anomalies indirectly as forecast residuals; few design ST-GNNs explicitly for self-supervised anomaly detection.
**Takeaway:** Self-supervised ST-GNN reconstruction for traffic incidents is under-explored.

## What makes the proposed project distinct
- Problem: anomaly/incident early warning (alerts, lead time) instead of only speed/flow RMSE.
- Learning: self-supervised reconstruction (masked/perturbed inputs) trained on normal SUMO traffic; minimal incident labels.
- Modeling: network-wide spillover on a road graph (GCN/GAT), temporal GRU/attention, with reconstruction + forecasting heads for anomaly scoring.
- Metrics: precision/recall, ROC-AUC, detection lead time, false alarm rate (operational metrics often missing in forecasting works).
- System: full pipeline OSM → SUMO incidents → ST-GNN → anomaly scores → dashboard with alerts.

## How to integrate in the report
- Related work sections:
  - ST-GNN for forecasting: STGCN, STG4Traffic, BigST, STCGCN, SimST, UAV ST-GNN [3][1][5][7][6][8][9]
  - Congestion/incident detection: congestion DL survey, SUMO incident propagation, BN + SUMO accidents, accident-aware management [2][10][11][12]
- Gap analysis:
  - ST-GNN mainly for forecasting, not self-supervised incident detection [3][1].
  - SUMO incident works seldom use ST-GNN self-supervision [10][11].
  - Lead-time/false-alarm metrics rarely reported.

## References (numbering as cited above)
[1] Liu et al., “Temporal Graph Neural Networks for Traffic Prediction (STG4Traffic),” 2023.  
[2] Kumar & Raubal, “Applications of deep learning in congestion detection, prediction, and alleviation: A survey,” Transp. Res. C, 2021.  
[3] Yu et al., “Spatio-Temporal Graph Convolutional Networks,” IJCAI, 2018.  
[4] Yu et al., “Graph Convolutional Networks for Traffic Forecasting,” arXiv:1709.04875.  
[5] Han et al., “BigST: Linear-Complexity Spatio-Temporal GNN,” PVLDB.  
[6] Liu et al., “Do We Really Need Graph Neural Networks for Traffic Forecasting? (SimST),” 2023.  
[7] STCGCN, “Spatial–Temporal Complex Graph Convolution Network,” Expert Syst. Appl., 2023.  
[8] ST-GNN for UAV-based monitoring, Sci. Reports, 2024.  
[9] Pretraining-improved ST-GNN for Traffic Forecasting, Sci. Reports, 2025.  
[10] Incident congestion propagation with SUMO incidents, SuMob’23.  
[11] Accident-driven congestion prediction with Bayesian Networks + SUMO, 2025 preprint.  
[12] Accident-aware traffic management survey, Nature, 2025.

