# Smart Traffic Management System - 3-Phase Project Plan

## Overview
This project implements an intelligent traffic management system using Graph Neural Networks (GNNs) and AI techniques. The system is organized into three phases, building from traffic prediction/control to anomaly detection and advanced routing features.

**Research & Patent Focus**: This project is designed to be **publication-ready** and **patent-worthy**. For detailed research methodology, patent strategy, and contributions analysis, see:
- **[RESEARCH_PAPER_PLAN.md](RESEARCH_PAPER_PLAN.md)** - Complete research paper plan with methodology
- **[PATENT_ANALYSIS.md](PATENT_ANALYSIS.md)** - Patent strategy and claims
- **[CONTRIBUTIONS_MATRIX.md](CONTRIBUTIONS_MATRIX.md)** - Novelty and impact analysis

---

## Phase 1: Traffic Prediction & Adaptive Control using GNN + RL
**Status:** 🟡 To be implemented (based on Smartcities_final.pdf approach)

### Objective
Develop an adaptive traffic light control system that uses Graph Neural Networks (GNNs) to model spatial relationships between intersections and Deep Q-Networks (DQN) for reinforcement learning-based signal control.

### Key Components

#### 1.1 Graph Construction Module
- **Nodes**: Traffic intersections/signalized junctions
- **Edges**: Road segments connecting intersections
- **Node Features**:
  - Current signal phase (one-hot encoded)
  - Time spent in current phase
  - Queue lengths (sum and max)
  - Total waiting time (normalized)
  - Vehicle counts per incoming lane
- **Edge Features** (optional):
  - Mean speed
  - Vehicle density
  - Distance between intersections

#### 1.2 GNN Encoder (Spatial Modeling)
- **Architecture**: Graph Convolutional Network (GCN) or Graph Attention Network (GAT)
- **Purpose**: Capture spatial dependencies between intersections
- **Output**: Node embeddings representing local + neighboring traffic states
- **Implementation**: Using PyTorch Geometric (GCNConv/GATConv)

#### 1.3 DQN Agent (Reinforcement Learning)
- **Algorithm**: Deep Q-Network (DQN) from Stable Baselines3
- **Action Space**: Multi-Discrete (one action per intersection)
  - Each intersection can select from available signal phases
- **State Space**: GNN-generated node embeddings
- **Reward Function**: Multi-objective
  - Negative weighted sum of:
    - Queue lengths (network-wide)
    - Waiting times (network-wide)
    - Fuel consumption (optional)
    - Average vehicle speed (optional)
- **Training Techniques**:
  - Experience replay buffer
  - Target networks for stability
  - Epsilon-greedy exploration

#### 1.4 SUMO Integration
- **Simulator**: SUMO (Simulation of Urban Mobility)
- **Interface**: TraCI API for real-time control
- **Network**: 2×2 grid (expandable to larger networks)
- **Vehicle Behavior**: Realistic traffic flow simulation

### Implementation Structure
```
src/
├── phase1/
│   ├── __init__.py
│   ├── graph_builder.py          # Build traffic network graph
│   ├── feature_extractor.py      # Extract node/edge features from SUMO
│   ├── gnn_encoder.py            # GCN/GAT encoder for spatial modeling
│   ├── dqn_agent.py              # DQN agent using SB3
│   ├── traffic_env.py            # SUMO + TraCI environment wrapper
│   ├── reward_calculator.py      # Multi-objective reward function
│   └── train_rl.py               # Training loop for DQN
```

### Expected Outcomes
- Reduced average vehicle waiting time (target: 5-10% improvement)
- Reduced queue lengths (target: 6-8% improvement)
- Better coordination across multiple intersections
- Adaptive response to changing traffic conditions

### Metrics
- Average waiting time per vehicle
- Average queue length per intersection
- Travel time reduction
- Reward score trajectory
- Loss function progression

---

## Phase 2: Anomaly Detection using ST-GNN
**Status:** ✅ Implemented (current codebase)

### Objective
Detect traffic anomalies and incidents using self-supervised spatio-temporal GNNs trained on normal traffic patterns.

### Current Implementation

#### 2.1 Model Architecture
- **Spatial Encoder**: GATv2Conv layers for spatial dependencies
- **Temporal Encoder**: GRU or Transformer for temporal patterns
- **Dual Heads**:
  - Reconstruction head: Reconstructs current state
  - Forecasting head: Predicts future traffic states

#### 2.2 Training Approach
- **Self-supervised**: Trained on normal traffic data
- **Input masking**: Random masking for robustness
- **Loss**: Combined reconstruction + forecasting loss

#### 2.3 Anomaly Detection
- **Scoring**: Reconstruction error + forecasting error
- **Thresholding**: Quantile-based threshold selection
- **Smoothing**: Moving average for score stability

### Current Files
- `src/models/st_gnn.py`: ST-GNN model
- `src/training/train.py`: Training pipeline
- `src/utils/metrics.py`: Anomaly evaluation metrics
- `src/dashboard/app.py`: Visualization dashboard

### Metrics
- Precision, Recall, F1-score
- ROC-AUC
- False alarm rate
- Detection lead time

### Future Enhancements (Phase 2.5)
- [ ] Real-time incident injection in SUMO
- [ ] Multi-type anomaly detection (accidents, congestion, equipment failure)
- [ ] Integration with Phase 1 for proactive control
- [ ] Alert system with severity levels

---

## Phase 3: Advanced Features & Expandable AI
**Status:** 🔵 Planned

### 3.1 Alternate Path Routing & Shortest Time Calculation

#### Objective
Provide real-time alternate route recommendations and shortest-time path calculations based on current traffic conditions and predicted anomalies.

#### Components

**3.1.1 Dynamic Route Planner**
- **Input**: Origin, destination, current traffic state, anomaly predictions
- **Algorithm**: A* or Dijkstra with dynamic weights
- **Weight Factors**:
  - Current traffic density (from Phase 1/2)
  - Predicted congestion (from Phase 1 forecasting)
  - Anomaly scores (from Phase 2)
  - Historical travel times
  - Road capacity
- **Output**: Top-K alternate routes with estimated travel times

**3.1.2 Shortest Time Calculator**
- Real-time calculation considering:
  - Current speeds from SUMO
  - Predicted speeds from Phase 1 GNN
  - Signal timing from Phase 1 RL agent
  - Incident delays from Phase 2
- Updates dynamically as conditions change

**3.1.3 Route Recommendation Engine**
- Multi-criteria optimization:
  - Shortest time
  - Shortest distance
  - Least congestion
  - Avoid anomalies
- Personalized preferences (user-defined weights)

#### Implementation Structure
```
src/
├── phase3/
│   ├── __init__.py
│   ├── route_planner.py          # Dynamic route planning
│   ├── path_finder.py            # A*/Dijkstra implementation
│   ├── travel_time_estimator.py  # Time calculation with predictions
│   ├── route_recommender.py      # Multi-criteria recommendation
│   └── integration.py            # Integrate Phase 1 & 2 outputs
```

### 3.2 Additional Expandable Features

#### 3.2.1 Predictive Congestion Management
- **Early Warning System**: Use Phase 2 anomalies to predict congestion
- **Proactive Signal Control**: Adjust Phase 1 signals before congestion occurs
- **Dynamic Speed Limits**: Suggest speed adjustments based on predictions

#### 3.2.2 Multi-Modal Traffic Integration
- **Public Transit**: Integrate bus/train schedules
- **Pedestrian Flow**: Consider pedestrian crossings in signal timing
- **Cyclist Routes**: Include bike lanes in routing

#### 3.2.3 Energy & Emissions Optimization
- **Fuel Consumption Model**: Estimate fuel savings from optimized routes
- **Emissions Tracking**: CO2 reduction metrics
- **Green Routing**: Routes optimized for lower emissions

#### 3.2.4 Real-Time Dashboard & API
- **Unified Dashboard**: Combine Phase 1, 2, and 3 visualizations
- **REST API**: For mobile apps and third-party integrations
- **WebSocket**: Real-time updates for live monitoring
- **Historical Analytics**: Trend analysis and reporting

#### 3.2.5 Edge Cases & Robustness
- **Network Failures**: Graceful degradation when sensors fail
- **Extreme Events**: Handling accidents, weather, events
- **Scalability**: Support for city-wide deployment
- **Privacy**: Anonymized data handling

### 3.3 Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Phase 3: Advanced Features          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Route        │  │ Predictive   │  │ Multi-Modal  │ │
│  │ Planner      │  │ Congestion   │  │ Integration  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                           │
│                    ┌───────▼────────┐                  │
│                    │ Integration    │                  │
│                    │ Layer          │                  │
│                    └───────┬────────┘                  │
│                            │                           │
└────────────────────────────┼───────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│ Phase 1:       │  │ Phase 2:       │  │ SUMO/Real      │
│ GNN + RL       │  │ Anomaly        │  │ Data Sources   │
│ Control        │  │ Detection      │  │                │
└────────────────┘  └────────────────┘  └────────────────┘
```

---

## Project Timeline & Milestones

### Phase 1 Implementation (Weeks 1-4)
- [ ] Week 1: Graph construction and feature extraction from SUMO
- [ ] Week 2: GNN encoder implementation and testing
- [ ] Week 3: DQN agent setup and reward function design
- [ ] Week 4: Training and evaluation on 2×2 grid

### Phase 2 Refinement (Weeks 5-6)
- [ ] Week 5: Integration with SUMO incident simulation
- [ ] Week 6: Dashboard improvements and real-time monitoring

### Phase 3 Development (Weeks 7-12)
- [ ] Week 7-8: Route planner and path finder implementation
- [ ] Week 9: Travel time estimator with Phase 1/2 integration
- [ ] Week 10: Route recommendation engine
- [ ] Week 11: Predictive congestion management
- [ ] Week 12: Unified dashboard and API development

---

## Technical Stack

### Core Libraries
- **PyTorch**: Neural network framework
- **PyTorch Geometric**: GNN implementations
- **Stable Baselines3**: Reinforcement learning algorithms
- **SUMO**: Traffic simulation
- **TraCI**: SUMO control interface

### Additional for Phase 3
- **NetworkX**: Graph algorithms for routing
- **OSMnx**: OpenStreetMap data integration
- **FastAPI**: REST API framework
- **Streamlit**: Dashboard (already in use)
- **PostgreSQL/InfluxDB**: Time-series data storage (optional)

---

## Data Requirements

### Phase 1
- SUMO network files (.net.xml)
- Route files (.rou.xml)
- Traffic demand patterns
- Signal phase definitions

### Phase 2
- Historical traffic data (normal conditions)
- Incident labels (optional, for evaluation)
- Real-time SUMO simulation data

### Phase 3
- Road network topology (OSM or SUMO)
- Historical travel time data
- Real-time traffic feeds
- Public transit schedules (optional)

---

## Evaluation & Testing

### Phase 1 Metrics
- Comparison with fixed-time controllers
- Comparison with other RL baselines
- Network-wide performance metrics
- Convergence analysis

### Phase 2 Metrics
- Anomaly detection accuracy (precision/recall)
- False alarm rate
- Detection lead time
- Real-time performance

### Phase 3 Metrics
- Route recommendation accuracy
- Travel time prediction error
- User satisfaction (if applicable)
- System scalability

---

## Future Research Directions

1. **Federated Learning**: Privacy-preserving multi-city learning
2. **Transfer Learning**: Adapt models to new cities quickly
3. **Explainable AI**: Interpretable decision-making for traffic control
4. **Multi-Agent RL**: Coordinated control across multiple agents
5. **Edge Computing**: Deploy models on edge devices for low latency
6. **Integration with V2X**: Vehicle-to-everything communication

---

## References

- Smartcities_final.pdf: Adaptive Traffic Light System using GNNs and RL
- Current codebase: Phase 2 anomaly detection implementation
- Literature review: `docs/literature_review.md`

---

## Notes

- Phase 1 should align with the approach described in Smartcities_final.pdf
- Phase 2 is already implemented but can be enhanced
- Phase 3 is designed to be modular and expandable
- All phases should integrate seamlessly for a complete system
