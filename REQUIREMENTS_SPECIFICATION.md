# Requirements Specification Document
## Systematic Analysis & Complete Requirements

---

## 1. Introduction

### 1.1 Purpose
This document provides a comprehensive requirements specification for the Predictive-Proactive Traffic Management System using GNN-RL with Self-Supervised Anomaly Detection.

### 1.2 Scope
The system addresses intelligent traffic signal control through:
- Predictive traffic state modeling using Graph Neural Networks
- Proactive signal control using Reinforcement Learning
- Early anomaly detection using self-supervised learning
- Unified integration architecture

### 1.3 Document Structure
1. Problem Analysis
2. Stakeholder Requirements
3. Functional Requirements
4. Non-Functional Requirements
5. System Requirements
6. Constraints and Assumptions
7. Acceptance Criteria

---

## 2. Problem Analysis

### 2.1 Current State Analysis

**Problem Statement**:
Urban traffic congestion is a critical issue caused by:
1. **Inefficient Signal Control**: Fixed-time or reactive controllers cannot adapt to real-time conditions
2. **Lack of Prediction**: Systems respond to current congestion but cannot predict future issues
3. **No Proactive Prevention**: Incidents are detected after occurrence, not prevented
4. **Limited Coordination**: Intersections operate independently without network-wide coordination
5. **Scalability Issues**: Solutions work on small networks but fail at city scale

**Current Solutions**:
- Fixed-time controllers: Simple but inefficient
- Actuated controllers: Better but still reactive
- RL-based controllers: Adaptive but reactive
- Anomaly detection: Detects but doesn't prevent

**Limitations**:
- Reactive nature (respond after congestion occurs)
- No integration between prediction, detection, and control
- Limited scalability
- High computational requirements

### 2.2 Desired State

**Vision**:
An intelligent traffic management system that:
- Predicts future traffic conditions and anomalies
- Proactively adjusts signals before congestion occurs
- Scales to city-wide deployment
- Operates in real-time with low latency
- Reduces congestion by 15-25%

**Key Capabilities**:
1. Real-time traffic state modeling
2. Future state prediction
3. Anomaly detection and prediction
4. Proactive signal control
5. Network-wide coordination
6. Scalable architecture

### 2.3 Gap Analysis

**Gap 1**: Reactive vs. Proactive Control
- **Current**: Systems respond to current state
- **Desired**: Systems predict and prevent future issues
- **Gap**: No proactive control mechanisms

**Gap 2**: Disconnected Components
- **Current**: Separate systems for prediction, detection, control
- **Desired**: Unified integrated system
- **Gap**: No integration architecture

**Gap 3**: Limited Anomaly Awareness
- **Current**: Control systems don't consider anomalies
- **Desired**: Anomaly-aware control decisions
- **Gap**: No anomaly integration in control

**Gap 4**: Self-Supervised Learning Gap
- **Current**: Supervised anomaly detection (needs labels)
- **Desired**: Self-supervised detection (no labels)
- **Gap**: Limited self-supervised approaches

**Gap 5**: Scalability Limitations
- **Current**: Small network solutions (2×2, 4×4)
- **Desired**: City-wide deployment (100+ intersections)
- **Gap**: Scalability challenges

**Gap 6**: Real-time Performance
- **Current**: High latency systems
- **Desired**: Real-time operation (<100ms)
- **Gap**: Performance optimization needed

---

## 3. Stakeholder Requirements

### 3.1 Primary Stakeholders

#### 3.1.1 Municipal Governments
**Needs**:
- Reduced traffic congestion
- Improved air quality
- Cost-effective solutions
- Scalable deployment

**Requirements**:
- 15-25% reduction in congestion
- 10-15% reduction in emissions
- Cost-effective (open-source components)
- Scalable to city-wide deployment

#### 3.1.2 Traffic Management Agencies
**Needs**:
- Real-time monitoring
- Predictive capabilities
- System reliability
- Easy maintenance

**Requirements**:
- Real-time dashboard
- Predictive alerts
- 99.9% uptime
- Modular architecture for maintenance

#### 3.1.3 Citizens
**Needs**:
- Reduced travel time
- Less congestion
- Better air quality
- Reliable transportation

**Requirements**:
- 15-25% reduction in travel time
- Reduced waiting at intersections
- Improved air quality
- Reliable system operation

#### 3.1.4 Transportation Companies
**Needs**:
- Efficient routing
- Reduced fuel consumption
- Cost savings
- Real-time information

**Requirements**:
- Optimized routes
- 10-15% fuel savings
- Real-time traffic information
- API access for integration

### 3.2 Secondary Stakeholders

- **Researchers**: Reproducible research, open-source code
- **Developers**: Well-documented APIs, modular code
- **Regulators**: Compliance with standards, safety

---

## 4. Functional Requirements

### 4.1 Data Acquisition & Processing

#### FR1: Graph Construction
**ID**: FR1
**Priority**: High
**Description**: System must construct graph representation of traffic network
**Input**: SUMO network files (.net.xml) or real traffic data
**Output**: Graph structure (nodes, edges, features)
**Constraints**: Must handle 100+ intersections
**Acceptance**: Graph correctly represents network topology

#### FR2: Feature Extraction
**ID**: FR2
**Priority**: High
**Description**: System must extract real-time traffic features
**Input**: Traffic sensor data or SUMO simulation
**Output**: Normalized feature vectors
**Features**: Queue lengths, waiting times, signal phases, vehicle counts
**Constraints**: <10ms extraction time
**Acceptance**: Features extracted correctly and normalized

#### FR3: Data Preprocessing
**ID**: FR3
**Priority**: Medium
**Description**: System must preprocess raw data for ML models
**Operations**: Normalization, missing data handling, outlier removal
**Constraints**: <5ms preprocessing time
**Acceptance**: Preprocessed data ready for models

### 4.2 Spatial-Temporal Modeling

#### FR4: GNN Encoder
**ID**: FR4
**Priority**: High
**Description**: System must encode spatial dependencies using GNN
**Input**: Graph structure and node features
**Output**: Node embeddings representing spatial relationships
**Architecture**: GAT or GCN (2-3 layers)
**Constraints**: <50ms inference time
**Acceptance**: Embeddings capture spatial dependencies

#### FR5: Temporal Modeling
**ID**: FR5
**Priority**: High
**Description**: System must model temporal patterns
**Input**: Temporal sequences of node embeddings
**Output**: Temporal representations
**Architecture**: GRU or Transformer
**Constraints**: <30ms inference time
**Acceptance**: Temporal patterns captured

### 4.3 Anomaly Detection

#### FR6: Self-Supervised Training
**ID**: FR6
**Priority**: High
**Description**: System must train anomaly detector without labeled data
**Input**: Normal traffic data
**Output**: Trained anomaly detection model
**Method**: Dual-head reconstruction + forecasting
**Constraints**: Training on normal data only
**Acceptance**: Model learns normal patterns

#### FR7: Anomaly Scoring
**ID**: FR7
**Priority**: High
**Description**: System must compute anomaly scores
**Input**: Current and predicted traffic states
**Output**: Anomaly score (0-1)
**Method**: Combined reconstruction + forecasting errors
**Constraints**: <30ms scoring time
**Acceptance**: >80% F1-score, <5% false alarm rate

#### FR8: Threshold Selection
**ID**: FR8
**Priority**: Medium
**Description**: System must select anomaly detection threshold
**Method**: Quantile-based (98th percentile)
**Constraints**: Adaptive threshold selection
**Acceptance**: Optimal threshold selected

### 4.4 Traffic Control

#### FR9: RL Agent
**ID**: FR9
**Priority**: High
**Description**: System must control traffic signals using RL agent
**Input**: Node embeddings, anomaly scores
**Output**: Signal phase actions
**Algorithm**: DQN (Deep Q-Network)
**Constraints**: <20ms decision time
**Acceptance**: Signals controlled correctly

#### FR10: Reward Calculation
**ID**: FR10
**Priority**: High
**Description**: System must calculate rewards for RL agent
**Input**: Current state, action, next state, anomaly score
**Output**: Reward value
**Formula**: `R = -α₁·waiting - α₂·queue - α₃·anomaly_score`
**Constraints**: Multi-objective optimization
**Acceptance**: Rewards guide agent correctly

#### FR11: Signal Control Execution
**ID**: FR11
**Priority**: High
**Description**: System must execute signal phase changes
**Input**: Actions from RL agent
**Output**: Signal phase changes in SUMO/real system
**Interface**: TraCI API or hardware interface
**Constraints**: Real-time execution
**Acceptance**: Signals change correctly

### 4.5 Integration & Coordination

#### FR12: System Integration
**ID**: FR12
**Priority**: High
**Description**: System must integrate all components seamlessly
**Components**: Graph construction, GNN, anomaly detection, RL control
**Constraints**: <100ms total latency
**Acceptance**: End-to-end system functional

#### FR13: Network-Wide Coordination
**ID**: FR13
**Priority**: High
**Description**: System must coordinate signals across network
**Method**: Multi-intersection RL control
**Constraints**: Scalable to 100+ intersections
**Acceptance**: Coordinated control achieved

### 4.6 Monitoring & Visualization

#### FR14: Real-Time Dashboard
**ID**: FR14
**Priority**: Medium
**Description**: System must provide real-time monitoring dashboard
**Features**: Traffic state, anomaly alerts, performance metrics
**Constraints**: <1s update latency
**Acceptance**: Dashboard displays current state

#### FR15: Performance Metrics
**ID**: FR15
**Priority**: Medium
**Description**: System must compute and display performance metrics
**Metrics**: Waiting time, queue length, travel time, throughput
**Constraints**: Real-time computation
**Acceptance**: Metrics computed correctly

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

#### NFR1: Latency
**ID**: NFR1
**Priority**: High
**Description**: System must operate in real-time
**Requirement**: <100ms per decision cycle
**Breakdown**:
- Graph construction: <10ms
- GNN inference: <50ms
- Anomaly detection: <30ms
- RL decision: <20ms
**Acceptance**: Latency measured and verified

#### NFR2: Throughput
**ID**: NFR2
**Priority**: Medium
**Description**: System must handle high data rates
**Requirement**: 10+ decisions per second
**Acceptance**: Throughput measured and verified

#### NFR3: Scalability
**ID**: NFR3
**Priority**: High
**Description**: System must scale to large networks
**Requirement**: Support 100+ intersections
**Acceptance**: Tested on large networks

### 5.2 Quality Requirements

#### NFR4: Accuracy
**ID**: NFR4
**Priority**: High
**Description**: System must achieve target performance
**Requirements**:
- 15-25% improvement in waiting time
- >80% F1-score for anomaly detection
- <5% false alarm rate
**Acceptance**: Performance verified through evaluation

#### NFR5: Reliability
**ID**: NFR5
**Priority**: Medium
**Description**: System must be reliable
**Requirement**: 99.9% uptime (for production)
**Acceptance**: Reliability tested

#### NFR6: Fault Tolerance
**ID**: NFR6
**Priority**: Medium
**Description**: System must handle failures gracefully
**Requirement**: Graceful degradation on sensor failure
**Acceptance**: Fault tolerance tested

### 5.3 Usability Requirements

#### NFR7: User Interface
**ID**: NFR7
**Priority**: Low
**Description**: System must provide intuitive interface
**Requirement**: Dashboard with clear visualizations
**Acceptance**: User testing completed

#### NFR8: Documentation
**ID**: NFR8
**Priority**: Medium
**Description**: System must be well-documented
**Requirement**: Complete API documentation, user guides
**Acceptance**: Documentation reviewed

### 5.4 Maintainability Requirements

#### NFR9: Modularity
**ID**: NFR9
**Priority**: Medium
**Description**: System must be modular
**Requirement**: Components can be developed/tested independently
**Acceptance**: Modular architecture verified

#### NFR10: Code Quality
**ID**: NFR10
**Priority**: Medium
**Description**: Code must follow standards
**Requirement**: PEP 8, type hints, docstrings
**Acceptance**: Code review completed

---

## 6. System Requirements

### 6.1 Hardware Requirements

#### SR1: Training Hardware
**ID**: SR1
**Priority**: High
**Description**: GPU required for training
**Requirement**: NVIDIA RTX 3090/4090 or A100
**RAM**: 32GB+
**Storage**: 500GB+
**Acceptance**: Hardware available

#### SR2: Inference Hardware
**ID**: SR2
**Priority**: Medium
**Description**: Hardware for deployment
**Requirement**: CPU or GPU (edge device compatible)
**RAM**: 8GB+
**Storage**: 50GB+
**Acceptance**: Hardware tested

### 6.2 Software Requirements

#### SR3: Operating System
**ID**: SR3
**Priority**: High
**Description**: Supported operating systems
**Requirement**: Linux, Windows 10+, macOS 12+
**Acceptance**: OS compatibility verified

#### SR4: Python Environment
**ID**: SR4
**Priority**: High
**Description**: Python version and libraries
**Requirement**: Python 3.10+, PyTorch 2.0+, PyTorch Geometric 2.4+
**Acceptance**: Dependencies installed

#### SR5: SUMO
**ID**: SR5
**Priority**: High
**Description**: Traffic simulation software
**Requirement**: SUMO 1.19+, TraCI API
**Acceptance**: SUMO installed and configured

### 6.3 Data Requirements

#### SR6: Training Data
**ID**: SR6
**Priority**: High
**Description**: Data for training models
**Requirement**: SUMO network files or real traffic data
**Volume**: 100GB+ for comprehensive training
**Acceptance**: Data available

#### SR7: Evaluation Data
**ID**: SR7
**Priority**: High
**Description**: Data for evaluation
**Requirement**: Test datasets with ground truth
**Acceptance**: Evaluation data prepared

---

## 7. Constraints and Assumptions

### 7.1 Constraints

#### C1: Computational Constraints
- Limited GPU resources (training time constraints)
- Real-time inference requirements
- Memory limitations

#### C2: Data Constraints
- Limited labeled anomaly data
- Synthetic data from SUMO (may not match real-world)
- Data quality and availability

#### C3: Time Constraints
- 16-week project timeline
- Limited time for extensive evaluation
- Publication deadlines

#### C4: Budget Constraints
- Open-source software only
- Limited cloud computing budget
- Patent filing costs

### 7.2 Assumptions

#### A1: Data Availability
- SUMO network files available
- Real traffic data available (optional)
- Sufficient data for training

#### A2: Technical Assumptions
- SUMO accurately simulates traffic
- GNN and RL techniques applicable
- Integration feasible

#### A3: Deployment Assumptions
- Infrastructure supports real-time operation
- Sensors provide reliable data
- System can be deployed incrementally

---

## 8. Acceptance Criteria

### 8.1 Functional Acceptance

**AC1**: Graph construction correctly represents network
**AC2**: Features extracted accurately
**AC3**: GNN encoder produces meaningful embeddings
**AC4**: Anomaly detection achieves >80% F1-score
**AC5**: RL agent controls signals correctly
**AC6**: System integrates all components
**AC7**: Network-wide coordination works

### 8.2 Performance Acceptance

**AC8**: Latency <100ms per decision
**AC9**: 15-25% improvement in waiting time
**AC10**: Scalability to 100+ intersections
**AC11**: >80% F1-score for anomaly detection
**AC12**: <5% false alarm rate

### 8.3 Quality Acceptance

**AC13**: Code follows standards (PEP 8)
**AC14**: Documentation complete
**AC15**: Tests pass (>80% coverage)
**AC16**: System reliable (99.9% uptime)

---

## 9. Requirements Traceability

### 9.1 Requirements to Design

| Requirement | Design Component |
|-------------|-------------------|
| FR1-FR3 | Graph construction module |
| FR4-FR5 | GNN encoder module |
| FR6-FR8 | Anomaly detection module |
| FR9-FR11 | RL control module |
| FR12-FR13 | Integration module |
| FR14-FR15 | Dashboard module |

### 9.2 Requirements to Testing

| Requirement | Test Case |
|-------------|-----------|
| FR1 | Graph construction test |
| FR4 | GNN encoder test |
| FR7 | Anomaly detection test |
| FR9 | RL agent test |
| NFR1 | Latency test |
| NFR4 | Performance test |

---

## 10. Requirements Validation

### 10.1 Validation Methods

1. **Reviews**: Stakeholder reviews of requirements
2. **Prototyping**: Early prototypes to validate feasibility
3. **Testing**: Comprehensive testing against requirements
4. **Evaluation**: Performance evaluation against acceptance criteria

### 10.2 Validation Schedule

- **Week 2**: Requirements review
- **Week 5**: Prototype validation
- **Week 11**: Comprehensive testing
- **Week 15**: Final validation

---

**Document Version**: 1.0
**Last Updated**: [Current Date]
**Status**: Complete
