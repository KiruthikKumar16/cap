# Project Report: Robust Multi-Agent Traffic Control Under Non-Stationarity With Anomaly-Aware Proactive Adaptation

**Core contribution statement:** Robust multi-agent traffic control under non-stationarity with anomaly-aware proactive adaptation.

**Status:** Technical Implementation Phase (Ongoing Training & Evaluation)

---

## 1. Introduction

### 1.1 Problem Statement
Urban traffic congestion represents a critical challenge for modern smart cities, leading to significant economic losses, increased environmental emissions, and reduced quality of life. Traditional traffic signal control (TSC) systems often rely on fixed-time or simple actuated logic, which fail to adapt to the highly dynamic and stochastic nature of urban traffic flow.

### 1.2 Motivation for Multi-Agent Systems
Centralized control of a city-wide traffic network is computationally intractable due to the curse of dimensionality. Multi-Agent Reinforcement Learning (MARL) offers a scalable alternative by distributing control across independent agents (intersections). However, decentralized agents must coordinate to prevent "selfish" optimizations that could shift congestion to neighboring areas.

### 1.3 Scope of Project
This project focuses on the development and evaluation of a MARL system on a large-scale **10x10 synthetic grid**, comprising **100 independent intersections**. The system integrates Spatio-Temporal Graph Neural Networks (ST-GNN) to handle spatial dependencies and coordinate decisions across the network.

---

## 2. Literature Review

### 2.1 Traditional Traffic Control
Classical approaches include Fixed-Time control and Actuated control (e.g., SCATS, SCOOT). While reliable, these methods are non-adaptive and require manual re-timing to handle changing traffic patterns.

### 2.2 Reinforcement Learning Approaches
Early RL research focused on single-agent TSC or small grids (e.g., 3x3). Methods like Q-Learning and standard PPO have shown promise but often struggle with the non-stationarity inherent in multi-agent environments.

### 2.3 Graph-Based Coordination
Recent advancements (e.g., CoLight, PressLight) have introduced Graph Neural Networks (GNNs) to TSC. These models allow agents to share hidden states with neighbors, enabling spatial awareness. However, many existing studies are limited to smaller grids or do not incorporate predictive/risk-aware elements.

---

## 3. System Architecture

### 3.1 High-Level Pipeline
The system follows a closed-loop control cycle:
1.  **Environment:** SUMO (Simulation of Urban MObility) provides the raw traffic state.
2.  **Graph Model:** An ST-GNN processes the state history to generate spatial-temporal embeddings.
3.  **Policy:** A MAPPO-style actor-critic framework chooses the optimal signal phase.
4.  **Action:** The selected phase is executed via TraCI.
5.  **Feedback:** The environment returns a reward signal based on performance metrics.

### 3.2 Implemented Components
*   **100-Agent Setup:** Each intersection is an independent agent within a unified vectorized environment.
*   **Graph Attention Network (GAT):** Used to prioritize information from adjacent "clogged" intersections.
*   **MAPPO Framework:** A centralized critic architecture designed to stabilize multi-agent training.
*   *Note: Diagrams illustrating the GAT attention mechanism and the centralized critic architecture are recommended for the final presentation.*

---

## 4. Methodology

### 4.1 State Representation
Each agent observes a 12-dimensional node feature vector, including:
*   One-hot encoded signal phase.
*   Normalized queue lengths and waiting times.
*   Directional vehicle counts.

### 4.2 Action Space
The agents utilize a **Multi-Discrete** action space, where each agent selects from 4 available traffic signal phases (e.g., North-South Green, East-West Green).

### 4.3 Reward Design
The reward function is designed to be smooth and continuous, utilizing a **sigmoid-based density factor**. It conceptually balances:
*   **Speed Bonus:** Rewarding higher average vehicle speeds.
*   **Congestion Penalty:** Penalizing waiting times and queue lengths, with weights that increase as density grows.

### 4.4 Coordination & Risk-Awareness
Coordination is achieved through the GAT layers, which allow hidden state sharing. The methodology also incorporates a forecasting head to predict future traffic states, conceptually allowing for risk-aware decision-making.

---

## 5. Experimental Setup

### 5.1 Grid Configuration
*   **Topology:** 10x10 Manhattan-style grid.
*   **Intersections:** 100 signalized junctions.
*   **Traffic Density:** High-density stochastic flow generated via SUMO's `randomTrips.py`.

### 5.2 Simulation Parameters
*   **Step Size:** 0.5 seconds.
*   **Episode Length:** 10,000 steps.
*   **Total Timesteps:** 5,000,000 (Target).

### 5.3 Training Configuration
*   **Algorithm:** PPO (Proximal Policy Optimization).
*   **Hyperparameters:** Learning rate (1e-4), Batch size (128), GNN hidden dimension (256).
*   **Hardware:** Training is accelerated via NVIDIA CUDA (RTX 2050 4GB).

---

## 6. Training Process (Work-in-Progress)

Training is currently conducted using a **Curriculum Learning** strategy (3x3 → 5x5 → 10x10) to improve convergence speed. 

**Status:** Training is ongoing. We are monitoring metrics such as:
*   Cumulative Episode Reward.
*   Critic Value Loss.
*   Explained Variance (a key indicator of model stability).

**Important Note:** Convergence has not yet been reached, and the final policy is still being refined through iterative runs.

---

## 7. Evaluation Plan

### 7.1 Performance Metrics
The system will be evaluated based on:
1.  **Average Waiting Time:** Mean delay per vehicle.
2.  **Queue Length:** Average number of halting vehicles.
3.  **Throughput:** Total number of vehicles that completed their trips.

### 7.2 Comparison Strategy
Results will be benchmarked against:
*   **Fixed-Time Baseline:** Standard non-adaptive control.
*   **Max-Pressure Baseline:** A proven optimal queue-balancing heuristic.
*   **Decentralized PPO:** To verify the benefits of the centralized critic.

---

## 8. Current Observations

Preliminary observations from early training stages indicate:
*   The agent successfully learns basic phase-switching logic in simpler grids (3x3).
*   Learning stability is higher when using the centralized critic compared to purely decentralized actors.
*   The reward signal is successfully trending upwards, though significant fluctuations remain.
*   *Note: These observations are preliminary and subject to change upon final convergence.*

---

## 9. Limitations

1.  **Training Instability:** Multi-agent environments are inherently prone to policy oscillations.
2.  **Computational Cost:** Simulating 100 agents with high-fidelity GNNs is resource-intensive.
3.  **VRAM Constraints:** The 4GB VRAM limit necessitates optimized batch sizes and model depth.
4.  **Incomplete Convergence:** Due to the scale of the 10x10 grid, reaching a global optimum requires extensive training time.

---

## 10. Future Work

*   **Final MAPPO Refinement:** Tuning the centralized critic's observation space.
*   **Ablation Studies:** Quantifying the impact of the GAT vs. standard GCN.
*   **Real-World Transfer:** Moving from synthetic grids to OpenStreetMap (OSM) data for urban networks like Bengaluru.
*   **Communication Protocols:** Implementing explicit agent-to-agent communication.

---

## 11. Conclusion

This project has successfully implemented a high-capacity MARL framework for large-scale traffic control. By combining MAPPO with a predictive GNN, we have built a system capable of coordinating 100 intersections. While training is still in progress, the architectural foundation is complete and ready for final benchmarking against traditional Baseline methods.

---
*End of Report*
