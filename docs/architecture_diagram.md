# MAPPO-STGNN Architecture and System Diagram

This document contains a publication-standard system architecture diagram for the MAPPO-STGNN framework.

## System Architecture Diagram

```mermaid
graph TD
    classDef environment fill:#eaf4f4,stroke:#6b9080,stroke-width:2px;
    classDef representation fill:#f6fff8,stroke:#a4c3b2,stroke-width:2px;
    classDef policy fill:#ffebd6,stroke:#e5989b,stroke-width:2px;
    classDef critical fill:#fcd5ce,stroke:#b5838d,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph Environment["Simulator Environment (SUMO)"]
        A1["Micro-Traffic Variables<br>(Krauss & LC2013 Models)"] --> A2["Road Network & Intersections"]
        A2 --> A3["Sensor Telemetry<br>(Inductive Loops, Cameras)"]
        AE["Adversarial Injector<br>(Accidents, Sensor Noise)"] -.-> A2
    end
    
    subgraph STGNN["Spatio-Temporal Graph Neural Network (ST-GNN)"]
        B1["Local Observation (o_i)<br>Queue, Delay, Phase, Flow"] --> B2["Feature Extractor"]
        
        B2 --> B3["Spatial Embeddings (GCN)<br>Neighborhood Aggregation via Adjacency Matrix (A)"]
        B3 --> B4["Temporal Encodings (GRU)<br>Extracting Traffic Congestion Dynamics"]
        B4 --> B5["Spatio-Temporal State Vector (h_t)"]
    end
    
    subgraph MAPPO["MAPPO Core (CTDE Framework)"]
        C1["Decentralized Actor Network<br>Policy (π)"]
        C2["Centralized Critic Network<br>Value Function (V)"]
        C3["Global State (s)<br>Concatenated Observations"]
    end
    
    A3 -- "Extracts Raw States" --> B1
    
    B5 --> C1
    
    C3 --> C2
    C2 -- "Advantage Estimation" --> C4["PPO Objective<br>(Entropy Bonus + Clip Threshold)"]
    C1 -- "Action Generation" --> C4
    
    C1 -- "Action (a_i)<br>(Change / Keep Phase)" --> A2
    
    B1 -. "Constructs" .-> C3
    
    class Environment environment;
    class STGNN representation;
    class MAPPO policy;
    class AE critical;
```

### Flowchart Breakdown

1.  **Environment (Simulation Layer):** The physical traffic kinematics execute inside SUMO. The road network generates raw sensor data detailing local intersection congestion (delay, queues). Additionally, the module highlights the injection of non-stationary anomalous events (sensor noise, network accidents) to test model robustness.
2.  **ST-GNN (State Representation Layer):**
    *   *Spatial Construction:* Evaluates intersection vectors using Graph Convolutional Networks (GCNs) mapping influences using a defined physical network Adjacency Matrix.
    *   *Temporal Dynamics:* Embeds spatial outputs sequentially through Gated Recurrent Units (GRUs), generating a high-density, anticipatory state vector reflecting approaching traffic waves.
3.  **MAPPO (Control Layer):** Structured under the **C**entralized **T**raining with **D**ecentralized **E**xecution paradigm.
    *   During active inference, only the decentral Actor responds (utilizing the rich Spatial-Temporal embeddings).
    *   During training algorithms, the Centralized Critic processes the composite global map, solving Markov violations and maintaining global stability across multiple independent actors.
