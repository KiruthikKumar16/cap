# MAPPO-STGNN Framework Diagrams

This document contains two distinct diagrams structured for publication, both configured with explicit white backgrounds to ensure clean rendering in research papers or presentations.

## 1. Algorithmic Architecture Diagram
This diagram focuses on the logical structure and mathematical flow of the deep learning components (ST-GNN + MAPPO).

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'background': '#ffffff'}}}%%
graph TD
    classDef bg fill:#ffffff,stroke-width:0px;
    classDef input fill:#e1f5fe,stroke:#29b6f6,stroke-width:2px,color:#000;
    classDef gnn fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px,color:#000;
    classDef rnn fill:#f3e5f5,stroke:#ab47bc,stroke-width:2px,color:#000;
    classDef mappo fill:#fff3e0,stroke:#ffa726,stroke-width:2px,color:#000;
    classDef output fill:#ffebee,stroke:#ef5350,stroke-width:2px,color:#000;
    
    subgraph STGNN [ST-GNN: Space-Time Encoding]
        A[Local Observables <br/> Queue, Delay, Phase]:::input --> B[Feature Normalization]:::input
        B --> C[Spatial Gating GCN]:::gnn
        AdjacencyMatrix[Network Adjacency Matrix A]:::input --> C
        C -->|Spatial Embedding H| D[Temporal Correlation GRU]:::rnn
        D -->|Recurrent Update z_t, r_t| E[Spatio-Temporal State vector h_t]:::rnn
    end

    subgraph MAPPO [MAPPO: CTDE Pipeline]
        E --> F[Decentralized Actor Network <br/> Policy π]:::mappo
        GlobalState[Global State Vector S]:::input --> G[Centralized Critic Network <br/> Value V]:::mappo
        
        G -->|Advantage A_t| H[PPO Loss & Clip Objective]:::mappo
        F -->|Action Choice a_i| H
    end

    H -->|Backpropagation| F
    H -->|Backpropagation| G
    
    F -->|Inference Output| Action[Signal Phase Action <br> Keep / Change]:::output

    style STGNN fill:#ffffff,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    style MAPPO fill:#ffffff,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
```

---

## 2. System Execution & Deployment Diagram
This diagram illustrates how the software operational loop executes within the real-world or simulated environment, focusing on component interactions via the API.

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'background': '#ffffff'}}}%%
graph LR
    classDef sim fill:#eceff1,stroke:#78909c,stroke-width:2px,color:#000;
    classDef api fill:#fff8e1,stroke:#ffca28,stroke-width:2px,color:#000;
    classDef exec fill:#e0f7fa,stroke:#26c6da,stroke-width:2px,color:#000;
    classDef train fill:#fdf4fc,stroke:#ec407a,stroke-width:2px,color:#000;

    subgraph Env [Environment Module]
        Sumo[SUMO Engine <br> Traffic Kinematics]:::sim
        Sensors[Virtual Sensors & Induction Loops]:::sim
        Sumo --> Sensors
    end

    subgraph API [API Gateway]
        TraCI[TraCI Python Interface]:::api
    end

    subgraph Engine [Inference & Actuation Module]
        Extractor[Observation Extractor]:::exec
        Deploy[Edge / Decentralized Actors]:::exec
        Controller[Signal Phase Controller]:::exec
    end

    subgraph TrainModule [Offline Training Module]
        Buffer[Replay Buffer <br> State, Action, Reward]:::train
        CTDE[Centralized Critic & MAPPO Optimizer]:::train
    end

    Sensors -->|Raw Vehicle Data| TraCI
    TraCI -->|Parsed Telemetry| Extractor
    Extractor -->|Localized State o_i| Deploy
    Deploy -->|Discrete Action a_i| Controller
    Controller -->|Phase Shifts| TraCI
    TraCI -->|Actuation| Sumo
    
    Extractor -.->|Global State & Rewards| Buffer
    Deploy -.->|Action Probs| Buffer
    Buffer -.->|Batched Data| CTDE
    CTDE -.->|Grad Updates| Deploy

    style Env fill:#ffffff,stroke:#333,stroke-width:1px;
    style API fill:#ffffff,stroke:#333,stroke-width:1px;
    style Engine fill:#ffffff,stroke:#333,stroke-width:1px;
    style TrainModule fill:#ffffff,stroke:#333,stroke-width:1px;
```

### Notes for Usage
* **White Backgrounds:** The Mermaid configuration block `%%{init: {'theme': 'default', 'themeVariables': { 'background': '#ffffff'}}}%%` explicitly forces these diagrams to render with a pure white background and dark text for contrast.
* The **Algorithmic Architecture** is optimized for sections detailing the mathematics and neural network structure of the multi-agent system.
* The **System Execution** diagram is optimized for methodology sections explaining how the software interacts with the simulator and how the simulation step loop operates.
