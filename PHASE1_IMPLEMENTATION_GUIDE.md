# Phase 1 Implementation Guide

## Overview
This guide provides step-by-step instructions for implementing Phase 1: Traffic Prediction & Adaptive Control using GNN + RL, based on the Smartcities_final.pdf approach.

## Step 1: Graph Construction Module

### File: `src/phase1/graph_builder.py`

**Purpose**: Build graph representation of traffic network from SUMO

**Key Functions**:
```python
def build_traffic_graph(sumo_net_file: str) -> Tuple[torch.Tensor, Dict]:
    """
    Extract intersections and road segments from SUMO network.
    
    Returns:
        edge_index: [2, E] tensor of edge connections
        node_info: Dict mapping node_id to intersection data
    """
    pass

def extract_node_features(traci, intersections: List[str]) -> torch.Tensor:
    """
    Extract real-time features for each intersection node.
    
    Features:
        - Signal phase (one-hot, 4 phases)
        - Phase duration
        - Queue lengths (sum, max)
        - Waiting time (normalized)
        - Vehicle counts per lane
    """
    pass
```

**Implementation Notes**:
- Use `sumolib` to parse SUMO network XML
- Identify signalized intersections (traffic lights)
- Create directed edges between connected intersections
- Map SUMO junction IDs to graph node indices

## Step 2: Feature Extraction

### File: `src/phase1/feature_extractor.py`

**Purpose**: Extract and normalize traffic features from SUMO via TraCI

**Key Functions**:
```python
class TrafficFeatureExtractor:
    def __init__(self, intersections: List[str]):
        self.intersections = intersections
    
    def extract(self, traci) -> torch.Tensor:
        """
        Extract features for all intersections.
        Returns: [N, F] tensor where N=num_intersections, F=feature_dim
        """
        pass
    
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to [0, 1] range"""
        pass
```

**Feature Dimensions**:
- Signal phase: 4 (one-hot)
- Phase duration: 1 (normalized)
- Queue sum: 1 (normalized)
- Queue max: 1 (normalized)
- Waiting time: 1 (normalized)
- Vehicle counts: 4 (one per direction, normalized)

**Total**: ~12 features per node

## Step 3: GNN Encoder

### File: `src/phase1/gnn_encoder.py`

**Purpose**: Spatial modeling using Graph Convolutional Networks

**Architecture**:
```python
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class TrafficGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(GCNConv(in_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(GCNConv(hidden_dim, out_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, F] node features
            edge_index: [2, E] edge connections
        Returns:
            embeddings: [N, out_dim] node embeddings
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x
```

**Alternative**: Use GAT (Graph Attention Network) for attention-based aggregation

## Step 4: SUMO Environment Wrapper

### File: `src/phase1/traffic_env.py`

**Purpose**: Gym-compatible environment for SUMO + TraCI

**Key Components**:
```python
import gymnasium as gym
from gymnasium import spaces
import traci

class SUMOTrafficEnv(gym.Env):
    def __init__(self, net_file: str, route_file: str, config_file: str):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.config_file = config_file
        
        # Initialize SUMO
        self._init_sumo()
        
        # Action space: MultiDiscrete for each intersection
        num_intersections = len(self.intersections)
        num_phases = 4  # Typically 4 phases per intersection
        self.action_space = spaces.MultiDiscrete([num_phases] * num_intersections)
        
        # Observation space: GNN embeddings
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_intersections, embedding_dim),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        """Reset SUMO simulation"""
        traci.close()
        self._init_sumo()
        return self._get_observation(), {}
    
    def step(self, actions):
        """
        Execute actions (set signal phases) and advance simulation.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Set phases for each intersection
        for i, intersection_id in enumerate(self.intersections):
            phase = int(actions[i])
            traci.trafficlight.setPhase(intersection_id, phase)
        
        # Advance simulation
        traci.simulationStep()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = traci.simulation.getMinExpectedNumber() == 0
        
        return self._get_observation(), reward, done, False, {}
    
    def _calculate_reward(self) -> float:
        """Multi-objective reward function"""
        # Get network-wide metrics
        total_waiting_time = 0.0
        total_queue_length = 0.0
        
        for intersection_id in self.intersections:
            lanes = traci.trafficlight.getControlledLanes(intersection_id)
            for lane in lanes:
                waiting_time = traci.lane.getWaitingTime(lane)
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                total_waiting_time += waiting_time
                total_queue_length += queue_length
        
        # Negative reward (minimize waiting and queues)
        reward = -0.1 * total_waiting_time - 0.05 * total_queue_length
        return reward
```

## Step 5: DQN Agent Setup

### File: `src/phase1/dqn_agent.py`

**Purpose**: Configure DQN agent using Stable Baselines3

**Implementation**:
```python
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

def create_dqn_agent(env, gnn_encoder):
    """
    Create DQN agent with GNN-based feature extraction.
    
    Note: SB3 DQN expects flat observations, so we need to flatten
    GNN embeddings or use a custom policy network.
    """
    
    # Option 1: Flatten GNN embeddings
    class FlattenGNNWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return obs.flatten()
    
    wrapped_env = FlattenGNNWrapper(env)
    
    # Create DQN agent
    model = DQN(
        "MlpPolicy",
        wrapped_env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,  # Hard update
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
    )
    
    return model

# Option 2: Custom policy network that accepts GNN embeddings
# This requires modifying SB3's policy network architecture
```

**Custom Policy Network** (if needed):
```python
from stable_baselines3.common.policies import BasePolicy
import torch.nn as nn

class GNNDQNPolicy(BasePolicy):
    def __init__(self, *args, gnn_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_encoder = gnn_encoder
        # Q-network that takes flattened GNN embeddings
        self.q_net = nn.Sequential(
            nn.Linear(gnn_encoder.out_dim * num_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
```

## Step 6: Training Loop

### File: `src/phase1/train_rl.py`

**Purpose**: Main training script

**Implementation**:
```python
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Build graph
    edge_index, intersections = build_traffic_graph(cfg["sumo"]["net_file"])
    
    # Create GNN encoder
    gnn_encoder = TrafficGNNEncoder(
        in_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        out_dim=cfg["model"]["embedding_dim"],
        num_layers=cfg["model"]["gnn_layers"]
    )
    
    # Create environment
    env = SUMOTrafficEnv(
        net_file=cfg["sumo"]["net_file"],
        route_file=cfg["sumo"]["route_file"],
        config_file=cfg["sumo"]["config_file"]
    )
    
    # Wrap environment to use GNN
    env = GNNObservationWrapper(env, gnn_encoder, edge_index)
    
    # Create DQN agent
    model = create_dqn_agent(env, gnn_encoder)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=cfg["output"]["checkpoint_dir"],
        name_prefix="dqn_traffic"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=cfg["output"]["best_model_dir"],
        log_path=cfg["output"]["log_dir"],
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        log_interval=10
    )
    
    # Save final model
    model.save(cfg["output"]["final_model_path"])

if __name__ == "__main__":
    main()
```

## Step 7: Configuration File

### File: `configs/phase1.yaml`

```yaml
sumo:
  net_file: data/raw/network.net.xml
  route_file: data/raw/routes.rou.xml
  config_file: data/raw/sumo_config.sumocfg
  step_length: 1.0
  simulation_steps: 3600  # 1 hour

model:
  feature_dim: 12
  hidden_dim: 64
  embedding_dim: 32
  gnn_layers: 2
  gnn_type: gcn  # or gat

training:
  total_timesteps: 1000000
  learning_rate: 1e-3
  buffer_size: 50000
  batch_size: 32
  gamma: 0.99
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.05
  exploration_fraction: 0.1

reward:
  waiting_time_weight: 0.1
  queue_length_weight: 0.05
  fuel_consumption_weight: 0.0  # optional
  speed_weight: 0.0  # optional

output:
  checkpoint_dir: outputs/phase1/checkpoints
  best_model_dir: outputs/phase1/best_models
  log_dir: outputs/phase1/logs
  final_model_path: outputs/phase1/dqn_traffic_final.zip
```

## Step 8: Evaluation Script

### File: `src/phase1/evaluate.py`

**Purpose**: Evaluate trained model against baselines

**Metrics to Compare**:
- Average waiting time
- Average queue length
- Travel time
- Throughput (vehicles/hour)
- Fuel consumption (if available)

**Baselines**:
- Fixed-time controller (SUMO default)
- Random controller
- Other RL methods (if available)

## Testing Checklist

- [ ] Graph construction correctly identifies intersections
- [ ] Feature extraction produces valid feature vectors
- [ ] GNN encoder produces embeddings of correct shape
- [ ] Environment wrapper integrates with SUMO correctly
- [ ] Actions (phase changes) are applied correctly
- [ ] Reward function calculates correctly
- [ ] DQN agent trains without errors
- [ ] Model converges (rewards increase over time)
- [ ] Evaluation shows improvement over baseline

## Common Issues & Solutions

1. **SUMO not found**: Install SUMO and add to PATH
2. **TraCI connection errors**: Ensure SUMO is started before TraCI
3. **Action space mismatch**: Verify number of phases per intersection
4. **Observation shape errors**: Check GNN output dimensions match DQN input
5. **Slow training**: Reduce simulation steps or use faster hardware

## Next Steps

After Phase 1 is complete:
1. Integrate with Phase 2 (anomaly detection) for proactive control
2. Expand to larger networks (4×4, 6×6 grids)
3. Add more sophisticated reward functions
4. Implement multi-agent RL for better coordination
