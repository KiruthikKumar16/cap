import argparse
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import deque
import random
import time
import os

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set global seed for reproducibility of the SEEDING logic itself (but variety between runs)
seed_val = int(time.time() % 10000)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

from src.phase1.traffic_env import SUMOTrafficEnv
from src.phase1.reward_calculator import RewardCalculator
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.baselines.nstlight import NSTLightAgent
from src.baselines.colight import CoLightAgent

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done, prev_state=None):
        self.buffer.append((state, action, reward, next_state, done, prev_state))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, prev_state = zip(*batch)
        
        state = np.stack(state)
        action = np.stack(action)
        reward = np.stack(reward)
        next_state = np.stack(next_state)
        done = np.stack(done)
        
        # Handle optional prev_state
        if prev_state[0] is not None:
            processed_prev = []
            sample_shape = state[0].shape
            for ps in prev_state:
                if ps is None:
                    processed_prev.append(np.zeros(sample_shape))
                else:
                    processed_prev.append(ps)
            prev_state = np.stack(processed_prev)
        else:
            prev_state = None
            
        return state, action, reward, next_state, done, prev_state
    def __len__(self):
        return len(self.buffer)

def get_eps(step, total_steps, init_eps=1.0, final_eps=0.05):
    fraction = min(1.0, float(step) / (total_steps * 0.5))
    return init_eps - fraction * (init_eps - final_eps)

def train_baseline_optimized(config, model_type, episodes=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    in_dim = 12 
    hidden_dim = 64
    if model_type == "nstlight":
        model = NSTLightAgent(in_dim, hidden_dim, 64, 2).to(device)
        target_model = NSTLightAgent(in_dim, hidden_dim, 64, 2).to(device)
    elif model_type == "colight":
        model = CoLightAgent(in_dim, hidden_dim, 64, 2).to(device)
        target_model = CoLightAgent(in_dim, hidden_dim, 64, 2).to(device)
    else:
        raise ValueError(f"Unknown baseline: {model_type}")

    target_model.load_state_dict(model.state_dict())
    
    # SOTA Tuning: Modern GNN-TSC agents (2024+) often use higher LRs for faster convergence
    adj_lr = 1e-3
    if model_type == "nstlight":
        adj_lr = 2e-3 
        
    optimizer = optim.Adam(model.parameters(), lr=adj_lr)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(50000)
    
    model_cfg = config["model"]
    gnn_dummy = PredictiveGNNRL(
        st_gnn_in_dim=model_cfg["feature_dim"], st_gnn_hidden_dim=model_cfg["hidden_dim"],
        st_gnn_heads=1, st_gnn_layers=1, st_gnn_dropout=0, 
        st_gnn_horizon=config.get("data", {}).get("window", {}).get("history", 3),
        rl_gnn_in_dim=model_cfg["feature_dim"], rl_gnn_hidden_dim=model_cfg["hidden_dim"],
        rl_gnn_embedding_dim=model_cfg["embedding_dim"], rl_gnn_layers=1, 
        rl_gnn_type="gat", rl_gnn_heads=1, rl_gnn_dropout=0
    ).to(device)
    
    reward_calc = RewardCalculator(
        waiting_time_weight=config["reward"]["waiting_time_weight"],
        queue_length_weight=config["reward"]["queue_length_weight"],
        normalize=True
    )
    
    env = SUMOTrafficEnv(
        net_file=config["sumo"]["net_file"],
        route_file=config["sumo"]["route_file"],
        model=gnn_dummy,
        reward_calculator=reward_calc,
        max_steps=config["sumo"]["simulation_steps"],
        enable_anomaly_awareness=False
    )
    
    max_steps = config["sumo"]["simulation_steps"]
    batch_size = 64
    gamma = 0.99
    global_step = 0
    total_steps = episodes * max_steps
    
    # We will use the actual edge_index from the environment instead of a mock identity graph.
    # This ensures the GNN can correctly aggregate neighboring intersection states.

    print(f"\n[FAST-TRACK] Training {model_type} on {device}...")
    print(f"Episodes: {episodes} | Batch Size: {batch_size} (OPTIMIZED)")
    
    start_time = time.time()
    
    try:
        for ep in range(episodes):
            # Introduce stochasticity per episode
            ep_seed = random.randint(0, 10000)
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            
            env.reset(seed=ep_seed)
            raw_obs = env._get_raw_observation()
            if torch.is_tensor(raw_obs): raw_obs = raw_obs.cpu().numpy()
            edge_index = env.edge_index.to(device)
            prev_raw_obs = None
            
            ep_reward = 0
            for step in range(max_steps):
                eps = get_eps(global_step, total_steps)
                
                # Epsilon Greedy
                if random.random() < eps:
                    actions = [env.action_space.sample() for _ in range(env.num_intersections)]
                else:
                    obs_t = torch.tensor(raw_obs, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        if model_type == "nstlight":
                            if prev_raw_obs is None: prev_raw_obs = np.zeros_like(raw_obs)
                            prev_t = torch.tensor(prev_raw_obs, dtype=torch.float32, device=device)
                            q_vals = model(obs_t, prev_t, edge_index)
                        else:
                            q_vals = model(obs_t, edge_index)
                # SOTA Differentiation: Inject 'Non-Stationary Rush Hour' Surge
                # This tests the model's ability to handle distribution shifts (NSTLight's specialty)
                if 1000 <= step <= 2000:
                    try:
                        import traci
                        if step % 5 == 0: # Inject every 5 steps to avoid overflow
                            # Find edges with demand and inject extra vehicles
                            edges = traci.edge.getIDList()
                            surge_edges = [e for e in edges if "to" in e or "from" in e][:3]
                            for i, edge in enumerate(surge_edges):
                                veh_id = f"surge_{step}_{i}"
                                try:
                                    # Create a route for the surge vehicle
                                    route_id = f"r_surge_{edge}"
                                    if route_id not in traci.route.getIDList():
                                        traci.route.add(route_id, [edge])
                                    traci.vehicle.add(veh_id, route_id)
                                    traci.vehicle.setSpeed(veh_id, 13.89) # 50 km/h
                                except: pass
                    except: pass

                _, reward, term, trunc, _ = env.step(np.array(actions))
                done = np.any(term) or np.any(trunc)
                next_raw_obs = env._get_raw_observation()
                if torch.is_tensor(next_raw_obs): next_raw_obs = next_raw_obs.cpu().numpy()
                
                # Batch transitions locally for buffer
                for i in range(env.num_intersections):
                    buffer.push(raw_obs[i], actions[i], reward[i], next_raw_obs[i], float(done), 
                                prev_state=prev_raw_obs[i] if prev_raw_obs is not None else None)
                    
                prev_raw_obs = raw_obs.copy()
                raw_obs = next_raw_obs
                ep_reward += np.mean(reward)
                global_step += 1
                
                # OPTIMIZED BATCH TRAINING
                if len(buffer) > batch_size and global_step % 4 == 0:
                    states, acts, rews, next_states, dones, prev_states = buffer.sample(batch_size)
                    s_t = torch.tensor(states, dtype=torch.float32, device=device)
                    a_t = torch.tensor(acts, dtype=torch.int64, device=device).unsqueeze(-1)
                    r_t = torch.tensor(rews, dtype=torch.float32, device=device).unsqueeze(-1)
                    n_s_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                    d_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
                    
                    # BATCHED GNN PASS (No more j-loop!)
                    # BATCHED GNN PASS (Using actual edge_index)
                    if model_type == "nstlight":
                        p_s_t = torch.tensor(prev_states, dtype=torch.float32, device=device) if prev_states is not None else torch.zeros_like(s_t)
                        q = model(s_t, p_s_t, edge_index)
                        with torch.no_grad():
                            q_next = target_model(n_s_t, s_t, edge_index)
                    else: # colight
                        q = model(s_t, edge_index)
                        with torch.no_grad():
                            q_next = target_model(n_s_t, edge_index)
                    
                    q_a = q.gather(1, a_t)
                    q_next_max = q_next.max(1, keepdim=True)[0]
                    target = r_t + gamma * (1 - d_t) * q_next_max
                    
                    loss = loss_fn(q_a, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                        
                if global_step % 1000 == 0:
                    target_model.load_state_dict(model.state_dict())
                    
                if done:
                    break
            
            elapsed = time.time() - start_time
            print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward/max_steps:.4f} | Eps: {eps:.2f} | Time: {elapsed/60:.1f}m")
            
            # The environment's reset() or close() handles logging to CSV. 
            # episode_count incrementing logic is moved inside SUMOTrafficEnv.
            pass
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        out_dir = Path("checkpoints_fast")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{model_type}_fast.pth"
        torch.save(model.state_dict(), out_path)
        print(f"\n[OK] Model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fast_validate.yaml")
    parser.add_argument("--model", type=str, choices=["nstlight", "colight"], required=True)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    train_baseline_optimized(config, args.model, args.episodes)
