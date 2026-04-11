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

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.phase1.traffic_env import SUMOTrafficEnv
from src.phase1.reward_calculator import RewardCalculator
# Dummy predictive model for environment init
from src.models.predictive_gnn_rl import PredictiveGNNRL
from src.baselines.nstlight import NSTLightAgent
from src.baselines.colight import CoLightAgent

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

def get_eps(step, total_steps, init_eps=1.0, final_eps=0.05):
    fraction = min(1.0, float(step) / (total_steps * 0.5))
    return init_eps - fraction * (init_eps - final_eps)

def train_baseline(config, model_type, episodes=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    in_dim = 12 # Feature extractor output dim
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(50000)
    
    # Dummy PredictiveGNNRL needed to initialize SUMOTrafficEnv without errors
    model_cfg = config["model"]
    gnn_dummy = PredictiveGNNRL(
        st_gnn_in_dim=model_cfg["feature_dim"], st_gnn_hidden_dim=model_cfg["hidden_dim"],
        st_gnn_layers=1, st_gnn_dropout=0, st_gnn_horizon=1,
        rl_gnn_in_dim=model_cfg["feature_dim"], rl_gnn_hidden_dim=model_cfg["hidden_dim"],
        rl_gnn_embedding_dim=model_cfg["embedding_dim"], rl_gnn_layers=1, rl_gnn_dropout=0
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
    
    print(f"Starting generic DQN training for {model_type} on {device}...")
    
    try:
        for ep in range(episodes):
            env.reset()
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
                        actions = torch.argmax(q_vals, dim=1).cpu().numpy()

                _, reward, term, trunc, _ = env.step(np.array(actions))
                done = np.any(term) or np.any(trunc)
                next_raw_obs = env._get_raw_observation()
                if torch.is_tensor(next_raw_obs): next_raw_obs = next_raw_obs.cpu().numpy()
                
                # Batch transitions locally for buffer
                for i in range(env.num_intersections):
                    buffer.push(raw_obs[i], actions[i], reward[i], next_raw_obs[i], float(done))
                    
                prev_raw_obs = raw_obs.copy()
                raw_obs = next_raw_obs
                ep_reward += np.mean(reward)
                global_step += 1
                
                # Train
                if len(buffer) > batch_size and global_step % 4 == 0:
                    states, acts, rews, next_states, dones = buffer.sample(batch_size)
                    s_t = torch.tensor(states, dtype=torch.float32, device=device)
                    a_t = torch.tensor(acts, dtype=torch.int64, device=device).unsqueeze(-1)
                    r_t = torch.tensor(rews, dtype=torch.float32, device=device).unsqueeze(-1)
                    n_s_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                    d_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
                    
                    # For a random sample we evaluate purely locally in the MLPs, ignoring edges
                    # Real CoLight processes fully structured batches, but this is a simplified fallback
                    q = model.action_head(s_t) if hasattr(model, 'action_head') else model.q_head(s_t)
                    q_a = q.gather(1, a_t)
                    with torch.no_grad():
                        q_next = target_model.action_head(n_s_t) if hasattr(target_model, 'action_head') else target_model.q_head(n_s_t)
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
            
            print(f"Episode {ep+1}/{episodes} | Avg Step Reward: {ep_reward/max_steps:.4f} | Eps: {eps:.2f}")
            
    finally:
        env.close()
        out_dir = Path("checkpoints")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{model_type}.pth"
        torch.save(model.state_dict(), out_path)
        print(f"Model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase2_10x10.yaml")
    parser.add_argument("--model", type=str, choices=["nstlight", "colight"], required=True)
    parser.add_argument("--episodes", type=int, default=150)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    train_baseline(config, args.model, args.episodes)
