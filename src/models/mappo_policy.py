
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, List, Optional, Tuple, Type, Union

class MAPPOPolicy(ActorCriticPolicy):
    """
    Custom MAPPO-style policy where:
    - Actor (Policy Network) uses local node features.
    - Critic (Value Network) uses the global graph embedding.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # We need to set net_arch before calling super().__init__
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[128, 128], vf=[128, 128])
            
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Determine dimensions from observation space (Expected 6 embeddings: 1 self, 4 neighbors, 1 global)
        obs_dim = self.observation_space.shape[0]
        self.embedding_dim = obs_dim // 6
        self.local_dim = self.embedding_dim * 5
        self.global_dim = obs_dim - self.local_dim # Ensure sum is exactly obs_dim
        
        # Support for Hierarchical Regional Critics
        self.use_regional_critics = kwargs.get("use_regional_critics", True)
        self.num_regions = 4 # Hardcoded for 25 nodes (4 regions of ~6 nodes)
        
        # Override the policy and value heads
        # pi network: local_dim -> pi latent -> action_net
        self.pi_features_extractor = nn.Sequential(
            nn.Linear(self.local_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        ).to(self.device)
        
        if self.use_regional_critics:
            # Create a list of vf feature extractors and value nets for each region
            # We approximate the regional state dimension as global_dim // num_regions
            self.regional_dim = max(1, self.global_dim // self.num_regions)
            
            self.vf_features_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.regional_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                ) for _ in range(self.num_regions)
            ]).to(self.device)
            
            self.value_nets = nn.ModuleList([
                nn.Linear(128, 1) for _ in range(self.num_regions)
            ]).to(self.device)
        else:
            # vf network: global_dim -> vf latent -> value_net
            self.vf_features_extractor = nn.Sequential(
                nn.Linear(self.global_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            ).to(self.device)
            self.value_net = nn.Linear(128, 1).to(self.device)
            
        # Final layers
        if isinstance(self.action_space, spaces.Tuple):
            self.is_parameterized_action = True
            self.action_net = nn.Linear(128, self.action_space[0].n).to(self.device)
            # Duration modifier head (mean, log_std)
            self.duration_mean_net = nn.Linear(128, 1).to(self.device)
            self.duration_log_std = nn.Parameter(torch.zeros(1)).to(self.device)
        else:
            self.is_parameterized_action = False
            self.action_net = nn.Linear(128, self.action_space.n).to(self.device)

        # RE-REGISTER OPTIMIZER
        # Because we created these new layers after calling super().__init__,
        # PyTorch failed to add them to the SB3 optimizer. We MUST update it here.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with separate Actor/Critic processing."""
        local_obs = obs[:, :self.local_dim]
        global_obs = obs[:, self.local_dim:]
        
        # Actor
        latent_pi = self.pi_features_extractor(local_obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if self.is_parameterized_action:
            # We must override get_actions to sample both
            # For this prototype, we'll manually sample here if deterministic
            phase_logits = self.action_net(latent_pi)
            dur_mean = self.duration_mean_net(latent_pi)
            if deterministic:
                actions_phase = torch.argmax(phase_logits, dim=-1)
                actions_dur = dur_mean
            else:
                actions_phase = torch.distributions.Categorical(logits=phase_logits).sample()
                actions_dur = torch.distributions.Normal(dur_mean, self.duration_log_std.exp()).sample()
            
            # Combine into a single tensor for SB3 compatibility (horrible hack but works for proof of concept)
            actions = torch.stack([actions_phase.float(), actions_dur.squeeze(-1)], dim=-1)
            
            # log prob approximation
            phase_log_prob = torch.distributions.Categorical(logits=phase_logits).log_prob(actions_phase)
            dur_log_prob = torch.distributions.Normal(dur_mean, self.duration_log_std.exp()).log_prob(actions_dur).squeeze(-1)
            log_prob = phase_log_prob + dur_log_prob
        else:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
        
        # Critic
        if self.use_regional_critics:
            # Split the global observation into regions
            # In a real setup, this would be based on an adjacency matrix or node IDs
            batch_size = global_obs.shape[0]
            values = torch.zeros(batch_size, 1, device=self.device)
            # Naive heuristic: divide the batch into chunks and assign to regions
            for i in range(self.num_regions):
                start_idx = i * self.regional_dim
                end_idx = min(start_idx + self.regional_dim, global_obs.shape[1])
                # Pad if necessary
                reg_obs = global_obs[:, start_idx:end_idx]
                if reg_obs.shape[1] < self.regional_dim:
                    padding = torch.zeros(batch_size, self.regional_dim - reg_obs.shape[1], device=self.device)
                    reg_obs = torch.cat([reg_obs, padding], dim=1)
                
                latent_vf = self.vf_features_extractors[i](reg_obs)
                # Average the values or sum them, we average here
                values += self.value_nets[i](latent_vf) / self.num_regions
        else:
            latent_vf = self.vf_features_extractor(global_obs)
            values = self.value_net(latent_vf)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        local_obs = obs[:, :self.local_dim]
        global_obs = obs[:, self.local_dim:]
        
        latent_pi = self.pi_features_extractor(local_obs)
        
        if self.use_regional_critics:
            batch_size = global_obs.shape[0]
            values = torch.zeros(batch_size, 1, device=self.device)
            for i in range(self.num_regions):
                start_idx = i * self.regional_dim
                end_idx = min(start_idx + self.regional_dim, global_obs.shape[1])
                reg_obs = global_obs[:, start_idx:end_idx]
                if reg_obs.shape[1] < self.regional_dim:
                    padding = torch.zeros(batch_size, self.regional_dim - reg_obs.shape[1], device=self.device)
                    reg_obs = torch.cat([reg_obs, padding], dim=1)
                
                latent_vf = self.vf_features_extractors[i](reg_obs)
                values += self.value_nets[i](latent_vf) / self.num_regions
        else:
            latent_vf = self.vf_features_extractor(global_obs)
            values = self.value_net(latent_vf)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        local_obs = obs[:, :self.local_dim]
        latent_pi = self.pi_features_extractor(local_obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        global_obs = obs[:, self.local_dim:]
        if self.use_regional_critics:
            batch_size = global_obs.shape[0]
            values = torch.zeros(batch_size, 1, device=self.device)
            for i in range(self.num_regions):
                start_idx = i * self.regional_dim
                end_idx = min(start_idx + self.regional_dim, global_obs.shape[1])
                reg_obs = global_obs[:, start_idx:end_idx]
                if reg_obs.shape[1] < self.regional_dim:
                    padding = torch.zeros(batch_size, self.regional_dim - reg_obs.shape[1], device=self.device)
                    reg_obs = torch.cat([reg_obs, padding], dim=1)
                
                latent_vf = self.vf_features_extractors[i](reg_obs)
                values += self.value_nets[i](latent_vf) / self.num_regions
            return values
        else:
            latent_vf = self.vf_features_extractor(global_obs)
            return self.value_net(latent_vf)
