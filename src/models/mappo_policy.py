
import torch
import torch.nn as nn
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
        
        # Determine dimensions from observation space
        obs_dim = self.observation_space.shape[0]
        self.embedding_dim = obs_dim // 6
        self.local_dim = self.embedding_dim * 5
        self.global_dim = self.embedding_dim
        
        # Override the policy and value heads
        # pi network: local_dim -> pi latent -> action_net
        self.pi_features_extractor = nn.Sequential(
            nn.Linear(self.local_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        ).to(self.device)
        
        # vf network: global_dim -> vf latent -> value_net
        self.vf_features_extractor = nn.Sequential(
            nn.Linear(self.global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        ).to(self.device)
        
        # Final layers
        self.action_net = nn.Linear(128, self.action_space.n).to(self.device)
        self.value_net = nn.Linear(128, 1).to(self.device)

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
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Critic
        latent_vf = self.vf_features_extractor(global_obs)
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        local_obs = obs[:, :self.local_dim]
        global_obs = obs[:, self.local_dim:]
        
        latent_pi = self.pi_features_extractor(local_obs)
        latent_vf = self.vf_features_extractor(global_obs)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        local_obs = obs[:, :self.local_dim]
        latent_pi = self.pi_features_extractor(local_obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        global_obs = obs[:, self.local_dim:]
        latent_vf = self.vf_features_extractor(global_obs)
        return self.value_net(latent_vf)
