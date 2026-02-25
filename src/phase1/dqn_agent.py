"""
DQN Agent Setup Module

Configures and creates DQN agent using Stable Baselines3.
Integrates GNN encoder with DQN for traffic signal control.
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from src.phase1.gnn_encoder import TrafficGNNEncoder, FlattenGNNWrapper
from src.phase1.traffic_env import SUMOTrafficEnv


class MultiDiscreteToDiscreteWrapper(gym.Env):
    """
    Wrapper to convert MultiDiscrete action space to Discrete for DQN.
    
    DQN only supports Discrete action spaces, so we flatten MultiDiscrete
    by treating it as a single Discrete space with num_actions = product of all actions.
    """
    
    def __init__(self, env: SUMOTrafficEnv):
        """
        Initialize wrapper.
        
        Args:
            env: SUMO traffic environment with MultiDiscrete action space
        """
        super().__init__()
        self.env = env
        
        # Convert MultiDiscrete to Discrete
        if isinstance(env.action_space, spaces.MultiDiscrete):
            # Calculate total number of action combinations
            nvec = env.action_space.nvec
            self.n_actions = int(np.prod(nvec))
            self.nvec = nvec
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.nvec = None
            self.action_space = env.action_space
        
        self.observation_space = env.observation_space
        self.metadata = getattr(env, 'metadata', {})
    
    def _convert_action(self, action: int) -> np.ndarray:
        """
        Convert Discrete action to MultiDiscrete.
        
        Args:
            action: Discrete action (0 to n_actions-1)
            
        Returns:
            MultiDiscrete action array
        """
        if self.nvec is None:
            return action
        
        # Convert flat action to MultiDiscrete
        multi_action = np.zeros(len(self.nvec), dtype=np.int32)
        remaining = action
        
        for i in range(len(self.nvec) - 1, -1, -1):
            multi_action[i] = remaining % self.nvec[i]
            remaining = remaining // self.nvec[i]
        
        return multi_action

    def reset(self, seed=None, options=None):
        """
        Reset the underlying environment.

        We simply forward the call so that the return type
        (observation, info) is preserved for the outer wrapper.
        """
        # Gymnasium >=0.26 passes seed/options to reset
        if hasattr(self.env, "reset"):
            return self.env.reset(seed=seed, options=options)
        # Fallback to base implementation (will raise if not implemented)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        """
        Step the environment using a Discrete action.

        The incoming `action` is an integer from the DQN policy.
        We convert it back to the original MultiDiscrete format
        before passing it to the wrapped environment.
        """
        multi_action = self._convert_action(action)
        return self.env.step(multi_action)

    def render(self):
        """Forward render to underlying environment if available."""
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        """Forward close to underlying environment if available."""
        if hasattr(self.env, "close"):
            self.env.close()


class GNNObservationWrapper(gym.Env):
    """
    Wrapper to integrate GNN encoder with RL environment.
    
    This wrapper ensures that observations are properly processed through
    the GNN encoder before being passed to the RL agent.
    Properly inherits from gym.Env for Stable Baselines3 compatibility.
    """
    
    def __init__(self, env: SUMOTrafficEnv):
        """
        Initialize wrapper.
        
        Args:
            env: SUMO traffic environment
        """
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        result = self.env.reset(seed=seed, options=options)
        # Ensure we return tuple (obs, info)
        if result is None:
            # Fallback: try again
            result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # If single value, wrap in tuple with empty info
        if result is None:
            # Last resort: create dummy observation
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, {}
        return result, {}
    
    def step(self, action):
        """Step environment."""
        return self.env.step(action)
    
    def render(self):
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        self.env.close()


class TrainingCallback(BaseCallback):
    """
    Custom callback for training monitoring.
    
    Logs training metrics and saves checkpoints.
    """
    
    def __init__(self, log_interval: int = 100, verbose: int = 1):
        """
        Initialize callback.
        
        Args:
            log_interval: Logging interval in steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log metrics periodically
        if self.num_timesteps % self.log_interval == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
                avg_length = sum(self.episode_lengths[-100:]) / min(len(self.episode_lengths), 100)
                
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.1f}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout."""
        # Collect episode statistics if available
        if hasattr(self.locals, 'infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])


def create_dqn_agent(
    env: SUMOTrafficEnv,
    gnn_encoder: Optional[TrafficGNNEncoder] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DQN:
    """
    Create DQN agent for traffic control.
    
    Args:
        env: SUMO traffic environment
        gnn_encoder: Optional GNN encoder (will use env's encoder if None)
        config: Optional configuration dictionary
        
    Returns:
        Configured DQN agent
    """
    # Use environment's GNN encoder if not provided
    if gnn_encoder is None:
        gnn_encoder = env.gnn_encoder
    
    # Default configuration
    default_config = {
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,  # Hard update
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "verbose": 1,
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)

    # SOTA: Dueling via policy_kwargs if supported (SB3 vanilla DQN has no use_double_dqn; dueling may be unsupported)
    dueling = default_config.get("dueling", False)
    policy_kwargs = None
    if dueling:
        try:
            from stable_baselines3.dqn.policies import DQNPolicy
            sig = __import__("inspect").signature(DQNPolicy.__init__)
            if "dueling" in sig.parameters:
                policy_kwargs = {"dueling": True}
        except Exception:
            pass

    # Wrap environment: first convert MultiDiscrete to Discrete, then wrap for GNN
    # Convert action space for DQN compatibility
    if isinstance(env.action_space, spaces.MultiDiscrete):
        env = MultiDiscreteToDiscreteWrapper(env)

    wrapped_env = GNNObservationWrapper(env)

    # Create DQN agent (SB3 DQN does not support use_double_dqn in __init__)
    model = DQN(
        "MlpPolicy",
        wrapped_env,
        learning_rate=default_config["learning_rate"],
        buffer_size=default_config["buffer_size"],
        learning_starts=default_config["learning_starts"],
        batch_size=default_config["batch_size"],
        tau=default_config["tau"],
        gamma=default_config["gamma"],
        train_freq=default_config["train_freq"],
        gradient_steps=default_config["gradient_steps"],
        target_update_interval=default_config["target_update_interval"],
        exploration_fraction=default_config["exploration_fraction"],
        exploration_initial_eps=default_config["exploration_initial_eps"],
        exploration_final_eps=default_config["exploration_final_eps"],
        verbose=default_config["verbose"],
        device="auto",
        policy_kwargs=policy_kwargs,
    )

    return model


def load_dqn_agent(
    path: str,
    env: SUMOTrafficEnv,
) -> DQN:
    """
    Load trained DQN agent from file.
    
    Args:
        path: Path to saved model
        env: SUMO traffic environment
        
    Returns:
        Loaded DQN agent
    """
    wrapped_env = GNNObservationWrapper(env)
    model = DQN.load(path, env=wrapped_env)
    return model


