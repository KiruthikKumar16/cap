"""
SUMO Traffic Environment Wrapper

Gym-compatible environment wrapper for SUMO traffic simulation.
Integrates with TraCI API for real-time traffic control.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import gymnasium.utils.seeding as seeding

# Suppress TraCI deprecation UserWarning (getAllProgramLogics) when we call getCompleteRedYellowGreenDefinition
warnings.filterwarnings("ignore", message=".*getAllProgramLogics.*", category=UserWarning)

import traci  # SUMO/TraCI is mandatory - no fallback
TRACI_AVAILABLE = True

from src.phase1.graph_builder import TrafficGraphBuilder
from src.phase1.feature_extractor import TrafficFeatureExtractor
from src.phase1.gnn_encoder import TrafficGNNEncoder
from src.phase1.reward_calculator import RewardCalculator
from src.phase3.integration import get_anomaly_controller


class SUMOTrafficEnv(gym.Env):
    """
    Gym-compatible environment for SUMO traffic simulation.
    
    This environment wraps SUMO simulation and provides:
    - Graph-structured observations via GNN encoder
    - Multi-discrete action space (one action per intersection)
    - Reward based on traffic metrics
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        config_file: Optional[str] = None,
        step_length: float = 1.0,
        max_steps: int = 3600,
        gnn_encoder: Optional[TrafficGNNEncoder] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        use_gui: bool = False,
        traci_port: Optional[int] = None,
        sumo_binary: Optional[str] = None,
        time_penalty_per_step: float = 0.0,
        enable_anomaly_awareness: bool = False,
    ):
        """
        Initialize SUMO traffic environment.
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            config_file: Optional path to SUMO config file (.sumocfg)
            step_length: Simulation step length in seconds
            max_steps: Maximum simulation steps per episode
            gnn_encoder: GNN encoder for observations (optional, will create if None)
            reward_calculator: Reward calculator (optional, will create if None)
            use_gui: Whether to use SUMO GUI
            traci_port: Port for TraCI (default 8813). Use different ports for train vs eval envs.
            sumo_binary: Full path to sumo/sumo-gui executable. If not set, uses PATH or SUMO_HOME/bin.
            time_penalty_per_step: Small per-step cost (standard RL) so baseline reward is non-zero when traffic metrics are 0.
            enable_anomaly_awareness: Whether to use Phase 2 anomaly detection for reward shaping.
        """
        super().__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.config_file = config_file
        self.step_length = step_length
        self.max_steps = max_steps
        self.use_gui = use_gui
        self.traci_port = traci_port if traci_port is not None else 8813
        self.sumo_binary = sumo_binary
        self.time_penalty_per_step = float(time_penalty_per_step)
        self.enable_anomaly_awareness = enable_anomaly_awareness
        
        # Initialize components
        self.graph_builder = TrafficGraphBuilder(net_file)
        self.intersections = self.graph_builder.intersections
        self.num_intersections = len(self.intersections)
        
        self.feature_extractor = TrafficFeatureExtractor(self.intersections)
        
        # GNN encoder (create if not provided)
        if gnn_encoder is None:
            self.gnn_encoder = TrafficGNNEncoder(
                in_dim=12,  # Feature dimension
                hidden_dim=64,
                out_dim=32,
                num_layers=2,
                gnn_type="gat",
                gat_heads=2,
                dropout=0.1
            )
        else:
            self.gnn_encoder = gnn_encoder
        
        # Reward calculator (create if not provided)
        if reward_calculator is None:
            self.reward_calculator = RewardCalculator(
                waiting_time_weight=0.1,
                queue_length_weight=0.05,
                anomaly_weight=0.0,
                normalize=True
            )
        else:
            self.reward_calculator = reward_calculator
        
        # Get edge index
        self.edge_index = self.graph_builder.get_edge_index()
        
        # Observation space: flattened GNN embeddings
        embedding_dim = self.gnn_encoder.out_dim
        obs_dim = self.num_intersections * embedding_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: MultiDiscrete (one action per intersection)
        # Each intersection can choose from 4 phases
        num_phases = 4
        self.action_space = spaces.MultiDiscrete([num_phases] * self.num_intersections)
        
        # State
        self.current_step = 0
        self.sumo_running = False
        self.np_random = None  # Will be initialized on first reset
        self._last_reward = 0.0
        self._max_phase_per_tl: Optional[Dict[str, int]] = None  # cached at reset
        self._tl_ids_for_exec: Optional[List[str]] = None  # SUMO TLS IDs at reset (A0,B0,...)
        self._veh_depart_times: Dict[str, float] = {}
        self._queue_length_step = 0.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Optional reset options
            
        Returns:
            Observation and info dict
        """
        # Set seed if provided
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)
        
        # Close existing SUMO connection if any
        if self.sumo_running:
            self._close_sumo()
        
        # Start SUMO simulation
        self._start_sumo()
        
        # Sync with SUMO TLS IDs for phase execution (handles graph placeholder J0 vs net A0)
        if self.sumo_running and TRACI_AVAILABLE:
            try:
                self._tl_ids_for_exec = list(traci.trafficlight.getIDList())
                self._max_phase_per_tl = {
                    tl_id: self._get_max_phase_index(tl_id) for tl_id in (self._tl_ids_for_exec or self.intersections)
                }
            except Exception:
                self._tl_ids_for_exec = None
                self._max_phase_per_tl = None
        else:
            self._max_phase_per_tl = None
        
        # Reset step counter and placeholder info
        self.current_step = 0
        self._last_reward = 0.0
        self._travel_time_step = 0.0
        self._waiting_time_step = 0.0
        self._queue_length_step = 0.0
        self._veh_depart_times = {}

        # Reset anomaly controller if enabled
        if self.enable_anomaly_awareness:
            anomaly_controller = get_anomaly_controller()
            if anomaly_controller is not None:
                anomaly_controller.reset()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment using SUMO simulation.
        
        Args:
            action: Action array [num_intersections] with phase selections
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute actions (set signal phases)
        self._execute_actions(action)
        
        # Advance simulation
        self._advance_simulation()
        
        # Calculate reward (real SUMO metrics) + optional per-step time penalty (standard RL)
        reward = self._calculate_reward() - self.time_penalty_per_step
        self._last_reward = reward
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        info = self._get_info()
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def _resolve_sumo_binary(self) -> str:
        """Resolve path to sumo/sumo-gui. Prefer sumo_binary, then SUMO_HOME/bin, then PATH."""
        name = "sumo-gui" if self.use_gui else "sumo"
        if self.sumo_binary:
            return self.sumo_binary
        import os
        sumo_home = os.environ.get("SUMO_HOME", "").strip()
        if sumo_home:
            candidate = Path(sumo_home) / "bin" / (name + (".exe" if os.name == "nt" else ""))
            if candidate.exists():
                return str(candidate)
        # Common Linux install (Google Colab / Ubuntu)
        if os.name != "nt":
            for prefix in ["/usr/share/sumo", "/usr/bin"]:
                candidate = Path(prefix) / "bin" / name if prefix == "/usr/share/sumo" else Path(prefix) / name
                if candidate.exists():
                    if "SUMO_HOME" not in os.environ:
                        os.environ["SUMO_HOME"] = prefix if prefix == "/usr/share/sumo" else "/usr/share/sumo"
                    return str(candidate)
        return name  # rely on PATH
    
    def _start_sumo(self) -> None:
        """Start SUMO simulation. SUMO is mandatory (no placeholder fallback)."""
        sumo_bin = self._resolve_sumo_binary()
        sumo_cmd = [sumo_bin]
        
        if self.config_file:
            sumo_cmd.extend(["-c", self.config_file])
        else:
            sumo_cmd.extend(["-n", self.net_file, "-r", self.route_file])
        
        sumo_cmd.extend(["--step-length", str(self.step_length)])
        sumo_cmd.append("--no-warnings")
        traci.start(sumo_cmd, port=self.traci_port)
        self.sumo_running = True
    
    def _close_sumo(self) -> None:
        """Close SUMO simulation."""
        if TRACI_AVAILABLE and self.sumo_running:
            try:
                traci.close()
            except Exception:
                pass
        self.sumo_running = False
    
    def _execute_actions(self, actions: np.ndarray) -> None:
        """Execute actions (set signal phases). Use SUMO TLS IDs when available (A0,B0,...)."""
        if not self.sumo_running:
            return
        use_ids = self._tl_ids_for_exec if self._tl_ids_for_exec is not None else self.intersections
        # Handle scalar actions (from VecEnv during evaluation callback)
        if not hasattr(actions, '__len__') or np.ndim(actions) == 0:
            return
        if len(use_ids) != len(actions):
            return
        try:
            for i, tl_id in enumerate(use_ids):
                phase = int(actions[i])
                max_phase = 3
                if self._max_phase_per_tl and tl_id in self._max_phase_per_tl:
                    max_phase = self._max_phase_per_tl[tl_id]
                else:
                    max_phase = self._get_max_phase_index(tl_id)
                phase = max(0, min(phase, max_phase))
                traci.trafficlight.setPhase(tl_id, phase)
        except Exception as e:
            self.sumo_running = False
            if not getattr(self, "_sumo_connection_warned", False):
                self._sumo_connection_warned = True
                print(f"Warning: SUMO connection lost ({e}). Continuing in placeholder mode.")
            try:
                traci.close()
            except Exception:
                pass
    
    def _get_queue_length_step(self) -> float:
        """Total halting vehicles on controlled lanes this step (real SUMO only). 0 if SUMO not running."""
        if not self.sumo_running or not TRACI_AVAILABLE:
            return 0.0
        total = 0.0
        try:
            tl_ids = traci.trafficlight.getIDList()
            use_ids = tl_ids if tl_ids else self.intersections
            for intersection_id in use_ids:
                for lane_id in traci.trafficlight.getControlledLanes(intersection_id):
                    total += traci.lane.getLastStepHaltingNumber(lane_id)
        except Exception:
            pass
        return total

    def _get_waiting_time_step(self) -> float:
        """Total waiting time (s) on controlled lanes + vehicle-based; real SUMO only."""
        if not self.sumo_running or not TRACI_AVAILABLE:
            return 0.0
        total = 0.0
        try:
            tl_ids = traci.trafficlight.getIDList()
            use_ids = tl_ids if tl_ids else self.intersections
            for intersection_id in use_ids:
                for lane_id in traci.trafficlight.getControlledLanes(intersection_id):
                    total += traci.lane.getWaitingTime(lane_id)
            # Vehicle-based waiting time (real SUMO) when lane-based is 0
            if total == 0:
                try:
                    for veh_id in traci.vehicle.getIDList():
                        try:
                            total += traci.vehicle.getWaitingTime(veh_id)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def _advance_simulation(self) -> None:
        """Advance SUMO simulation by one step. Track travel time via depart/arrive events."""
        self._travel_time_step = 0.0
        self._waiting_time_step = 0.0
        self._queue_length_step = 0.0
        if self.sumo_running and TRACI_AVAILABLE:
            try:
                traci.simulationStep()
                try:
                    sim_time = traci.simulation.getTime()
                except Exception:
                    sim_time = None
                # Track departures so we can compute travel time at arrival
                try:
                    for veh_id in traci.simulation.getDepartedIDList():
                        if sim_time is not None:
                            self._veh_depart_times[veh_id] = sim_time
                except Exception:
                    pass
                # Sum travel time for vehicles that arrived this step
                try:
                    for veh_id in traci.simulation.getArrivedIDList():
                        depart_time = self._veh_depart_times.pop(veh_id, None)
                        if depart_time is not None and sim_time is not None:
                            self._travel_time_step += max(0.0, sim_time - depart_time)
                except Exception:
                    pass
                self._waiting_time_step = self._get_waiting_time_step()
                self._queue_length_step = self._get_queue_length_step()
            except Exception as e:
                self.sumo_running = False
                if not getattr(self, "_sumo_connection_warned", False):
                    self._sumo_connection_warned = True
                    print(f"Warning: SUMO connection lost ({e}). Continuing in placeholder mode.")
                try:
                    traci.close()
                except Exception:
                    pass
    
    def _get_max_phase_index(self, tl_id: str) -> int:
        """Return max valid phase index for this TLS (0-based). Falls back to 3 if SUMO not running."""
        if not self.sumo_running or not TRACI_AVAILABLE:
            return 3
        try:
            # TraCI: returns list of (duration, state) per phase (module-level filter suppresses deprecation)
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
            if program and len(program) > 0:
                return max(0, len(program) - 1)
        except Exception:
            pass
        return 3
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (GNN embeddings).
        
        Returns:
            Flattened GNN embeddings
        """
        # Extract features
        features = self.feature_extractor.extract()  # [num_nodes, feature_dim]
        
        # Encode with GNN
        self.gnn_encoder.eval()
        with torch.no_grad():
            embeddings = self.gnn_encoder(features, self.edge_index)  # [num_nodes, embedding_dim]
        
        # Flatten
        observation = embeddings.numpy().flatten().astype(np.float32)
        
        return observation
    
    def _calculate_reward(self) -> float:
        """Calculate reward from current traffic state."""
        # Get anomaly scores if anomaly awareness is enabled
        anomaly_scores = None
        if self.enable_anomaly_awareness:
            anomaly_controller = get_anomaly_controller()
            if anomaly_controller is not None:
                # Get current features for anomaly detection
                current_features = self.feature_extractor.extract()
                anomaly_scores = anomaly_controller.get_anomaly_scores(
                    current_features.numpy() if hasattr(current_features, 'numpy') else current_features,
                    self.edge_index
                )

        if self.sumo_running:
            reward = self.reward_calculator.calculate_from_sumo(self.intersections, anomaly_scores)
        else:
            # Placeholder reward
            reward = self.reward_calculator._calculate_placeholder(self.intersections, anomaly_scores)

        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode is terminated."""
        if not self.sumo_running:
            return False
        
        try:
            # Episode ends when no more vehicles expected
            return traci.simulation.getMinExpectedNumber() == 0
        except Exception:
            return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary. In placeholder mode, throughput/travel_time/waiting_time are 0 (not reported as results)."""
        info = {
            "step": self.current_step,
            "sumo_running": self.sumo_running,
            "num_intersections": self.num_intersections,
            "travel_time": 0.0,
            "waiting_time": 0.0,
            "queue_length": 0.0,
            "departed": 0,
            "placeholder_mode": not self.sumo_running,
        }
        if self.sumo_running and TRACI_AVAILABLE:
            try:
                info["simulation_time"] = traci.simulation.getTime()
                info["num_vehicles"] = traci.simulation.getMinExpectedNumber()
                info["departed"] = traci.simulation.getDepartedNumber()
                info["arrived"] = traci.simulation.getArrivedNumber()
                info["travel_time"] = getattr(self, "_travel_time_step", 0.0)
                info["waiting_time"] = getattr(self, "_waiting_time_step", 0.0)
                info["queue_length"] = getattr(self, "_queue_length_step", 0.0)
            except Exception:
                pass
        return info
    
    def close(self) -> None:
        """Close the environment."""
        self._close_sumo()
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            None (SUMO GUI handles rendering)
        """
        # SUMO GUI handles rendering automatically
        return None


