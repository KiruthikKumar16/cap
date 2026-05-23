"""
SUMO Traffic Environment Wrapper

Gym-compatible environment wrapper for SUMO traffic simulation.
Integrates with TraCI API for real-time traffic control.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import socket
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
from src.models.predictive_gnn_rl import PredictiveGNNRL
from collections import deque
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
        model: PredictiveGNNRL,
        config_file: Optional[str] = None,
        step_length: float = 1.0,
        max_steps: int = 3600,
        st_gnn_horizon: int = 5,
        reward_calculator: Optional[RewardCalculator] = None,
        use_gui: bool = False,
        traci_port: Optional[int] = None,
        sumo_binary: Optional[str] = None,
        time_penalty_per_step: float = 0.0,
        enable_anomaly_awareness: bool = False,
        config: Optional[Dict] = None,
    ):
        """
        Initialize SUMO traffic environment.
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            model: The PredictiveGNNRL model for observation generation.
            config_file: Optional path to SUMO config file (.sumocfg)
            step_length: Simulation step length in seconds
            max_steps: Maximum simulation steps per episode
            st_gnn_horizon: Number of historical steps for the ST-GNN.
            reward_calculator: Reward calculator (optional, will create if None)
            use_gui: Whether to use SUMO GUI
            traci_port: Port for TraCI (default 8813). Use different ports for train vs eval envs.
            sumo_binary: Full path to sumo/sumo-gui executable. If not set, uses PATH or SUMO_HOME/bin.
            time_penalty_per_step: Small per-step cost (standard RL) so baseline reward is non-zero when traffic metrics are 0.
            enable_anomaly_awareness: Whether to use Phase 2 anomaly detection for reward shaping.
            config: Full global dictionary configuration mapping.
        """
        super().__init__()
        
        self.net_file_path = Path(net_file)
        if self.net_file_path.is_dir():
            self.is_procedural = True
            self.procedural_maps = list(self.net_file_path.glob("*.net.xml"))
            if not self.procedural_maps:
                raise ValueError(f"No .net.xml files in {self.net_file_path}")
            # Pick first for initialization
            self.net_file = str(self.procedural_maps[0])
            prefix = self.procedural_maps[0].name.replace('.net.xml', '')
            rou_files = list(self.net_file_path.glob(f"{prefix}*.rou.xml"))
            self.route_file = str(rou_files[0]) if rou_files else route_file
        else:
            self.is_procedural = False
            self.net_file = net_file
            self.route_file = route_file
            
        self.config_file = config_file
        self.step_length = step_length
        self.max_steps = max_steps
        self.use_gui = use_gui
        self.traci_port = traci_port if traci_port is not None else self._find_free_port()
        self.sumo_binary = sumo_binary
        self.time_penalty_per_step = float(time_penalty_per_step)
        self.enable_anomaly_awareness = enable_anomaly_awareness
        self.config = config if config is not None else {}
        
        # Initialize components
        self._init_graph()
        
        # Predictive GNN model and state history
        self.model = model
        self.state_history = deque(maxlen=st_gnn_horizon)

        
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
            
        # Each agent (intersection) observation is a concatenation of:
        #   [self_embedding] + [neighbor_embeddings (max_neighbors)] + [global_embedding]
        # Total length = (1 (self) + max_neighbors + 1 (global)) * embedding_dim.
        self.max_neighbors = 4
        embedding_dim = None
        if hasattr(self.model, "controller"):
            embedding_dim = getattr(self.model.controller, "out_dim", None)
            if embedding_dim is None and hasattr(self.model.controller, "get_output_dim"):
                embedding_dim = self.model.controller.get_output_dim()
        if embedding_dim is None:
            raise ValueError("Could not infer embedding_dim from model.controller")

        # 1 self + 4 neighbors + 1 global = 6 embeddings total
        obs_vector_dim = int((2 + self.max_neighbors) * int(embedding_dim))
        
        # Hybrid Parameterized Action Space: (Phase, Duration Modifier)
        self.is_parameterized_action = self.config.get("use_parameterized_actions", False)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_vector_dim,),
            dtype=np.float32,
        )
        
        if self.is_parameterized_action:
            # Tuple: (Discrete Phase, Box Duration Delta [-5s, +5s])
            self.action_space = spaces.Tuple((
                spaces.Discrete(4),
                spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
            ))
        else:
            # Assuming 4 phases per intersection (standard for our grid)
            self.action_space = spaces.Discrete(4)
        
        # Internal multi-agent tracking
        self.num_agents = self.num_intersections
        
        # State
        self.current_step = 0
        self.sumo_running = False
        self.np_random = None  # Will be initialized on first reset
        self._last_reward = 0.0
        self._max_phase_per_tl: Optional[Dict[str, int]] = None  # cached at reset
        self._tl_ids_for_exec: Optional[List[str]] = None  # SUMO TLS IDs at reset (A0,B0,...)
        self._veh_depart_times: Dict[str, float] = {}
        self._queue_length_step = 0.0
        self._last_action_info: Dict[str, Any] = {
            "requested_len": 0,
            "traffic_light_count": 0,
            "applied_count": 0,
            "skipped_reason": "not_started",
            "applied_phases": [],
        }
        
        # Episode-level metrics for SOTA evaluation
        self.episode_metrics = {
            "episode_total_waiting_time": 0.0,
            "episode_total_queue_length": 0.0,
            "episode_total_travel_time": 0.0,
            "episode_arrived_vehicles": 0,
            "episode_stopped_vehicles": 0,
            "episode_steps": 0,
        }
        self.log_file = "episode_metrics.csv"
        self.episode_count = 0
        self._init_log_file()

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _init_log_file(self):
        if not Path(self.log_file).exists():
            with open(self.log_file, "w") as f:
                f.write("episode,avg_waiting_time,avg_queue_length,throughput,avg_stopped_vehicles\n")

    def _log_episode(self, episode_idx: int):
        total_steps = max(1, self.episode_metrics["episode_steps"])
        avg_wait = self.episode_metrics["episode_total_waiting_time"] / total_steps
        avg_queue = self.episode_metrics["episode_total_queue_length"] / total_steps
        throughput = self.episode_metrics["episode_arrived_vehicles"]
        avg_stopped = self.episode_metrics["episode_stopped_vehicles"] / total_steps
        
        # Log to CSV
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{episode_idx},{avg_wait:.2f},{avg_queue:.2f},{throughput},{avg_stopped:.2f}\n")
        except Exception as e:
            print(f"Warning: Could not log to {self.log_file}: {e}")
        
        # Print for visibility
        print(f"\n[Episode {episode_idx} Metrics]")
        print(f"  Avg Wait: {avg_wait:.2f}s | Avg Queue: {avg_queue:.2f} | Throughput: {throughput} | Avg Stopped: {avg_stopped:.2f}")
        
    def _init_graph(self):
        """Initialize graph builder and extractors for current map."""
        self.graph_builder = TrafficGraphBuilder(self.net_file)
        self.intersections = self.graph_builder.intersections
        self.num_intersections = len(self.intersections)
        self.feature_extractor = TrafficFeatureExtractor(self.intersections)
        self.edge_index = self.graph_builder.get_edge_index()
        
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
            
        if self.is_procedural and self.np_random:
            # Pick a random procedural map
            map_path = self.np_random.choice(self.procedural_maps)
            self.net_file = str(map_path)
            prefix = map_path.name.replace('.net.xml', '')
            rou_files = list(self.net_file_path.glob(f"{prefix}*.rou.xml"))
            if rou_files:
                self.route_file = str(rou_files[0])
            self._init_graph()
        
        # Close existing SUMO connection if any
        if self.sumo_running:
            self._close_sumo()
        
        # Start SUMO simulation with the provided seed for multi-episode variance
        self._start_sumo(seed)
        
        # Log previous episode metrics if any steps were taken
        if self.episode_metrics["episode_steps"] > 0:
            self.episode_count += 1
            self._log_episode(self.episode_count)
        
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
        self._last_action_info = {
            "requested_len": 0,
            "traffic_light_count": len(self._tl_ids_for_exec or []),
            "applied_count": 0,
            "skipped_reason": "reset",
            "applied_phases": [],
        }

        # Reset episode metrics
        self.episode_metrics = {
            "episode_total_waiting_time": 0.0,
            "episode_total_queue_length": 0.0,
            "episode_total_travel_time": 0.0,
            "episode_arrived_vehicles": 0,
            "episode_stopped_vehicles": 0,
            "episode_steps": 0,
        }

        # Reset anomaly controller if enabled
        if self.enable_anomaly_awareness:
            anomaly_controller = get_anomaly_controller()
            if anomaly_controller is not None:
                anomaly_controller.reset()

        # Get initial observation
        self.state_history.clear()
        # Get initial observation
        self.state_history.append(self._get_raw_observation())
        observation = self._get_observation()
        base_info = self._get_info()
        # Vectorized info
        info = [base_info.copy() for _ in range(self.num_agents)]
        
        return observation, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute one step in the environment using SUMO simulation.
        
        Args:
            actions: Action array [num_agents] with phase selections
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute actions (set signal phases)
        self._execute_actions(actions)
        
        # Advance simulation
        self._advance_simulation()
        
        # Log progress occasionally for visibility
        if self.current_step % 100 == 0 and self.current_step > 0:
            try:
                running = traci.vehicle.getIDCount()
                print(f"  Step #{self.current_step}.00 (vehicles ACT {running})")
            except Exception:
                pass
        
        # Calculate global reward
        global_reward = self._calculate_reward() - self.time_penalty_per_step
        self._last_reward = global_reward
        
        # Get termination flags
        terminated_bool = self._is_terminated()
        truncated_bool = self.current_step >= self.max_steps
        
        # Get observation
        self.state_history.append(self._get_raw_observation())
        observation = self._get_observation()
        
        # Prepare vectorized outputs
        reward = np.full(self.num_agents, global_reward, dtype=np.float32)
        terminated = np.full(self.num_agents, terminated_bool, dtype=bool)
        truncated = np.full(self.num_agents, truncated_bool, dtype=bool)
        
        # Base info
        base_info = self._get_info()
        # Vectorized info
        info = [base_info.copy() for _ in range(self.num_agents)]
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def _resolve_sumo_binary(self) -> str:
        """Resolve path to sumo/sumo-gui. Prefer sumo_binary, then SUMO_HOME/bin, then PATH."""
        name = "sumo-gui" if self.use_gui else "sumo"
        if self.sumo_binary:
            configured = Path(self.sumo_binary)
            if self.use_gui and configured.name.lower() in {"sumo", "sumo.exe"}:
                gui_binary = configured.with_name("sumo-gui.exe" if configured.suffix.lower() == ".exe" else "sumo-gui")
                if gui_binary.exists():
                    return str(gui_binary)
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
    
    def _start_sumo(self, seed: Optional[int] = None) -> None:
        """Start SUMO simulation with specific seed for reproducibility/variance."""
        sumo_bin = self._resolve_sumo_binary()
        sumo_cmd = [sumo_bin]
        
        if self.config_file:
            sumo_cmd.extend(["-c", self.config_file])
        else:
            sumo_cmd.extend(["-n", self.net_file, "-r", self.route_file])
        
        sumo_cmd.extend(["--step-length", str(self.step_length)])
        sumo_cmd.append("--no-warnings")
        
        # SOTA: Explicit seeding for research-grade variance
        if seed is not None:
            sumo_cmd.extend(["--seed", str(seed)])
            
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
        self._last_action_info = {
            "requested_len": 0,
            "traffic_light_count": 0,
            "applied_count": 0,
            "skipped_reason": None,
            "applied_phases": [],
        }
        if not self.sumo_running:
            self._last_action_info["skipped_reason"] = "sumo_not_running"
            return
        use_ids = self._tl_ids_for_exec if self._tl_ids_for_exec is not None else self.intersections
        self._last_action_info["traffic_light_count"] = len(use_ids)
        
        if self.is_parameterized_action:
            # actions is a tuple: (discrete_phases, continuous_durations)
            phase_arr = np.asarray(actions[0]).reshape(-1)
            duration_arr = np.asarray(actions[1]).reshape(-1)
            self._last_action_info["requested_len"] = int(phase_arr.size)
        else:
            action_arr = np.asarray(actions)
            if action_arr.ndim == 0:
                action_arr = np.full(len(use_ids), int(action_arr.item()), dtype=np.int32)
            else:
                action_arr = action_arr.reshape(-1)
            self._last_action_info["requested_len"] = int(action_arr.size)
            phase_arr = action_arr
            duration_arr = np.zeros_like(phase_arr, dtype=np.float32)

        if len(use_ids) != len(phase_arr):
            self._last_action_info["skipped_reason"] = "action_count_mismatch"
            return
            
        try:
            for i, tl_id in enumerate(use_ids):
                phase = int(phase_arr[i])
                duration_mod = float(duration_arr[i])
                
                max_phase = 3
                if self._max_phase_per_tl and tl_id in self._max_phase_per_tl:
                    max_phase = self._max_phase_per_tl[tl_id]
                else:
                    max_phase = self._get_max_phase_index(tl_id)
                phase = max(0, min(phase, max_phase))
                
                traci.trafficlight.setPhase(tl_id, phase)
                
                if self.is_parameterized_action and duration_mod != 0.0:
                    current_duration = traci.trafficlight.getPhaseDuration(tl_id)
                    new_duration = max(5.0, current_duration + duration_mod) # min 5s safety
                    traci.trafficlight.setPhaseDuration(tl_id, new_duration)
                    
                self._last_action_info["applied_count"] += 1
                if len(self._last_action_info["applied_phases"]) < 16:
                    self._last_action_info["applied_phases"].append(phase)
            self._last_action_info["skipped_reason"] = None
        except Exception as e:
            self.sumo_running = False
            self._last_action_info["skipped_reason"] = f"traci_error: {e}"
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
                # SOTA: Phase 3 Adversarial Accident Injection
                if self.config.get("evaluation", {}).get("adversarial_accidents", False):
                    # Randomly stop 5 vehicles in the network to simulate a gridlock crash
                    if self.current_step == 500: # Trigger crash exactly at step 500
                        try:
                            veh_list = traci.vehicle.getIDList()
                            if len(veh_list) >= 5:
                                np.random.seed(42) # Deterministic crash nodes
                                crash_vehs = np.random.choice(veh_list, 5, replace=False)
                                for vid in crash_vehs:
                                    traci.vehicle.setSpeed(vid, 0.0)
                                    traci.vehicle.setColor(vid, (255, 0, 0, 255))
                                print(f"[Adversarial] Triggered artificial multi-car crash on {crash_vehs} at step 500!")
                        except Exception as e:
                            pass

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
    
    def _get_raw_observation(self) -> torch.Tensor:
        """Get the raw feature observation from the feature extractor."""
        features = self.feature_extractor.extract()
        tensor_feats = features.detach().clone().to(torch.float32) if torch.is_tensor(features) else torch.tensor(features, dtype=torch.float32)
        return tensor_feats

    def _get_observation(self) -> np.ndarray:
        """Get observation from GNN encoder (including local features and global embedding)."""
        if len(self.state_history) < self.state_history.maxlen:
            # Pad with zeros if we don't have enough history
            padding = [torch.zeros_like(self.state_history[0])] * (self.state_history.maxlen - len(self.state_history))
            history = padding + list(self.state_history)
        else:
            history = list(self.state_history)
        
        x_seq = torch.stack(history, dim=0).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            embedding, global_embedding, mean_forecast, variance_forecast = self.model(x_seq, self.edge_index)
            # Store forecasts for reward calculation
            self.last_mean_forecast = mean_forecast
            self.last_variance_forecast = variance_forecast

        # Create coordinated observations
        embedding_dim = embedding.shape[1]
        obs_dim = self.observation_space.shape[0]
        obs = np.zeros((self.num_intersections, obs_dim), dtype=np.float32)
        
        global_emb_np = global_embedding.cpu().numpy().flatten()

        for i in range(self.num_intersections):
            neighbors = self.edge_index[1][self.edge_index[0] == i]
            neighbor_embeddings = embedding[neighbors]
            
            # Pad neighbor embeddings
            padded_neighbors = np.zeros((self.max_neighbors, embedding_dim), dtype=np.float32)
            num_neighbors = min(len(neighbors), self.max_neighbors)
            padded_neighbors[:num_neighbors] = neighbor_embeddings.cpu().numpy()[:num_neighbors]
            
            # Concatenate self embedding, neighbor embeddings, and global embedding
            obs[i] = np.concatenate([
                embedding[i].cpu().numpy(), 
                padded_neighbors.flatten(),
                global_emb_np
            ])
            
        return obs
    
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
            
            # Phase 3: Risk-aware penalty (uses GNN forecast)
            if hasattr(self, "last_mean_forecast") and hasattr(self, "last_variance_forecast"):
                risk_penalty = self.reward_calculator.risk_model.calculate_risk(
                    self.last_mean_forecast,
                    self.last_variance_forecast
                )
                reward -= risk_penalty
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
    
    def _get_mean_speed(self) -> float:
        """Get the mean speed of all vehicles in the network."""
        if not self.sumo_running or not TRACI_AVAILABLE:
            return 0.0
        try:
            vehicle_ids = traci.vehicle.getIDList()
            if not vehicle_ids:
                return 0.0
            speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids]
            return np.mean(speeds) if speeds else 0.0
        except Exception:
            return 0.0

    def _get_stopped_vehicles_count(self) -> int:
        """Count vehicles with speed < 0.1 m/s."""
        if not self.sumo_running or not TRACI_AVAILABLE:
            return 0
        try:
            vehicle_ids = traci.vehicle.getIDList()
            stopped = 0
            for veh_id in vehicle_ids:
                if traci.vehicle.getSpeed(veh_id) < 0.1:
                    stopped += 1
            return stopped
        except Exception:
            return 0

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary with detailed metrics."""
        info = {
            "step": self.current_step,
            "sumo_running": self.sumo_running,
            "num_intersections": self.num_intersections,
            "travel_time": 0.0,
            "waiting_time": 0.0,
            "queue_length": 0.0,
            "departed": 0,
            "placeholder_mode": not self.sumo_running,
            "actions_requested_len": self._last_action_info.get("requested_len", 0),
            "traffic_light_count": self._last_action_info.get("traffic_light_count", 0),
            "actions_applied_count": self._last_action_info.get("applied_count", 0),
            "actions_skipped_reason": self._last_action_info.get("skipped_reason"),
            "actions_applied_phases": list(self._last_action_info.get("applied_phases", [])),
        }
        if self.sumo_running and TRACI_AVAILABLE:
            try:
                # Step-level metrics
                info["step_total_waiting_time"] = self._get_waiting_time_step()
                info["step_total_queue_length"] = self._get_queue_length_step()
                info["step_travel_time"] = self._travel_time_step
                info["step_mean_speed"] = self._get_mean_speed()
                info["step_stopped_vehicles"] = self._get_stopped_vehicles_count()
                info["step_arrived_vehicles"] = traci.simulation.getArrivedNumber()

                # Update episode-level metrics
                self.episode_metrics["episode_total_waiting_time"] += info["step_total_waiting_time"]
                self.episode_metrics["episode_total_queue_length"] += info["step_total_queue_length"]
                self.episode_metrics["episode_total_travel_time"] += info["step_travel_time"]
                self.episode_metrics["episode_stopped_vehicles"] += info["step_stopped_vehicles"]
                self.episode_metrics["episode_arrived_vehicles"] += traci.simulation.getArrivedNumber()
                self.episode_metrics["episode_steps"] += 1

                # Final episode metrics (averages)
                terminated_bool = self._is_terminated()
                truncated_bool = self.current_step >= self.max_steps
                
                if terminated_bool or truncated_bool:
                    total_steps = max(1, self.episode_metrics["episode_steps"])
                    info["episode_avg_waiting_time"] = self.episode_metrics["episode_total_waiting_time"] / total_steps
                    info["episode_avg_queue_length"] = self.episode_metrics["episode_total_queue_length"] / total_steps
                    info["episode_total_travel_time"] = self.episode_metrics["episode_total_travel_time"]
                    info["episode_avg_travel_time"] = self.episode_metrics["episode_total_travel_time"] / max(
                        1, self.episode_metrics["episode_arrived_vehicles"]
                    )
                    info["episode_throughput"] = self.episode_metrics["episode_arrived_vehicles"]
                    info["episode_avg_stopped_vehicles"] = self.episode_metrics["episode_stopped_vehicles"] / total_steps

            except Exception:
                pass
        return info
    
    def close(self) -> None:
        """Close the environment and log the final episode metrics."""
        if self.episode_metrics["episode_steps"] > 0:
            self.episode_count += 1
            self._log_episode(self.episode_count)
            self.episode_metrics["episode_steps"] = 0
            
        self._close_sumo()
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            None (SUMO GUI handles rendering)
        """
        # SUMO GUI handles rendering automatically
        return None


