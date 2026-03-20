"""
Reward Calculator Module

Calculates rewards for reinforcement learning based on traffic metrics.
Supports multi-objective rewards including waiting time, queue length, and anomaly scores.
"""

import numpy as np
from typing import Dict, List, Optional
from src.phase3.risk_model import CongestionRiskModel
import torch


class RewardCalculator:
    """
    Calculates rewards for RL agent based on traffic state.
    
    Reward function:
        R = -α₁·waiting_time - α₂·queue_length - α₃·anomaly_score + α₄·throughput
    
    Where:
        - waiting_time: Total waiting time across all vehicles
        - queue_length: Total queue length across all intersections
        - anomaly_score: Predicted anomaly score (optional, for Phase 3)
        - throughput: Vehicles departed (optional, rewards flow like Smartcities)
    """
    
    def __init__(
        self,
        waiting_time_weight: float = 0.1,
        queue_length_weight: float = 0.05,
        anomaly_weight: float = 0.0,
        throughput_weight: float = 0.0,
        pressure_weight: float = 0.0,
        speed_reward_weight: float = 0.0,
        emission_weight: float = 0.0,  # New: Multi-objective emission penalty
        fuel_weight: float = 0.0,      # New: Multi-objective fuel penalty
        adaptive_weighting: bool = True, # New: Self-adaptive reward mechanism
        normalize: bool = True,
        max_waiting: float = 300.0,
        max_queue: float = 100.0,
        max_throughput_per_step: float = 20.0,
        max_speed: float = 13.89,
        risk_density_threshold: float = 0.8,
        risk_penalty_factor: float = 1.0,
        risk_sensitivity: float = 0.5,  # Lambda for uncertainty penalty
    ):
        """
        Initialize reward calculator.
        
        Args:
            waiting_time_weight: Weight for waiting time penalty (α₁)
            queue_length_weight: Weight for queue length penalty (α₂)
            anomaly_weight: Weight for anomaly score penalty (α₃, for Phase 3)
            throughput_weight: Weight for throughput bonus (α₄; set > 0 to reward flow like Smartcities)
            pressure_weight: Weight for pressure term (PressLight-style; set > 0 with SUMO)
            speed_reward_weight: Weight for speed bonus (higher speed = better flow; guarantees differentiation)
            emission_weight: Weight for CO2 emission penalty
            fuel_weight: Weight for fuel consumption penalty
            adaptive_weighting: Whether to use self-adaptive reward shaping (Patent-ready)
            normalize: Whether to normalize metrics
            max_waiting: Maximum waiting time for normalization
            max_queue: Maximum queue length for normalization
            max_throughput_per_step: Maximum departed per step for throughput normalization
            max_speed: Maximum speed for normalization (m/s)
            risk_density_threshold: Density threshold for congestion risk model
            risk_penalty_factor: Penalty factor for congestion risk model
        """
        self.waiting_time_weight = waiting_time_weight
        self.queue_length_weight = queue_length_weight
        self.anomaly_weight = anomaly_weight
        self.throughput_weight = throughput_weight
        self.pressure_weight = pressure_weight
        self.speed_reward_weight = speed_reward_weight
        self.emission_weight = emission_weight
        self.fuel_weight = fuel_weight
        self.adaptive_weighting = adaptive_weighting
        self.normalize = normalize
        self.max_waiting = max_waiting
        self.max_queue = max_queue
        self.max_throughput_per_step = max_throughput_per_step
        self.max_speed = max_speed
        self.risk_model = CongestionRiskModel(
            density_threshold=risk_density_threshold,
            risk_penalty_factor=risk_penalty_factor,
            risk_sensitivity=risk_sensitivity
        )

    def _get_adaptive_weights(
        self, 
        density: float, 
        anomaly_severity: float, 
        sim_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Self-adaptive reward shaping mechanism.
        Dynamically adjusts weights based on real-time traffic conditions.
        (Patent Angle: A self-adaptive reward shaping mechanism for multi-agent traffic systems)
        """
        if not self.adaptive_weighting:
            return {
                "waiting": self.waiting_time_weight,
                "queue": self.queue_length_weight,
                "anomaly": self.anomaly_weight
            }

        # Base weights
        w_waiting = self.waiting_time_weight
        w_queue = self.queue_length_weight
        w_anomaly = self.anomaly_weight

        # 1. Density-based adjustment: If density is high, prioritize queue reduction
        if density > 0.7:
            w_queue *= (1.0 + density)
            w_waiting *= 0.8  # Slightly reduce waiting time priority to focus on clearing queues

        # 2. Anomaly-based adjustment: If anomaly is severe, prioritize safety/anomaly reduction
        if anomaly_severity > 0.5:
            w_anomaly *= (1.0 + anomaly_severity * 2)
            w_queue *= 1.2
            w_waiting *= 1.2 # Everything is more important during an anomaly

        # 3. Time-of-day adjustment (Simulated): Prioritize different metrics during peak hours
        if sim_time is not None:
            # Assume peak hours are 28800-36000 (8-10 AM) and 61200-68400 (5-7 PM)
            is_peak = (28800 <= sim_time <= 36000) or (61200 <= sim_time <= 68400)
            if is_peak:
                w_waiting *= 1.5  # People care more about delay during peak hours
                w_queue *= 1.3

        return {
            "waiting": w_waiting,
            "queue": w_queue,
            "anomaly": w_anomaly
        }

    def calculate(
        self,
        waiting_times: Dict[str, float],
        queue_lengths: Dict[str, float],
        anomaly_info: Optional[Dict[str, Dict]] = None,
        forecasted_state: Optional[torch.Tensor] = None,
        sim_time: Optional[float] = None
    ) -> float:
        """Calculate the reward based on the provided metrics and optional forecasted state."""
        reward = 0.0

        # Calculate total metrics
        total_waiting = sum(waiting_times.values())
        total_queue = sum(queue_lengths.values())
        
        # Calculate density proxy and anomaly severity for adaptive weighting
        num_nodes = max(1, len(waiting_times))
        avg_queue = total_queue / num_nodes
        density_proxy = min(1.0, avg_queue / self.max_queue)
        
        anomaly_severity = 0.0
        if anomaly_info:
            anomaly_severity = np.mean([info.get('smoothed_score', 0.0) for info in anomaly_info.values()])

        # Get adaptive weights
        weights = self._get_adaptive_weights(density_proxy, anomaly_severity, sim_time)

        if self.normalize:
            total_waiting = total_waiting / max(1e-6, self.max_waiting * len(waiting_times))
            total_queue = total_queue / max(1e-6, self.max_queue * len(queue_lengths))

        reward -= weights["waiting"] * total_waiting
        reward -= weights["queue"] * total_queue

        # Add anomaly penalty if provided
        if anomaly_info is not None and weights["anomaly"] > 0:
            from src.phase3.integration import get_anomaly_controller
            controller = get_anomaly_controller()
            if controller is not None:
                anomaly_penalty = controller.get_anomaly_penalty(anomaly_info)
                reward -= weights["anomaly"] * anomaly_penalty

        # Risk-aware penalty (uses forecasted state)
        if forecasted_state is not None:
            # Check if forecasted_state is a tuple (mean, variance)
            if isinstance(forecasted_state, tuple) and len(forecasted_state) == 2:
                risk_penalty = self.risk_model.calculate_risk(forecasted_state[0], forecasted_state[1])
            else:
                # Fallback: create dummy variance if not provided
                dummy_variance = torch.zeros_like(forecasted_state)
                risk_penalty = self.risk_model.calculate_risk(forecasted_state, dummy_variance)
            reward -= risk_penalty

        return reward
    
    def add_throughput_bonus(self, reward: float, departed_count: float) -> float:
        """Add throughput bonus to reward (call when throughput_weight > 0)."""
        if self.throughput_weight <= 0:
            return reward
        norm = min(1.0, departed_count / max(1e-6, self.max_throughput_per_step))
        return reward + self.throughput_weight * norm
    
    def calculate_from_sumo(
        self,
        intersections: list,
        anomaly_info: Optional[Dict[str, Dict]] = None,
    ) -> float:
        """
        Calculate reward directly from SUMO via TraCI.
        
        Args:
            intersections: List of intersection IDs
            anomaly_info: Optional dict mapping intersection_id to anomaly info
            
        Returns:
            Reward value
        """
        try:
            import traci
        except ImportError:
            # Return placeholder reward if TraCI not available
            return self._calculate_placeholder(intersections)
        
        waiting_times = {}
        queue_lengths = {}
        # Use TraCI's traffic light IDs when SUMO is running (handles graph placeholder vs net IDs, e.g. J0 vs A0)
        try:
            tl_ids = traci.trafficlight.getIDList()
        except Exception:
            tl_ids = []
        use_ids = tl_ids if tl_ids else intersections

        try:
            for intersection_id in use_ids:
                # Get controlled lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
                
                intersection_waiting = 0.0
                intersection_queue = 0.0
                
                for lane_id in controlled_lanes:
                    # Waiting time
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    intersection_waiting += waiting_time
                    
                    # Queue length
                    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                    intersection_queue += queue_length
                
                waiting_times[intersection_id] = intersection_waiting
                queue_lengths[intersection_id] = intersection_queue
        
        except Exception as e:
            # Fallback to placeholder on error; warn once to avoid spamming
            if not getattr(self, "_sumo_reward_warned", False):
                self._sumo_reward_warned = True
                print(f"Warning: Error calculating reward from SUMO: {e}")
            return self._calculate_placeholder(intersections)
        
        # When lane-based waiting is 0, use real vehicle-based waiting time (no proxy)
        total_waiting_sum = sum(waiting_times.values())
        if total_waiting_sum == 0:
            try:
                vehicle_waiting = 0.0
                for veh_id in traci.vehicle.getIDList():
                    try:
                        vehicle_waiting += traci.vehicle.getWaitingTime(veh_id)
                    except Exception:
                        pass
                if vehicle_waiting > 0:
                    n = max(len(use_ids), 1)
                    for intersection_id in use_ids:
                        waiting_times[intersection_id] = vehicle_waiting / n
            except Exception:
                pass

        try:
            sim_time = traci.simulation.getTime()
        except Exception:
            sim_time = None

        reward = self.calculate(waiting_times, queue_lengths, anomaly_info, sim_time=sim_time)
        # Pressure penalty: vehicle count on controlled lanes (non-zero when traffic present; differentiates policies)
        if self.pressure_weight > 0:
            try:
                total_vehicles_on_lanes = 0.0
                for intersection_id in use_ids:
                    for lane_id in traci.trafficlight.getControlledLanes(intersection_id):
                        total_vehicles_on_lanes += traci.lane.getLastStepVehicleNumber(lane_id)
                reward -= self.pressure_weight * total_vehicles_on_lanes
            except Exception:
                pass
        # Speed bonus: higher speed = better flow (GUARANTEES differentiation when policies differ)
        if self.speed_reward_weight > 0:
            try:
                total_speed = 0.0
                lane_count = 0
                for intersection_id in use_ids:
                    for lane_id in traci.trafficlight.getControlledLanes(intersection_id):
                        total_speed += traci.lane.getLastStepMeanSpeed(lane_id)
                        lane_count += 1
                if lane_count > 0:
                    avg_speed = total_speed / lane_count
                    if self.normalize:
                        avg_speed = avg_speed / self.max_speed
                    reward += self.speed_reward_weight * avg_speed
            except Exception:
                pass
        
        # New: Multi-objective Emission and Fuel Penalties
        if self.emission_weight > 0 or self.fuel_weight > 0:
            try:
                total_emission = 0.0
                total_fuel = 0.0
                for intersection_id in use_ids:
                    for lane_id in traci.trafficlight.getControlledLanes(intersection_id):
                        if self.emission_weight > 0:
                            total_emission += traci.lane.getCO2Emission(lane_id)
                        if self.fuel_weight > 0:
                            total_fuel += traci.lane.getFuelConsumption(lane_id)
                
                if self.normalize:
                    # Very rough normalization for emissions (mg/s) and fuel (ml/s)
                    total_emission /= 10000.0 
                    total_fuel /= 1000.0
                
                reward -= self.emission_weight * total_emission
                reward -= self.fuel_weight * total_fuel
            except Exception:
                pass

        # Throughput bonus (Smartcities-style multi-objective: reward flow)
        if self.throughput_weight > 0:
            try:
                departed = traci.simulation.getDepartedNumber()
                reward = self.add_throughput_bonus(reward, float(departed))
            except Exception:
                pass
        return reward
    
    def _calculate_placeholder(self, intersections: list, anomaly_info: Optional[Dict[str, Dict]] = None) -> float:
        """
        Calculate placeholder reward for testing.
        
        Args:
            intersections: List of intersection IDs
            anomaly_info: Optional dict mapping intersection_id to anomaly info
            
        Returns:
            Placeholder reward value
        """
        # Generate random metrics for testing
        num_intersections = len(intersections)
        total_waiting = np.random.uniform(0, self.max_waiting * num_intersections)
        total_queue = np.random.uniform(0, self.max_queue * num_intersections)
        
        if self.normalize:
            total_waiting = total_waiting / self.max_waiting
            total_queue = total_queue / self.max_queue
        
        reward = -self.waiting_time_weight * total_waiting - self.queue_length_weight * total_queue
        
        # Add anomaly penalty if provided
        if anomaly_info is not None and self.anomaly_weight > 0:
            from src.phase3.integration import get_anomaly_controller
            controller = get_anomaly_controller()
            if controller is not None:
                anomaly_penalty = controller.get_anomaly_penalty(anomaly_info)
                reward -= anomaly_penalty
        
        return float(reward)
    
    def get_reward_components(
        self,
        waiting_times: Dict[str, float],
        queue_lengths: Dict[str, float],
        anomaly_info: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, float]:
        """
        Get individual reward components for analysis.
        
        Args:
            waiting_times: Dict mapping intersection_id to waiting time
            queue_lengths: Dict mapping intersection_id to queue length
            anomaly_info: Optional dict mapping intersection_id to anomaly info
            
        Returns:
            Dictionary with reward components
        """
        total_waiting = sum(waiting_times.values())
        total_queue = sum(queue_lengths.values())
        
        if self.normalize:
            total_waiting = total_waiting / self.max_waiting
            total_queue = total_queue / self.max_queue
        
        components = {
            "waiting_time_penalty": -self.waiting_time_weight * total_waiting,
            "queue_length_penalty": -self.queue_length_weight * total_queue,
        }
        
        if anomaly_info is not None and self.anomaly_weight > 0:
            from src.phase3.integration import get_anomaly_controller
            controller = get_anomaly_controller()
            if controller is not None:
                components["anomaly_penalty"] = -controller.get_anomaly_penalty(anomaly_info)
        
        components["total_reward"] = sum(components.values())
        
        return components


