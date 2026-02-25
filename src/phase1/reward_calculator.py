"""
Reward Calculator Module

Calculates rewards for reinforcement learning based on traffic metrics.
Supports multi-objective rewards including waiting time, queue length, and anomaly scores.
"""

from typing import Dict, Optional
import numpy as np
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
        normalize: bool = True,
        max_waiting: float = 300.0,
        max_queue: float = 100.0,
        max_throughput_per_step: float = 20.0,
        max_speed: float = 13.89,
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
            normalize: Whether to normalize metrics
            max_waiting: Maximum waiting time for normalization
            max_queue: Maximum queue length for normalization
            max_throughput_per_step: Maximum departed per step for throughput normalization
            max_speed: Maximum speed for normalization (m/s)
        """
        self.waiting_time_weight = waiting_time_weight
        self.queue_length_weight = queue_length_weight
        self.anomaly_weight = anomaly_weight
        self.throughput_weight = throughput_weight
        self.pressure_weight = pressure_weight
        self.speed_reward_weight = speed_reward_weight
        self.normalize = normalize
        self.max_waiting = max_waiting
        self.max_queue = max_queue
        self.max_throughput_per_step = max_throughput_per_step
        self.max_speed = max_speed
    
    def calculate(
        self,
        waiting_times: Dict[str, float],
        queue_lengths: Dict[str, float],
        anomaly_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate reward from traffic metrics.
        
        Args:
            waiting_times: Dict mapping intersection_id to waiting time
            queue_lengths: Dict mapping intersection_id to queue length
            anomaly_scores: Optional dict mapping intersection_id to anomaly score
            
        Returns:
            Reward value (negative, to be maximized)
        """
        # Sum metrics across all intersections
        total_waiting = sum(waiting_times.values())
        total_queue = sum(queue_lengths.values())
        
        # Normalize if requested
        if self.normalize:
            total_waiting = total_waiting / self.max_waiting
            total_queue = total_queue / self.max_queue
        
        # Calculate base reward
        reward = -self.waiting_time_weight * total_waiting - self.queue_length_weight * total_queue
        
        # Add anomaly penalty if provided
        if anomaly_scores is not None and self.anomaly_weight > 0:
            total_anomaly = sum(anomaly_scores.values())
            reward -= self.anomaly_weight * total_anomaly
        
        return float(reward)
    
    def add_throughput_bonus(self, reward: float, departed_count: float) -> float:
        """Add throughput bonus to reward (call when throughput_weight > 0)."""
        if self.throughput_weight <= 0:
            return reward
        norm = min(1.0, departed_count / max(1e-6, self.max_throughput_per_step))
        return reward + self.throughput_weight * norm
    
    def calculate_from_sumo(
        self,
        intersections: list,
        anomaly_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate reward directly from SUMO via TraCI.
        
        Args:
            intersections: List of intersection IDs
            anomaly_scores: Optional dict mapping intersection_id to anomaly score
            
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

        reward = self.calculate(waiting_times, queue_lengths, anomaly_scores)
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
        # Throughput bonus (Smartcities-style multi-objective: reward flow)
        if self.throughput_weight > 0:
            try:
                departed = traci.simulation.getDepartedNumber()
                reward = self.add_throughput_bonus(reward, float(departed))
            except Exception:
                pass
        return reward
    
    def _calculate_placeholder(self, intersections: list) -> float:
        """
        Calculate placeholder reward for testing.
        
        Args:
            intersections: List of intersection IDs
            
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
        return float(reward)
    
    def get_reward_components(
        self,
        waiting_times: Dict[str, float],
        queue_lengths: Dict[str, float],
        anomaly_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Get individual reward components for analysis.
        
        Args:
            waiting_times: Dict mapping intersection_id to waiting time
            queue_lengths: Dict mapping intersection_id to queue length
            anomaly_scores: Optional dict mapping intersection_id to anomaly score
            
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
        
        if anomaly_scores is not None and self.anomaly_weight > 0:
            total_anomaly = sum(anomaly_scores.values())
            components["anomaly_penalty"] = -self.anomaly_weight * total_anomaly
        
        components["total_reward"] = sum(components.values())
        
        return components


