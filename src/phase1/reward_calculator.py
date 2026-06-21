"""
Reward Calculator Module

Per-step reward for MAPPO traffic control. Primary signal: negative normalized
waiting time (aligned with SUMO episode metrics). Optional terms are gated off
until the baseline learns.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from src.phase3.risk_model import CongestionRiskModel


class RewardCalculator:
    """
    Calculates per-step rewards from SUMO traffic state.

    Core (baseline) reward:
        normalized_wait = total_wait / (num_vehicles + eps)
        reward = -clip(normalized_wait / wait_scale, 0, 1)

    Final reward is clipped to [-1, 1].
    """

    def __init__(
        self,
        waiting_time_weight: float = 1.0,
        queue_length_weight: float = 0.0,
        anomaly_weight: float = 0.0,
        throughput_weight: float = 0.0,
        pressure_weight: float = 0.0,
        speed_reward_weight: float = 0.0,
        emission_weight: float = 0.0,
        fuel_weight: float = 0.0,
        adaptive_weighting: bool = False,
        normalize: bool = True,
        max_waiting: float = 300.0,
        max_queue: float = 100.0,
        max_throughput_per_step: float = 20.0,
        max_speed: float = 13.89,
        wait_normalization_scale: float = 300.0,
        reward_clip: float = 1.0,
        risk_density_threshold: float = 0.8,
        risk_penalty_factor: float = 1.0,
        risk_sensitivity: float = 0.5,
    ):
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
        self.wait_normalization_scale = max(1e-6, float(wait_normalization_scale))
        self.reward_clip = float(reward_clip)
        self.risk_model = CongestionRiskModel(
            density_threshold=risk_density_threshold,
            risk_penalty_factor=risk_penalty_factor,
            risk_sensitivity=risk_sensitivity,
        )

    @staticmethod
    def _clip_reward(reward: float, clip: float) -> float:
        return float(np.clip(reward, -clip, clip))

    def _normalized_wait_penalty(
        self,
        total_wait: float,
        num_vehicles: float,
    ) -> float:
        """Negative normalized wait in [0, 1] before weighting."""
        denom = max(num_vehicles, 1.0)
        avg_wait_per_vehicle = total_wait / denom
        if self.normalize:
            norm = min(1.0, avg_wait_per_vehicle / self.wait_normalization_scale)
        else:
            norm = avg_wait_per_vehicle / self.wait_normalization_scale
        return -norm

    def calculate(
        self,
        waiting_times: Dict[str, float],
        queue_lengths: Dict[str, float],
        anomaly_info: Optional[Dict[str, Dict]] = None,
        forecasted_state=None,
        sim_time: Optional[float] = None,
        mean_speed: float = 0.0,
        num_vehicles: Optional[float] = None,
    ) -> float:
        """
        Compute reward from per-intersection metrics.

        Uses total network waiting time normalized by active vehicle count.
        Queue, speed, and anomaly terms are disabled until baseline works.
        """
        del forecasted_state, sim_time, mean_speed  # reserved for later phases

        total_wait = float(sum(waiting_times.values()))
        if num_vehicles is None:
            num_vehicles = float(max(len(waiting_times), 1))

        wait_penalty = self._normalized_wait_penalty(total_wait, num_vehicles)
        reward = self.waiting_time_weight * wait_penalty

        # Optional queue term (off by default: queue_length_weight=0)
        if self.queue_length_weight > 0 and queue_lengths:
            total_queue = float(sum(queue_lengths.values()))
            num_nodes = max(1, len(queue_lengths))
            avg_queue = total_queue / num_nodes
            norm_queue = min(1.0, avg_queue / max(1e-6, self.max_queue))
            reward -= self.queue_length_weight * norm_queue

        return self._clip_reward(reward, self.reward_clip)

    def add_throughput_bonus(self, reward: float, departed_count: float) -> float:
        if self.throughput_weight <= 0:
            return reward
        norm = min(1.0, departed_count / max(1e-6, self.max_throughput_per_step))
        return self._clip_reward(reward + self.throughput_weight * norm, self.reward_clip)

    def calculate_from_sumo(
        self,
        intersections: list,
        anomaly_info: Optional[Dict[str, Dict]] = None,
    ) -> float:
        """Calculate reward directly from SUMO via TraCI."""
        try:
            import traci
        except ImportError:
            return self._calculate_placeholder(intersections)

        waiting_times: Dict[str, float] = {}
        queue_lengths: Dict[str, float] = {}

        try:
            tl_ids = traci.trafficlight.getIDList()
        except Exception:
            tl_ids = []
        use_ids = tl_ids if tl_ids else intersections

        try:
            for intersection_id in use_ids:
                controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
                intersection_waiting = 0.0
                intersection_queue = 0.0
                for lane_id in controlled_lanes:
                    intersection_waiting += traci.lane.getWaitingTime(lane_id)
                    intersection_queue += traci.lane.getLastStepHaltingNumber(lane_id)
                waiting_times[intersection_id] = intersection_waiting
                queue_lengths[intersection_id] = intersection_queue
        except Exception as e:
            if not getattr(self, "_sumo_reward_warned", False):
                self._sumo_reward_warned = True
                print(f"Warning: Error calculating reward from SUMO: {e}")
            return self._calculate_placeholder(intersections)

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
                    per_node = vehicle_waiting / n
                    for intersection_id in use_ids:
                        waiting_times[intersection_id] = per_node
            except Exception:
                pass

        try:
            num_vehicles = float(len(traci.vehicle.getIDList()))
        except Exception:
            num_vehicles = float(max(len(use_ids), 1))

        reward = self.calculate(
            waiting_times,
            queue_lengths,
            anomaly_info,
            num_vehicles=num_vehicles,
        )

        if self.pressure_weight > 0:
            try:
                total_vehicles_on_lanes = 0.0
                for intersection_id in use_ids:
                    for lane_id in traci.trafficlight.getControlledLanes(intersection_id):
                        total_vehicles_on_lanes += traci.lane.getLastStepVehicleNumber(lane_id)
                norm_pressure = min(1.0, total_vehicles_on_lanes / max(1.0, num_vehicles))
                reward -= self.pressure_weight * norm_pressure
            except Exception:
                pass

        if self.throughput_weight > 0:
            try:
                departed = traci.simulation.getDepartedNumber()
                reward = self.add_throughput_bonus(reward, float(departed))
            except Exception:
                pass

        return self._clip_reward(reward, self.reward_clip)

    def _calculate_placeholder(
        self,
        intersections: list,
        anomaly_info: Optional[Dict[str, Dict]] = None,
    ) -> float:
        num_intersections = max(len(intersections), 1)
        total_waiting = np.random.uniform(0, self.wait_normalization_scale * num_intersections)
        num_vehicles = float(np.random.randint(1, max(2, num_intersections * 4)))
        reward = self.waiting_time_weight * self._normalized_wait_penalty(
            total_waiting, num_vehicles
        )
        return self._clip_reward(reward, self.reward_clip)

    def get_reward_components(
        self,
        waiting_times: Dict[str, float],
        queue_lengths: Dict[str, float],
        anomaly_info: Optional[Dict[str, Dict]] = None,
        num_vehicles: Optional[float] = None,
    ) -> Dict[str, float]:
        total_wait = float(sum(waiting_times.values()))
        if num_vehicles is None:
            num_vehicles = float(max(len(waiting_times), 1))

        wait_penalty = self.waiting_time_weight * self._normalized_wait_penalty(
            total_wait, num_vehicles
        )
        components = {"waiting_time_penalty": wait_penalty}

        if self.queue_length_weight > 0 and queue_lengths:
            total_queue = float(sum(queue_lengths.values()))
            norm_queue = min(1.0, total_queue / max(1e-6, self.max_queue * len(queue_lengths)))
            components["queue_length_penalty"] = -self.queue_length_weight * norm_queue

        components["total_reward"] = self._clip_reward(
            sum(components.values()), self.reward_clip
        )
        return components
