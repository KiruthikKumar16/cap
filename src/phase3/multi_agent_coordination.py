"""
Multi-Agent Coordination Module for Phase 3

Enables coordinated anomaly-aware control across multiple intersections.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CoordinationMessage:
    """Message exchanged between agents."""

    source_intersection: str
    target_intersection: str
    anomaly_severity: float
    recommended_action: str
    confidence: float


class MultiAgentCoordinator:
    """Coordinates traffic control across multiple intersections."""

    def __init__(
        self,
        intersections: List[str],
        communication_radius: int = 2,
        coordination_weight: float = 0.1,
    ):
        """
        Initialize multi-agent coordinator.

        Args:
            intersections: List of all intersection IDs
            communication_radius: Max hops for message propagation
            coordination_weight: Weight for coordination influence in rewards
        """
        self.intersections = intersections
        self.communication_radius = communication_radius
        self.coordination_weight = coordination_weight

        # Build adjacency matrix (simplified: distance-based)
        self.adjacency_matrix = self._build_adjacency_matrix(intersections)

        # Message queue for coordination
        self.message_queue: List[CoordinationMessage] = []

        # Coordination state
        self.consensus_actions = {}
        self.message_history = defaultdict(list)

    def _build_adjacency_matrix(self, intersections: List[str]) -> np.ndarray:
        """
        Build adjacency matrix for intersection network.

        Simplified: assumes grid layout (e.g., A0, A1, B0, B1).
        """
        n = len(intersections)
        adj = np.zeros((n, n))

        for i, int_i in enumerate(intersections):
            for j, int_j in enumerate(intersections):
                if i == j:
                    continue

                # Parse intersection names (e.g., 'A0' -> row 0, col 0)
                try:
                    row_i, col_i = ord(int_i[0]) - ord("A"), int(int_i[1])
                    row_j, col_j = ord(int_j[0]) - ord("A"), int(int_j[1])

                    # Distance-based adjacency
                    distance = abs(row_i - row_j) + abs(col_i - col_j)
                    if distance <= self.communication_radius:
                        adj[i, j] = 1.0 / (1.0 + distance)  # Closer = stronger link
                except (IndexError, ValueError):
                    pass

        return adj

    def broadcast_anomaly(
        self,
        source: str,
        anomaly_severity: float,
        recommended_action: str = "none",
        confidence: float = 1.0,
    ) -> None:
        """
        Broadcast anomaly detection to neighboring intersections.

        Args:
            source: Intersection with detected anomaly
            anomaly_severity: Severity of anomaly
            recommended_action: Suggested action for neighbors
            confidence: Confidence in detection
        """
        for target in self.intersections:
            if target == source:
                continue

            source_idx = self.intersections.index(source)
            target_idx = self.intersections.index(target)

            if self.adjacency_matrix[source_idx, target_idx] > 0:
                message = CoordinationMessage(
                    source_intersection=source,
                    target_intersection=target,
                    anomaly_severity=anomaly_severity,
                    recommended_action=recommended_action,
                    confidence=confidence,
                )
                self.message_queue.append(message)
                self.message_history[target].append(message)

    def process_messages(self) -> Dict[str, List[CoordinationMessage]]:
        """
        Process incoming coordination messages.

        Returns:
            Dict mapping intersection_id to received messages
        """
        received = defaultdict(list)
        for message in self.message_queue:
            received[message.target_intersection].append(message)

        self.message_queue.clear()
        return dict(received)

    def compute_consensus_action(
        self,
        intersection_id: str,
        local_anomaly_score: float,
        received_messages: List[CoordinationMessage],
    ) -> str:
        """
        Compute consensus action based on local state and neighbor info.

        Args:
            intersection_id: Current intersection ID
            local_anomaly_score: Local anomaly score
            received_messages: Messages from neighbors

        Returns:
            Recommended action
        """
        if not received_messages and local_anomaly_score < 0.5:
            return "normal"

        # Aggregate neighbor severity
        neighbor_severities = [msg.anomaly_severity for msg in received_messages]
        max_neighbor_severity = (
            max(neighbor_severities) if neighbor_severities else 0.0
        )

        # Weighted combination
        combined_severity = (
            0.6 * local_anomaly_score + 0.4 * max_neighbor_severity
        )

        # Determine action
        if combined_severity > 0.8:
            return "urgent_control"
        elif combined_severity > 0.6:
            return "coordinated_control"
        elif combined_severity > 0.4:
            return "cooperative_control"
        else:
            return "normal"

    def get_coordination_bonus(
        self, intersection_id: str, taken_action: str, consensus_action: str
    ) -> float:
        """
        Get reward bonus for coordinated action.

        Args:
            intersection_id: Intersection ID
            taken_action: Action taken by local agent
            consensus_action: Consensus recommended action

        Returns:
            Bonus reward
        """
        if taken_action == consensus_action or (
            taken_action == "normal" and consensus_action == "normal"
        ):
            return 0.0  # No bonus for agreement
        else:
            # Penalize deviation from consensus
            return -self.coordination_weight

    def get_coordination_summary(self) -> Dict:
        """Get summary of coordination state."""
        return {
            "num_messages_processed": sum(
                len(msgs) for msgs in self.message_history.values()
            ),
            "consensus_actions": self.consensus_actions,
            "active_intersections": len(self.message_history),
        }
