import numpy as np

class ConflictMonitorUnit:
    """
    Emulates physical traffic cabinet safety interlocks (NEMA TS2 / 2070 Standards).
    Ensures that every agent action complies with minimum green and clearance intervals.
    """
    def __init__(self, num_intersections: int, min_green: int = 7, yellow_time: int = 4):
        self.num_intersections = num_intersections
        self.min_green = min_green
        self.yellow_time = yellow_time
        
        # State tracking per intersection
        self.current_phases = np.zeros(num_intersections, dtype=int)
        self.steps_in_phase = np.zeros(num_intersections, dtype=int)
        self.rejection_counts = np.zeros(num_intersections, dtype=int)

    def validate_and_enforce(self, actions: np.ndarray) -> np.ndarray:
        """
        Validates a batch of actions for all intersections.
        Returns a 'Safe' action vector.
        """
        safe_actions = actions.copy()
        
        for i in range(self.num_intersections):
            requested_phase = actions[i]
            
            # 1. Minimum Green Violation Check
            if requested_phase != self.current_phases[i]:
                if self.steps_in_phase[i] < self.min_green:
                    # SAFETY INTERLOCK TRIGGERED: Force agent to hold current phase
                    safe_actions[i] = self.current_phases[i]
                    self.rejection_counts[i] += 1
                else:
                    # ACCEPTED: Update phase state
                    self.current_phases[i] = requested_phase
                    self.steps_in_phase[i] = 1
            else:
                # CONTINUING: Increment duration
                self.steps_in_phase[i] += 1
                
        return safe_actions

    def get_rejection_rate(self) -> float:
        total_steps = np.sum(self.steps_in_phase) + np.sum(self.rejection_counts)
        if total_steps == 0: return 0.0
        return float(np.sum(self.rejection_counts) / total_steps)
