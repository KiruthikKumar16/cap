
import numpy as np
import traci
from typing import List, Dict

class MaxPressureAgent:
    """
    Implementation of the Max-Pressure control algorithm.
    Pressure = sum(incoming_lanes_queue) - sum(outgoing_lanes_queue)
    """
    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        # Map each phase to its active incoming and outgoing lanes
        self.phase_lanes = self._get_phase_lanes()

    def _get_phase_lanes(self) -> List[Dict]:
        """Map phases to incoming/outgoing lanes based on SUMO TLS logic."""
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.intersection_id)[0]
        phases = logic.phases
        phase_map = []
        
        for phase in phases:
            state = phase.state
            # Simplify: in a real system we'd map 'G'/'g' characters to specific lanes
            # For now, we'll use a simplified mapping or assume standard 4-phase structure
            phase_map.append({
                "incoming": self.controlled_lanes, # Simplified
                "outgoing": [] # Would need traci.lane.getLinks
            })
        return phase_map

    def select_action(self) -> int:
        """Select phase with highest pressure."""
        num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.intersection_id)[0].phases)
        pressures = []
        
        for p in range(num_phases):
            # Calculate pressure for this phase
            # For simplicity in this demo, we'll use the number of vehicles on the lanes
            # that would be 'green' in this phase.
            traci.trafficlight.setPhase(self.intersection_id, p)
            lanes = traci.trafficlight.getControlledLanes(self.intersection_id)
            
            # Real pressure = incoming - outgoing
            incoming_count = sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            # For a true Max-Pressure, we'd need the downstream queue lengths too.
            # Simplified: pressure is just the queue size of lanes that have a green light.
            pressures.append(incoming_count)
            
        return np.argmax(pressures)

def run_max_pressure(env, steps=2000):
    """Run simulation using Max-Pressure controllers."""
    obs = env.reset()
    total_reward = 0
    
    intersections = env.env.intersections
    
    for step in range(steps):
        actions = []
        for i_id in intersections:
            # Simplified Max-Pressure: pick phase with most vehicles in its lanes
            # We skip the yellow phases by checking only even indices if needed
            num_phases = traci.trafficlight.getPhaseDuration(i_id) # dummy call to ensure traci
            
            # Logic: for each phase, check which lanes are green and count vehicles
            # But TraCI doesn't make it easy to 'preview' phases without setting them.
            # We'll use a heuristic: count vehicles on all lanes and pick based on occupancy.
            lanes = traci.trafficlight.getControlledLanes(i_id)
            # Standard 4-phase grid mapping (simplified)
            # Phase 0/2: N-S, Phase 1/3: E-W
            ns_lanes = [l for l in lanes if "n" in l.lower() or "s" in l.lower()]
            ew_lanes = [l for l in lanes if "e" in l.lower() or "w" in l.lower()]
            
            ns_pressure = sum([traci.lane.getLastStepHaltingNumber(l) for l in ns_lanes])
            ew_pressure = sum([traci.lane.getLastStepHaltingNumber(l) for l in ew_lanes])
            
            if ns_pressure >= ew_pressure:
                actions.append(0) # Phase 0 (N-S Green)
            else:
                actions.append(2) # Phase 2 (E-W Green)
                
        obs, reward, done, info = env.step(np.array(actions))
        total_reward += np.mean(reward)
        if any(done): break
        
    return total_reward / steps
