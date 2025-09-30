from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import itertools
from collections import defaultdict

@dataclass
class ToyState:
    op_count: int
    spares: int
    health: int  # 0 = degraded, 1 = healthy
    coverage: int  # 0 = no coverage, 1 = coverage
    
    def to_tuple(self):
        return (self.op_count, self.spares, self.health, self.coverage)
    
    def is_terminal(self):
        return self.op_count == 0 and self.spares == 0

class ToyConstellationMDP:
    def __init__(self):
        # State space parameters
        self.op_counts = [0, 1, 2]
        self.spares = [0, 1]
        self.allowed_health = [0.0, 0.5, 1.0]  # Discretized health levels
        self.coverage_states = [0, 1]
        
        # Build state space
        self.states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        idx = 0
        for oc in self.op_counts:
            for sp in self.spares:
                for h in self.allowed_health:  # Use allowed_health instead
                    for cov in self.coverage_states:
                        state = (oc, sp, h, cov)
                        self.states.append(state)
                        self.state_to_idx[state] = idx
                        self.idx_to_state[idx] = state
                        idx += 1
        
        self.nS = len(self.states)
        self.actions = ["NO_OP", "REPLACE", "ACTIVATE_BACKUP", "BOOST"]
        self.nA = len(self.actions)
        
        # Recovery tracking parameters
        self.boost_history = defaultdict(int)
        self.recovery_progress = defaultdict(int)
        self.MAX_BOOSTS = 3
        self.RECOVERY_STEPS = 2  # Add this line: defines steps needed for full recovery
    
        # Verify health states match recovery steps
        assert len(self.allowed_health) == self.RECOVERY_STEPS + 1, \
            f"Health states ({len(self.allowed_health)}) should match recovery steps + 1 ({self.RECOVERY_STEPS + 1})"
        
    def is_terminal(self, state: Tuple) -> bool:
        """Check if state is terminal (no operational satellites and no spares)"""
        op_count, spares, _, _ = state
        return op_count == 0 and spares == 0
    
    def snap_health(self, h):
        """Helper function to discretize health values"""
        return min(self.allowed_health, key=lambda x: abs(x - h))

    def get_operational_reward(self, state: Tuple) -> float:
        """Calculate base operational reward based on constellation status"""
        oc, sp, h, cov = state
        
        # Base reward for operational capability
        if oc >= 2 and h >= 0.5:
            return 5    # Full operational capability
        elif oc >= 1 and h >= 0.5:
            return 2    # Reduced but functional
        elif oc >= 1 and h < 0.5:
            return -1   # Barely functional
        else:
            return -10  # Mission failure
        
    def get_coverage_reward(self, state: Tuple) -> float:
        """Calculate reward based on coverage capability"""
        oc, sp, h, cov = state
        
        if cov == 1 and oc >= 2:
            return 3    # Full coverage with redundancy
        elif cov == 1 and oc >= 1:
            return 1    # Coverage but no redundancy
        else:
            return -5   # No coverage - mission critical failure

    def calculate_total_reward(self, state: Tuple, action: str, base_action_reward: float) -> float:
        """Calculate total reward including operational and coverage components"""
        operational_reward = self.get_operational_reward(state)
        coverage_reward = self.get_coverage_reward(state)
        
        # Action-specific modifiers
        if action == "BOOST" and state[2] < 1.0:  # Reward proactive maintenance
            proactive_bonus = 2
        elif action == "REPLACE" and state[2] == 0.0:  # Reward necessary replacements
            proactive_bonus = 1
        else:
            proactive_bonus = 0
        
        return base_action_reward + operational_reward + coverage_reward + proactive_bonus

    def transition(self, state: Tuple, action: str) -> List[Tuple[float, Tuple, float]]:
        """Modified transition function with new reward structure"""
        oc, sp, h, cov = state
        
        if action == "BOOST":
            if h < 1.0:
                if h == 0.0:
                    next_state_success = (oc, sp, 0.5, 1)
                    next_state_fail = (oc, sp, 0.0, 0)
                    return [
                        (0.85, next_state_success, self.calculate_total_reward(next_state_success, action, 12)),
                        (0.15, next_state_fail, self.calculate_total_reward(next_state_fail, action, -3))
                    ]
                elif h == 0.5:
                    next_state_success = (oc, sp, 1.0, 1)
                    next_state_fail = (oc, sp, 0.5, 1)
                    return [
                        (0.75, next_state_success, self.calculate_total_reward(next_state_success, action, 8)),
                        (0.25, next_state_fail, self.calculate_total_reward(next_state_fail, action, -2))
                    ]
            else:
                return [(1.0, state, self.calculate_total_reward(state, action, -8))]
        
        elif action == "REPLACE" and sp > 0:
            # Only allow replacement if satellite is critically damaged or failed
            if h == 0.0 or oc < 2:  # Critical damage or insufficient operational satellites
                success_prob = 0.85 if h == 0.0 else 0.75
                next_state_success = (2, sp-1, 1.0, 1)
                next_state_fail = (oc, sp-1, 0.0, 0)
                return [
                    (success_prob, next_state_success, self.calculate_total_reward(next_state_success, action, 10)),
                    (1-success_prob, next_state_fail, self.calculate_total_reward(next_state_fail, action, -12))
                ]
            elif h < 1.0:  # Damaged but not critical - should try BOOST first
                return [(1.0, (oc, sp-1, h, cov), -20)]  # Heavy penalty for premature replacement
            else:  # Trying to replace healthy satellite - forbidden
                return [(1.0, (oc, sp, h, cov), -25)]  # Severe penalty
        
        elif action == "ACTIVATE_BACKUP" and sp > 0:
            # Only useful when operational capacity is low
            if oc < 2:  # Need more operational satellites
                success_prob = 0.8 if h >= 0.5 else 0.6  # Higher success if main constellation healthier
                return [
                    (success_prob, (min(oc+1, 2), sp-1, h, 1), 6),  # Successfully activated backup
                    (1-success_prob, (oc, sp-1, max(h-0.5, 0), 0), -5)  # Activation failed, system stress
                ]
            else:  # Already have sufficient operational satellites
                return [(1.0, (oc, sp, h, cov), -10)]  # Penalty for unnecessary activation
        
        else:  # NO_OP - natural evolution
            # Calculate degradation probability based on current health and operational stress
            if h == 1.0:  # Healthy satellites
                stable_prob = 0.85 if oc >= 2 else 0.75  # Less stable if overworked
                return [
                    (stable_prob, (oc, sp, 1.0, cov), 2),        # Stay healthy, small operational reward
                    (1-stable_prob, (oc, sp, 0.5, max(cov-1, 0)), -3)  # Minor degradation
                ]
            elif h == 0.5:  # Partially damaged
                # Can either recover naturally, stay same, or degrade further
                recover_prob = 0.2 if oc >= 2 else 0.1  # Better chance if not overworked
                degrade_prob = 0.3 if oc < 2 else 0.2   # Higher chance if overworked
                stable_prob = 1 - recover_prob - degrade_prob
                
                return [
                    (recover_prob, (oc, sp, 1.0, 1), 5),              # Natural recovery
                    (stable_prob, (oc, sp, 0.5, cov), -1),            # Remains damaged
                    (degrade_prob, (max(oc-1, 0), sp, 0.0, 0), -8)    # Critical failure
                ]
            else:  # h == 0.0 - critically damaged
                # High chance of complete failure, low chance of staying critical
                failure_prob = 0.6 if oc < 2 else 0.4   # Higher failure if overworked
                return [
                    (1-failure_prob, (oc, sp, 0.0, 0), -4),           # Stays critical
                    (failure_prob, (max(oc-1, 0), sp, 0.0, 0), -15)   # Complete satellite loss
                ]
    
    def reset_tracking(self):
        """Reset tracking dictionaries"""
        self.boost_history.clear()
        self.recovery_progress.clear()