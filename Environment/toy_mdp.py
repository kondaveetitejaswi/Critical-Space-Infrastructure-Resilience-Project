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

    def snap_state(self, state):
        """Ensure next state’s health value matches allowed discrete levels"""
        oc, sp, h, cov = state
        return (oc, sp, self.snap_health(h), cov)


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
            return 0   # No coverage - mission critical failure

    def calculate_total_reward(self, state: Tuple, action: str, base_action_reward: float) -> float:
        """Calculate total reward including operational and coverage components"""
        operational_reward = self.get_operational_reward(state)
        coverage_reward = self.get_coverage_reward(state)
        
        # # Action-specific modifiers 
        """
        These immediate proactive rewards are actually useless in our implementation and cause issues in the rewarding wise.
        """
        # if action == "BOOST" and state[2] < 1.0:  # Reward proactive maintenance
        #     proactive_bonus = 2
        # elif action == "REPLACE" and state[2] == 0.0:  # Reward necessary replacements
        #     proactive_bonus = 1
        # elif action == "BOOST" and state[2] == 1.0:
        #     proactive_bonus = -5  # Penalize unnecessary boosts
        # elif action == "NO_OP" and state[0] >= 2:
        #     proactive_bonus = 2
        # else:
        #     proactive_bonus = 0
        
        return base_action_reward + operational_reward + coverage_reward 

    def transition(self, state: Tuple, action: str) -> List[Tuple[float, Tuple, float]]:
        """
        Transition function with realistic and well-separated satellite operations.
        Each action has distinct meaning and time evolution to prevent overlap.
        """
        oc, sp, h, cov = state

        # Terminal state check
        if self.is_terminal(state):
            return [(1.0, state, 0)]

        # 1️⃣ NO_OP — Maintain current system
        if action == "NO_OP":
            if oc == 2 and sp >= 0 and h == 1.0 and cov == 1:
                # Perfectly healthy — stay same with reward
                # FIX: Use calculate_total_reward for consistency
                total_reward = self.calculate_total_reward(state, action, 2)  # Base reward of 2
                return [(1.0, state, total_reward)]
            else:
                degrade_prob = 0.1 if h == 1.0 else (0.3 if h == 0.5 else 0.5)
                degraded_state = self.snap_state((max(oc - 1, 0), sp, max(h - 0.5, 0.0), 0))
                stable_state = self.snap_state((oc, sp, h, cov))
                return [
                    (1 - degrade_prob, stable_state,
                    self.calculate_total_reward(stable_state, action, 1)),
                    (degrade_prob, degraded_state,
                    self.calculate_total_reward(degraded_state, action, -2))
                ]


        # 2️⃣ BOOST — Gradually improve health
        if action == "BOOST":
            if h < 1.0:
                improvement_prob = 0.7 if h == 0.0 else 0.8
                next_health = 0.5 if h == 0.0 else 1.0
                next_state_success = self.snap_state((oc, sp, next_health, 1))
                next_state_fail = self.snap_state((oc, sp, h, cov))
                return [
                    (improvement_prob, next_state_success,
                    self.calculate_total_reward(next_state_success, action, 6)),
                    (1 - improvement_prob, next_state_fail,
                    self.calculate_total_reward(next_state_fail, action, -2))
                ]
            else:
                # FIX: Make penalty even stronger for boosting healthy system
                # Use calculate_total_reward but with very negative base
                return [(1.0, self.snap_state(state), 
                        self.calculate_total_reward(self.snap_state(state), action, -30))]  # Increased penalty

        # 3️⃣ REPLACE — Replace completely failed satellite using spare
        if action == "REPLACE":
            if sp > 0 and h == 0.0:
                # Replacement begins (partial restoration stage)
                start_prob = 0.9
                next_state_partial = self.snap_state((min(oc + 1, 2), sp - 1, 0.5, 1))
                next_state_fail = self.snap_state((oc, sp - 1, 0.0, 0))
                return [
                    (start_prob, next_state_partial,
                    self.calculate_total_reward(next_state_partial, action, 8)),
                    (1 - start_prob, next_state_fail,
                    self.calculate_total_reward(next_state_fail, action, -5))
                ]
            elif h > 0.0:
                # Replacing a working satellite → penalize
                return [(1.0, self.snap_state((oc, sp, h, cov)),
         self.calculate_total_reward(self.snap_state((oc, sp, h, cov)), action, -15))]
            else:
                # No spares → no replacement possible
                return [(1.0, self.snap_state(state), -10)]

        # 4️⃣ ACTIVATE_BACKUP — Deploy extra spare to improve coverage
        if action == "ACTIVATE_BACKUP":
            if sp > 0 and oc < 2:
                success_prob = 0.85
                #next_state_success = (min(oc + 1, 2), sp - 1, h, 1)
                next_state_success = self.snap_state((min(oc + 1, 2), sp - 1, h, 1))
                next_state_fail = self.snap_state((oc, sp - 1, h, 0))

                return [
                    (success_prob, next_state_success,
                    self.calculate_total_reward(next_state_success, action, 5)),
                    (1 - success_prob, next_state_fail,
                    self.calculate_total_reward(next_state_fail, action, -3))
                ]
            else:
                # Unnecessary activation when already optimal
                return [(1.0, state, -8)]

        # Default fallback (should never hit)
        return [(1.0, state, -1)]

                    
    
    def reset_tracking(self):
        """Reset tracking dictionaries"""
        self.boost_history.clear()
        self.recovery_progress.clear()