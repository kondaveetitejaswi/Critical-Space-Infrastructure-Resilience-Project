# Essential libraries
import numpy as np
import random
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Custom imports
from ADP_SOLVER_1 import ADPSolver

# Libraries for the GNSS satellite implementation
import os
import sys

@dataclass
class GNSS_constellation:
    def __init__(self):
        self.op_counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 
        self.spares = [0, 1, 2, 3, 4, 5, 6]
        self.allowed_health = [0.0, 0.5, 1.0]
        self.coverage_states = [0, 1]

        # Build state space
        self.states = []
        self.state_to_idx = {}
        self.idx_to_state = {}

        idx = 0
        for oc in self.op_counts:
            for sp in self.spares:
                for h in self.allowed_health:
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
        self.RECOVERY_STEPS = 2 # We are using 2 steps because we only have three health levels.

    def is_terminal(self, state: Tuple) -> bool:
        op_count, spares, _, _ = state
        return op_count == 0 and spares == 0
    
    def snap_health(self, h):
        return min(self.allowed_health, key=lambda x: abs(x - h))
    
    def snap_state(self, state):
        oc, sp, h, cov = state
        return (oc, sp, self.snap_health(h), cov)
    
    def get_operational_reward(self, state: Tuple) -> float:
        oc, sp, h, cov = state

        if oc == (len(self.op_counts)) and h == 1:
            return 10 #reward for being extremely healthy
        elif oc >= (len(self.op_counts) / 2) and h >= 0.5:
            return 8 # reward for half satellites working with health definitely more than 0.5
        elif oc >= (len(self.op_counts) / 2) and h < 0.5:
            return 6 # reward for the half satellites working with health less than 0.5
        elif oc <= (len(self.op_counts) / 2) and h < 0.5:
            return 4 # reward for most of the satellites not working and the health less than 0.5
        elif oc <= (len(self.op_counts) / 4) and h < 0.5:
            return -1 # reward for barely being functional
        elif oc ==0 and h == 0:
            return -10 # reward for being complete non functional 
        
        return 0.0
    def get_coverage_reward(self, state: Tuple) -> float:
        oc, sp, h, cov = state

        if oc == (len(self.op_counts)) and cov == 1:
            return 5
        elif oc >= (len(self.op_counts) / 2) and cov == 1:
            return 3
        elif oc <= (len(self.op_counts) / 2) and cov == 1:
            return 2
        elif oc == 0 and cov == 0:
            return 0
        else:
            return 0
        return 0
    
    def calculate_total_reward(self, state: Tuple, action: str, base_action_reward: float) -> float:
        """
        Calculation of the total reward including both the operational reward and the coverage reward
        """

        operational_reward = self.get_operational_reward(state)
        coverage_reward = self.get_coverage_reward(state)

        return (base_action_reward) + (operational_reward) + (coverage_reward)
    
    def transition(self, state: Tuple, action: str) -> List[Tuple[float, Tuple, float]]:
        """
        The following transition function is defined such that the this can be used for the bigger satellite constellatio as well
        """
        oc, sp, h, cov = state
        # Terminal state check
        if self.is_terminal(state):
            return [(1.0, state, 0)]
        
        if action == "NO_OP":
            if oc == (len(self.op_counts)) and sp >= 0 and cov == 1:
                total_reward = self.calculate_total_reward(state, action, 2) # the satellite is completely healthy and the base reward is 2
                return [(1.0, state, total_reward)]
            else:
                degrade_prob = 0.1 if h == 1.0 else (0.3 if h == 0.5 else 0.5)
                degraded_state = self.snap_state((max(oc -1, 0), sp, max(h - 0.5, 0.0), 0))
                stable_state = self.snap_state((oc, sp, h, cov))
                return [
                    ( 1- degrade_prob, stable_state,
                        self.calculate_total_reward(stable_state, action, 1)),
                        (degrade_prob, degraded_state,
                        self.calculate_total_reward(degraded_state, action, -2))
                ]
            
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
                return [(1.0, self.snap_state(state),
                            self.calculate_total_reward(self.snap_state(state), action, -10))]
            
        if action == "REPLACE":
            if sp > 0 and h ==0.0:
                start_prob = 0.9
                next_state_partial = self.snap_state((min(oc + 1, len(self.op_counts)), sp - 1, 0.5 , 1))
                next_state_fail = self.snap_state((oc, sp - 1, 0.0, 0))
                return [
                    (start_prob, next_state_partial,
                        self.calculate_total_reward(next_state_partial, action, 8)),
                        (1 - start_prob, next_state_fail,
                        self.calculate_total_reward(next_state_fail, action , -5))
                ]
            elif h > 0.0:
                return [(1.0, self.snap_state((oc, sp, h, cov)),
                            self.calculate_total_reward(self.snap_state((oc, sp, h, cov)), action, -15))]
            else:
                return [(1.0, self.snap_state(state), -10)]
            

        if action == "ACTIVATE_BACKUP":
            if sp >0 and oc < 2:
                success_prob = 0.85
                next_state_success = self.snap_state((min(oc + 1, len(self.op_counts)), sp - 1, h, 1))
                next_state_fail = self.snap_state((oc, sp -1, h, 0))
                return [
                    (success_prob, next_state_success,
                        self.calculate_total_reward(next_state_success, action, 5)),
                        (1 - success_prob, next_state_fail,
                        self.calculate_total_reward(next_state_fail, action, -3))
                ]
            else:
                return [(1.0, state, -8)]
            
        return [(1.0, state, -1)]
        

    def reset_tracking(self):
        self.boost_history.clear()
        self.recovery_progress.clear()

def implement_ADP_on_GNSS():
    """
    Run the ADP implementation on the GNSS constellation with 24 operational satellites and 6 spares
    """
    print("\n" + "#"*100)
    print("# ADP IMPLEMENTATION WITH GNSS CONSTELLATION : 24 OPERATIONAL SATELLITES AND 6 SPARES")
    print("#"*100)

    mdp = GNSS_constellation()
    solver = ADPSolver(mdp, gamma = 0.95, learning_rate= 0.05, max_iterations = 1000)

    V, policy = solver.value_iteration_adp()

    # Plot faceted Heatmaps
    print("\n PLotting Heatmaps")
    fig = solver.create_faceted_heatmaps_constellation_level()
    fig.savefig("GNSS constellation ADP Heatmap.png", bbox_inches = 'tight', 
                facecolor = 'white', edgecolor = 'none')
    plt.show()

    # Create comprehensive analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS")
    print("="*70)
    fig = solver.create_comprehensive_analysis_constellation_level(figure_size='medium', save_plots=False, plot_dpi=150)
    fig.savefig("GNSS constellation ADP Comprehensive Analysis.png", bbox_inches = 'tight', 
                facecolor = 'white', edgecolor = 'none')
    plt.show()

    # Print policy statistics
    action_counts = np.bincount(policy, minlength=len(mdp.actions))
    print("\nFinal Policy Distribution:")
    print("-" * 70)
    for i, action in enumerate(mdp.actions):
        count = action_counts[i]
        percentage = count / len(policy) * 100
        print(f"{action:<20}: {count:2d} states ({percentage:5.1f}%)")
    
    # Print value statistics
    print("\nValue Statistics:")
    print("-" * 70)
    print(f"Mean: {np.mean(V):.3f}")
    print(f"Std: {np.std(V):.3f}")
    print(f"Min: {np.min(V):.3f}")
    print(f"Max: {np.max(V):.3f}")

    return solver, V, policy

if __name__ == "__main__":
    solver, V, policy = implement_ADP_on_GNSS()