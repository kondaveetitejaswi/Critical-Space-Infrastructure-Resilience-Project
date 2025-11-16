import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
import itertools
from collections import defaultdict

# Importing from our repository
from toy_mdp import ToyConstellationMDP

@dataclass
class constellation_with_8_satellites:
    def __init__(self):
        self.op_counts = [0, 1, 2, 3, 4, 5, 6]
        self.spares = [0, 1, 2]
        self.allowed_health = [0.0, 0.5, 1.0]
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
        self.RECOVERY_STEPS = 2 # MNumber of steps required for the satellite to completely recover

        # The helper functions: is_terminal, snap_health and snap_state can be used from the toy_mdp python file, ToyConstellationMDP
        # The other functions need to be changed

        def operational_reward_based_on_constellation_state(self, state: Tuple) -> float:
            """ Calculate the base reward based on the current operational status of the constellation.
                We are trying to keep it general such that it can be used flexibly with constellation of any satellite size
            """
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
            
        def coverage_reward_based_on_coverage(self, state: Tuple) -> float:
            """
            The reward to be calcualted on the basis of the coverage capability
            """           
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
            
        def total_reward_calculation(self, state: Tuple, action: str, base_action_reward: float) -> float:
            """
            Calculation of the total reward including both the operational reward and the coverage reward
            """

            operational_reward = self.operational_reward_based_on_constellation_state(state)
            coverage_reward = coverage_reward_based_on_coverage(state)

            # 