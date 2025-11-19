# Essential libraries
import numpy as np
import random
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
import itertools
from matplotlib.patches import mpatches

# Custom imports
from ADP_SOLVER_1 import ADPSolver

@dataclass
class GNSS_constellation:
    def __init__(self):
        self.op_counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 
        self.spares = [0, 1, 2, 3, 4, 5, 6]
        self.allowed_health = [0.0, 0.5, 1.0]
        self.coverage_states = [0, 1]

        # Build state space