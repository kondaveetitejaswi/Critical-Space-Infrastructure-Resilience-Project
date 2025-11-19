from toy_mdp import ToyConstellationMDP
from ADP_SOLVER_1 import ADPSolver
from constellation_with_8_satellites import constellation_with_8_satellites

# Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
import itertools
from collections import defaultdict
import matplotlib.patches as mpatches


def implement_adp_on_constellation():
    """
    Run the ADP implementation on the 8 satellite constellation with 6 operational satellites and 2 spares
    """
    print("\n" + "#"*70)
    print("# ADP IMPLEMENTATION WITH 8 SATELLITE CONSTELLATION : 6 OPERATIONAL SATELLITES AND 2 SPARES")
    print("#"*70)

    mdp = constellation_with_8_satellites()
    solver = ADPSolver(mdp, gamma = 0.95, learning_rate= 0.05, max_iterations = 1000)

    V, policy = solver.value_iteration_adp()

    # Plot faceted Heatmaps
    print("\n PLotting Heatmaps")
    fig = solver.create_faceted_heatmaps_constellation_level()
    fig.savefig("8 constellation ADP Heatmap.png", bbox_inches = 'tight', 
                facecolor = 'white', edgecolor = 'none')
    plt.show()

    # Create comprehensive analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS")
    print("="*70)
    solver.create_comprehensive_analysis_constellation_level(figure_size='medium', save_plots=True, plot_dpi=150)

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
    solver, V, policy = implement_adp_on_constellation()

