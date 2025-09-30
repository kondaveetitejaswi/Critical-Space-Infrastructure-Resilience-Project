import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from toy_mdp import ToyConstellationMDP
from exact_solver import ExactDPSolver
from collections import defaultdict
import pandas as pd

# Set plotting style (using matplotlib built-in styles)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def create_individual_plots(mdp, solver, V, policy, figure_size='medium', save_plots=False):
    """
    Create individual plots that can be displayed separately for better screen management
    """
    size_options = {
        'small': (8, 6),
        'medium': (10, 8), 
        'large': (12, 9),
        'xlarge': (14, 10)
    }
    
    individual_figsize = size_options.get(figure_size, (10, 8))
    
    plots = {
        'policy_heatmap': create_policy_heatmap,
        'reward_structure': create_reward_structure,
        'value_breakdown': create_value_breakdown,
        'convergence': create_convergence_plot,
        'performance': create_performance_plot,
        'value_distribution': create_value_distribution,
        'action_distribution': create_action_distribution
    }
    
    for plot_name, plot_func in plots.items():
        try:
            plt.figure(figsize=individual_figsize)
            plot_func(mdp, solver, V, policy)
            
            if save_plots:
                filename = f'{plot_name}_{figure_size}.png'
                plt.savefig(filename, dpi=100, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"Saved: {filename}")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error creating {plot_name}: {e}")

def create_policy_heatmap(mdp, solver, V, policy):
    """Create policy heatmap as individual plot"""
    policy_matrix = np.full((len(mdp.op_counts), 2, len(mdp.allowed_health)), -1)
    
    for state in mdp.states:
        oc, sp, h, _ = state
        if oc in mdp.op_counts and h in mdp.allowed_health:
            idx = mdp.state_to_idx[state]
            action_idx = policy[idx]
            oc_idx = list(mdp.op_counts).index(oc)
            h_idx = list(mdp.allowed_health).index(h)
            policy_matrix[oc_idx, sp, h_idx] = action_idx
    
    healthy_policy = policy_matrix[:, :, 1]
    plt.imshow(healthy_policy, cmap='tab10', aspect='auto')
    plt.xlabel('Spare Status')
    plt.ylabel('Operational Count')
    plt.title('Policy Heatmap (Healthy Systems)')
    plt.xticks([0, 1], ['No Spares', 'Has Spares'])
    plt.yticks(range(len(mdp.op_counts)), mdp.op_counts)
    
    for i in range(len(mdp.op_counts)):
        for j in range(2):
            if healthy_policy[i, j] >= 0:
                action_name = mdp.actions[int(healthy_policy[i, j])]
                plt.text(j, i, action_name, ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')

def create_reward_structure(mdp, solver, V, policy):
    """Create reward structure plot"""
    actions = mdp.actions
    rewards_by_action = {a: [] for a in actions}
    
    for state in mdp.states:
        for action in actions:
            transitions = mdp.transition(state, action)
            avg_reward = sum(p * r for p, _, r in transitions)
            rewards_by_action[action].append(avg_reward)
    
    data_for_box = [rewards_by_action[a] for a in actions]
    bp = plt.boxplot(data_for_box, labels=actions, patch_artist=True, showmeans=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xticks(rotation=45)
    plt.ylabel('Expected Reward')
    plt.title('Reward Distribution by Action')
    plt.grid(True, alpha=0.3)

def create_value_breakdown(mdp, solver, V, policy):
    """Create value breakdown plot"""
    colors = plt.cm.viridis(np.linspace(0, 1, len(mdp.allowed_health)))
    
    for i, h in enumerate(mdp.allowed_health):
        values_h = []
        for oc in mdp.op_counts:
            try:
                state_idx = mdp.state_to_idx.get((oc, 1, h, 1), None)
                if state_idx is not None:
                    values_h.append(V[state_idx])
                else:
                    values_h.append(0)
            except:
                values_h.append(0)
        
        plt.plot(mdp.op_counts, values_h, 
                label=f'Health={h}', marker='o', color=colors[i], linewidth=2)
    
    plt.xlabel('Operational Count')
    plt.ylabel('State Value')
    plt.title('Value Function vs Operational Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_convergence_plot(mdp, solver, V, policy):
    """Create convergence plot"""
    if hasattr(solver, 'delta_history') and solver.delta_history:
        plt.semilogy(solver.delta_history, 'r-', linewidth=2)
        plt.axhline(y=1e-6, color='k', linestyle='--', alpha=0.7, label='Threshold')
        plt.xlabel('Iteration')
        plt.ylabel('Delta (log scale)')
        plt.title('Value Iteration Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Convergence history not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Convergence Analysis')

def create_performance_plot(mdp, solver, V, policy):
    """Create performance comparison plot"""
    plt.text(0.5, 0.5, 'Performance validation requires\nMonte Carlo implementation', 
            ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Performance Validation')

def create_value_distribution(mdp, solver, V, policy):
    """Create value distribution plot"""
    plt.hist(V, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(np.mean(V), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(V):.2f}')
    plt.axvline(np.median(V), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(V):.2f}')
    plt.xlabel('State Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of State Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_action_distribution(mdp, solver, V, policy):
    """Create action distribution plot"""
    action_counts = defaultdict(int)
    for state_idx, action_idx in enumerate(policy):
        action = mdp.actions[action_idx]
        action_counts[action] += 1
    
    actions_list = list(action_counts.keys())
    percentages = [action_counts[a]/len(policy)*100 for a in actions_list]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(actions_list)))
    bars = plt.bar(actions_list, percentages, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Percentage of States (%)')
    plt.title('Policy Action Distribution')
    plt.xticks(rotation=45)
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
def get_expected_rational_action(state, mdp):
    """
    Determine expected rational action based on state and MDP structure.
    Uses domain knowledge about satellite constellation management.
    
    Parameters:
    -----------
    state : tuple
        State tuple (oc, sp, h, cov) where:
        - oc: operational count
        - sp: spare parts available
        - h: health level (0.0, 0.5, 1.0)
        - cov: coverage status (0 or 1)
    mdp : ToyConstellationMDP
        The MDP instance for reference
    
    Returns:
    --------
    str : Expected rational action name
    """
    oc, sp, h, cov = state  # Correct unpacking order
    
    # Priority 1: Critical failure - replace if possible
    if h == 0.0 and sp > 0:
        return "REPLACE"
    
    # Priority 2: Low operational capacity - activate backup if available
    if oc < 2 and sp > 0 and h >= 0.5:
        return "ACTIVATE_BACKUP"
    
    # Priority 3: Damaged satellites - boost recovery
    if 0.0 < h < 1.0:
        return "BOOST"
    
    # Priority 4: Healthy with good capacity - maintain status quo
    if h == 1.0 and oc >= 2:
        return "NO_OP"
    
    # Edge case: Healthy but low operational count and no spares
    if h == 1.0 and oc < 2 and sp == 0:
        return "NO_OP"  # Nothing we can do
    
    # Default: Try to improve health if damaged, otherwise no-op
    return "BOOST" if h < 1.0 else "NO_OP"


def plot_policy_vs_rationality(mdp, policy):
    """
    Plot the policy's chosen action vs. expected rational action.
    Uses only matplotlib (no seaborn).
    
    Parameters:
    -----------
    mdp : ToyConstellationMDP
        The MDP instance
    policy : np.ndarray
        Policy array mapping state indices to action indices
    
    Returns:
    --------
    pd.DataFrame : DataFrame with detailed comparison data
    """
    data = []
    
    for state in mdp.states:
        oc, sp, h, cov = state  # Correct order
        
        # Get policy action
        state_idx = mdp.state_to_idx[state]
        action_idx = policy[state_idx]
        policy_action = mdp.actions[action_idx]
        
        # Get expected rational action
        expected_action = get_expected_rational_action(state, mdp)
        
        # Check if they match
        match = (policy_action == expected_action)
        
        data.append({
            "Operational Count": oc,
            "Spares": sp,
            "Health": h,
            "Coverage": cov,
            "Policy Action": policy_action,
            "Expected Action": expected_action,
            "Match": match
        })
    
    df = pd.DataFrame(data)
    
    # Create pivot table: Health vs Operational Count (averaged over spares and coverage)
    pivot = df.pivot_table(
        index="Health", 
        columns="Operational Count",
        values="Match", 
        aggfunc=lambda x: sum(x) / len(x)  # Match ratio
    )
    
    # Convert to numpy array for plotting
    grid = pivot.values
    health_vals = pivot.index.tolist()
    oc_vals = pivot.columns.tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap
    cax = ax.imshow(grid, cmap="RdYlGn", origin="lower", aspect="auto", vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Action Match Ratio", rotation=270, labelpad=20)
    
    # Annotate cells with match ratio and symbol
    for i in range(len(health_vals)):
        for j in range(len(oc_vals)):
            ratio = grid[i, j]
            symbol = "✓" if ratio >= 0.75 else ("~" if ratio >= 0.5 else "✗")
            color = "white" if ratio < 0.5 else "black"
            ax.text(j, i, f"{ratio:.2f}\n{symbol}", 
                   ha="center", va="center", color=color, fontweight="bold")
    
    # Set labels and ticks
    ax.set_xticks(np.arange(len(oc_vals)))
    ax.set_yticks(np.arange(len(health_vals)))
    ax.set_xticklabels(oc_vals)
    ax.set_yticklabels(health_vals)
    ax.set_xlabel("Operational Count")
    ax.set_ylabel("Health Level")
    ax.set_title("Policy vs Expected Rational Action\n(Averaged over Spares and Coverage States)")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("POLICY RATIONALITY ANALYSIS")
    print("="*60)
    print(f"Overall Match Rate: {df['Match'].mean()*100:.1f}%")
    print(f"Total States: {len(df)}")
    print(f"Matching States: {df['Match'].sum()}")
    print(f"Mismatching States: {(~df['Match']).sum()}")
    
    # Show mismatches by action
    print("\nMismatches by Expected Action:")
    print("-" * 40)
    for expected in df['Expected Action'].unique():
        subset = df[df['Expected Action'] == expected]
        match_rate = subset['Match'].mean() * 100
        print(f"{expected:>20}: {match_rate:5.1f}% match rate")
    
    # Show specific mismatches
    mismatches = df[~df['Match']]
    if len(mismatches) > 0:
        print("\nDetailed Mismatches (first 10):")
        print("-" * 60)
        for idx, row in mismatches.head(10).iterrows():
            print(f"State (OC:{row['Operational Count']}, SP:{row['Spares']}, "
                  f"H:{row['Health']}, C:{row['Coverage']})")
            print(f"  Expected: {row['Expected Action']:>20} | "
                  f"Policy: {row['Policy Action']}")
    
    print("="*60 + "\n")
    
    return df
def analyze_dp_solution(figure_size='medium', save_plots=False, plot_dpi=100, individual_plots=False):
    """
    Analyze pure DP solution with enhanced visualizations
    
    Parameters:
    -----------
    figure_size : str, optional
        Size of the plots. Options: 'small' (12x10), 'medium' (16x12), 'large' (20x16), 'xlarge' (24x20)
    save_plots : bool, optional
        Whether to save plots to files instead of displaying
    plot_dpi : int, optional
        DPI for saved plots (default: 100)
    individual_plots : bool, optional
        Whether to create individual plots instead of one large subplot figure
    """
    print("\nAnalyzing Dynamic Programming Solution...")
    mdp = ToyConstellationMDP()
    solver = ExactDPSolver(mdp, gamma=0.95)
    
    # Run value iteration
    V, policy = solver.value_iteration(theta=1e-6)
    
    # Check if user wants individual plots for better screen management
    if individual_plots:
        create_individual_plots(mdp, solver, V, policy, figure_size, save_plots)
        return  # Skip the large subplot figure
    
    # Configure figure size based on screen size preference
    size_options = {
        'small': (12, 10),
        'medium': (16, 12), 
        'large': (20, 16),
        'xlarge': (24, 20)
    }
    
    figsize = size_options.get(figure_size, (16, 12))
    
    # Adjust font sizes based on figure size
    base_font_size = 8 if figsize[0] <= 12 else (10 if figsize[0] <= 16 else 12)
    plt.rcParams.update({
        'font.size': base_font_size,
        'axes.titlesize': base_font_size + 2,
        'axes.labelsize': base_font_size,
        'legend.fontsize': base_font_size - 1,
        'xtick.labelsize': base_font_size - 1,
        'ytick.labelsize': base_font_size - 1
    })
    
    # Create a comprehensive figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Enhanced Policy Heatmap
    ax1 = plt.subplot(3, 3, 1)
    policy_matrix = np.full((len(mdp.op_counts), 2, len(mdp.allowed_health)), -1)
    
    for state in mdp.states:
        oc, sp, h, _ = state
        if oc in mdp.op_counts and h in mdp.allowed_health:
            idx = mdp.state_to_idx[state]
            action_idx = policy[idx]
            oc_idx = list(mdp.op_counts).index(oc)
            h_idx = list(mdp.allowed_health).index(h)
            policy_matrix[oc_idx, sp, h_idx] = action_idx
    
    # Show policy for healthy systems (h=1)
    healthy_policy = policy_matrix[:, :, 1]  # Assuming h=1 is index 1
    im1 = ax1.imshow(healthy_policy, cmap='tab10', aspect='auto')
    ax1.set_xlabel('Spare Status')
    ax1.set_ylabel('Operational Count')
    ax1.set_title('Policy Heatmap (Healthy Systems)\nAction Selection Pattern')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Spares', 'Has Spares'])
    ax1.set_yticks(range(len(mdp.op_counts)))
    ax1.set_yticklabels(mdp.op_counts)
    
    # Add action labels on cells (adjust font size based on figure size)
    label_fontsize = max(6, base_font_size - 2)
    for i in range(len(mdp.op_counts)):
        for j in range(2):
            if healthy_policy[i, j] >= 0:
                action_name = mdp.actions[int(healthy_policy[i, j])]
                ax1.text(j, i, action_name, ha='center', va='center', 
                        fontsize=label_fontsize, fontweight='bold', color='white')
    
    # 2. Enhanced Reward Structure Analysis
    ax2 = plt.subplot(3, 3, 2)
    actions = mdp.actions
    rewards_by_action = {a: [] for a in actions}
    
    for state in mdp.states:
        for action in actions:
            transitions = mdp.transition(state, action)
            avg_reward = sum(p * r for p, _, r in transitions)
            rewards_by_action[action].append(avg_reward)
    
    # Create enhanced box plot for better distribution visualization
    data_for_box = [rewards_by_action[a] for a in actions]
    bp = ax2.boxplot(data_for_box, labels=actions, patch_artist=True, showmeans=True)
    
    # Color the boxes with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel('Expected Reward')
    ax2.set_title('Reward Distribution by Action\n(Box Plot with Means)')
    ax2.grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, action in enumerate(actions):
        mean_reward = np.mean(rewards_by_action[action])
        ax2.text(i+1, mean_reward, f'{mean_reward:.2f}', 
                ha='center', va='bottom', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    # # 3. Enhanced Value Function Breakdown
    # ax3 = plt.subplot(3, 3, 3)
    # colors = plt.cm.viridis(np.linspace(0, 1, len(mdp.allowed_health)))
    
    # for i, h in enumerate(mdp.allowed_health):
    #     values_h = []
    #     for oc in mdp.op_counts:
    #         # Get value for state with spares available
    #         try:
    #             state_idx = mdp.state_to_idx.get((oc, 1, h, 1), None)
    #             if state_idx is not None:
    #                 values_h.append(V[state_idx])
    #             else:
    #                 values_h.append(0)  # Default if state doesn't exist
    #         except:
    #             values_h.append(0)
        
    #     ax3.plot(mdp.op_counts, values_h, 
    #             label=f'Health={h}', marker='o', color=colors[i], linewidth=2)
    
    # ax3.set_xlabel('Operational Count')
    # ax3.set_ylabel('State Value')
    # ax3.set_title('Value Function vs Operational Count\n(With Spares Available)')
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)
    
    # 4. Enhanced Convergence Analysis
    ax4 = plt.subplot(3, 3, 3)
    if hasattr(solver, 'delta_history') and solver.delta_history:
        ax4.semilogy(solver.delta_history, 'r-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Delta (log scale)')
        ax4.set_title('Value Iteration Convergence')
        ax4.grid(True, alpha=0.3)
        
        # Add convergence threshold line
        ax4.axhline(y=1e-6, color='k', linestyle='--', alpha=0.7, label='Threshold')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Convergence history\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Convergence Analysis')
    
    # # 5. State Value Distribution
    # ax5 = plt.subplot(3, 3, 5)
    # n, bins, patches = ax5.hist(V, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    # ax5.axvline(np.mean(V), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(V):.2f}')
    # ax5.axvline(np.median(V), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(V):.2f}')
    # ax5.set_xlabel('State Value')
    # ax5.set_ylabel('Frequency')
    # ax5.set_title('Distribution of State Values')
    # ax5.legend()
    # ax5.grid(True, alpha=0.3)
    
    # 6. Policy Action Distribution
    ax6 = plt.subplot(3, 3, 4)
    action_counts = defaultdict(int)
    for state_idx, action_idx in enumerate(policy):
        action = mdp.actions[action_idx]
        action_counts[action] += 1
    
    actions_list = list(action_counts.keys())
    counts_list = list(action_counts.values())
    percentages = [count/len(policy)*100 for count in counts_list]
    
    # Use different colors for each bar
    colors = plt.cm.tab10(np.linspace(0, 1, len(actions_list)))
    bars = ax6.bar(actions_list, percentages, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Percentage of States (%)')
    ax6.set_title('Policy Action Distribution')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 7. Value Function Heatmap by State Dimensions
    ax7 = plt.subplot(3, 3, 5)
    # Create a 2D representation of values
    value_matrix = np.zeros((len(mdp.op_counts), 2))
    for oc_idx, oc in enumerate(mdp.op_counts):
        for sp in [0, 1]:
            # Average across health states for this (oc, sp) combination
            values_for_combo = []
            for h in mdp.allowed_health:
                for t in [0, 1]:  # time states
                    state = (oc, sp, h, t)
                    if state in mdp.state_to_idx:
                        values_for_combo.append(V[mdp.state_to_idx[state]])
            if values_for_combo:
                value_matrix[oc_idx, sp] = np.mean(values_for_combo)
    
    im7 = ax7.imshow(value_matrix, cmap='viridis', aspect='auto')
    ax7.set_xlabel('Spare Status')
    ax7.set_ylabel('Operational Count')
    ax7.set_title('Average State Values Heatmap')
    ax7.set_xticks([0, 1])
    ax7.set_xticklabels(['No Spares', 'Has Spares'])
    ax7.set_yticks(range(len(mdp.op_counts)))
    ax7.set_yticklabels(mdp.op_counts)
    
    # Add colorbar
    cbar = plt.colorbar(im7, ax=ax7)
    cbar.set_label('Average State Value')
    
    # Add value labels on cells (responsive font size)
    for i in range(len(mdp.op_counts)):
        for j in range(2):
            ax7.text(j, i, f'{value_matrix[i, j]:.1f}', 
                    ha='center', va='center', fontweight='bold', 
                    fontsize=label_fontsize)
    
    # # 8. Empirical Performance Validation with Monte Carlo Rollouts
    # ax8 = plt.subplot(3, 3, 6)
    # test_states = {
    #     (2,1,1.0,1): "Optimal",
    #     (1,1,0.0,0): "Degraded",
    #     (0,1,0.0,0): "Critical"
    # }

    # try:
    #     print("\nRunning Monte Carlo validation (this may take a moment)...")
    #     results = {}
    #     baseline_results = {}
        
    #     for state, desc in test_states.items():
    #         if state in mdp.state_to_idx:
    #             # Run optimal policy rollouts
    #             opt_mean, opt_std = solver.run_policy_rollout(
    #                 state, n_episodes=1000, max_steps=100
    #             )
                
    #             # Run random policy rollouts
    #             random_mean, random_std = solver.run_random_policy_rollout(
    #                 state, n_episodes=1000, max_steps=100
    #             )
                
    #             results[state] = (opt_mean, opt_std)
    #             baseline_results[state] = (random_mean, random_std)
                
    #             print(f"  {desc:10s}: Optimal={opt_mean:7.2f}±{opt_std:5.2f}, "
    #                 f"Random={random_mean:7.2f}±{random_std:5.2f}")
    #         else:
    #             results[state] = (0, 0)
    #             baseline_results[state] = (0, 0)
        
    #     x = np.arange(len(test_states))
    #     width = 0.35
        
    #     opt_means = [results[s][0] for s in test_states]
    #     opt_stds = [results[s][1] for s in test_states]
    #     base_means = [baseline_results[s][0] for s in test_states]
    #     base_stds = [baseline_results[s][1] for s in test_states]
        
    #     # Plot with empirical error bars
    #     ax8.bar(x - width/2, opt_means, width, label='Optimal Policy',
    #         yerr=opt_stds, capsize=5, alpha=0.8, color='lightcoral', 
    #         edgecolor='black', error_kw={'linewidth': 2})
    #     ax8.bar(x + width/2, base_means, width, label='Random Policy',
    #         yerr=base_stds, capsize=5, alpha=0.8, color='lightblue', 
    #         edgecolor='black', error_kw={'linewidth': 2})
        
    #     # Add value labels on bars
    #     for i, (opt_m, rand_m) in enumerate(zip(opt_means, base_means)):
    #         ax8.text(i - width/2, opt_m + opt_stds[i] + 2, f'{opt_m:.1f}',
    #                 ha='center', va='bottom', fontweight='bold', fontsize=label_fontsize-1)
    #         ax8.text(i + width/2, rand_m + base_stds[i] + 2, f'{rand_m:.1f}',
    #                 ha='center', va='bottom', fontweight='bold', fontsize=label_fontsize-1)
        
    #     ax8.set_xlabel('Initial State Condition')
    #     ax8.set_ylabel('Average Return (1000 episodes)')
    #     ax8.set_title('Empirical Policy Performance\n(Monte Carlo Validation)')
    #     ax8.set_xticks(x)
    #     ax8.set_xticklabels(test_states.values())
    #     ax8.legend(loc='upper left')
    #     ax8.grid(True, alpha=0.3, axis='y')
        
    #     # Add improvement percentage annotations
    #     for i, (state, desc) in enumerate(test_states.items()):
    #         if baseline_results[state][0] != 0:
    #             improvement = ((results[state][0] - baseline_results[state][0]) / 
    #                         abs(baseline_results[state][0])) * 100
    #             ax8.text(i, min(opt_means[i], base_means[i]) - 10,
    #                     f'+{improvement:.0f}%' if improvement > 0 else f'{improvement:.0f}%',
    #                     ha='center', va='top', fontsize=label_fontsize-1,
    #                     color='green' if improvement > 0 else 'red',
    #                     fontweight='bold')
        
    #     print("Monte Carlo validation complete.\n")


        # 3. Policy vs Rationality Check
        ax3 = plt.subplot(3, 3, 6)

        # Calculate match ratios for each (health, operational_count) combination
        oc_vals = list(mdp.op_counts)
        h_vals = list(mdp.allowed_health)
        grid = np.zeros((len(h_vals), len(oc_vals)))
        count_grid = np.zeros((len(h_vals), len(oc_vals)))  # Track number of states

        for state in mdp.states:
            oc, sp, h, cov = state
            if oc not in oc_vals or h not in h_vals:
                continue
            
            idx = mdp.state_to_idx[state]
            chosen_action = mdp.actions[policy[idx]]
            
            # Get expected rational action using the corrected function
            expected = get_expected_rational_action(state, mdp)
            
            match = int(chosen_action == expected)
            
            oc_idx = oc_vals.index(oc)
            h_idx = h_vals.index(h)
            
            # Accumulate matches (we'll average over spares and coverage)
            grid[h_idx, oc_idx] += match
            count_grid[h_idx, oc_idx] += 1

        # Calculate average match ratio
        grid = np.divide(grid, count_grid, where=count_grid > 0)

        # Plot heatmap
        im = ax3.imshow(grid, cmap="RdYlGn", origin="lower", aspect="auto", vmin=0, vmax=1)

        # Annotate with symbols
        for i in range(len(h_vals)):
            for j in range(len(oc_vals)):
                ratio = grid[i, j]
                symbol = "✓" if ratio >= 0.75 else ("~" if ratio >= 0.5 else "✗")
                color = "white" if ratio < 0.5 else "black"
                ax3.text(j, i, f"{ratio:.2f}\n{symbol}", 
                        ha="center", va="center", color=color, 
                        fontsize=label_fontsize, fontweight="bold")

        ax3.set_xticks(range(len(oc_vals)))
        ax3.set_yticks(range(len(h_vals)))
        ax3.set_xticklabels(oc_vals)
        ax3.set_yticklabels(h_vals)
        ax3.set_xlabel("Operational Count")
        ax3.set_ylabel("Health")
        ax3.set_title("Policy vs Expected Rational Action")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label("Match Ratio", rotation=270, labelpad=15)
        
    # except Exception as e:
    #     ax8.text(0.5, 0.5, f'Performance validation\nnot available\n({str(e)})', 
    #             ha='center', va='center', transform=ax8.transAxes)
    #     ax8.set_title('Performance Validation')
    
    # # 9. Bellman Residual Analysis
    # ax9 = plt.subplot(3, 3, 9)
    # try:
    #     residual = solver.compute_bellman_residual()
    #     ax9.text(0.5, 0.7, f'Bellman Residual:\n{residual:.2e}', 
    #             ha='center', va='center', transform=ax9.transAxes,
    #             fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
    #     # Add convergence quality assessment
    #     if residual < 1e-6:
    #         quality = "Excellent"
    #         color = "green"
    #     elif residual < 1e-4:
    #         quality = "Good"
    #         color = "orange"
    #     else:
    #         quality = "Poor"
    #         color = "red"
            
    #     ax9.text(0.5, 0.3, f'Solution Quality:\n{quality}', 
    #             ha='center', va='center', transform=ax9.transAxes,
    #             fontsize=12, color=color, fontweight='bold')
        
    # except Exception as e:
    #     ax9.text(0.5, 0.5, f'Residual analysis\nnot available\n({str(e)})', 
    #             ha='center', va='center', transform=ax9.transAxes)
    
    # # ax9.set_title('Solution Quality Assessment')
    # # ax9.axis('off')
    
    # Adjust layout and spacing based on figure size
    if figsize[0] <= 12:
        plt.tight_layout(pad=1.0)
        suptitle_y = 0.95
        suptitle_size = base_font_size + 2
    else:
        plt.tight_layout(pad=2.0)
        suptitle_y = 0.325
        suptitle_size = base_font_size + 4
    
    plt.suptitle('Dynamic Programming Solution Analysis', 
                fontsize=suptitle_size, y=suptitle_y, fontweight='bold')
    
    # Save or show plots
    if save_plots:
        filename = f'dp_analysis_{figure_size}.png'
        plt.savefig(filename, dpi=plot_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nPlots saved as: {filename}")
        plt.close()
    else:
        plt.show()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED ANALYSIS RESULTS")
    print("="*60)
    
    # Policy statistics
    print("\nPolicy Action Distribution:")
    print("-" * 30)
    total_states = len(policy)
    for action, count in action_counts.items():
        percentage = count/total_states*100
        print(f"{action:>15}: {count:>3} states ({percentage:5.1f}%)")
    
    # Value statistics
    print(f"\nValue Function Statistics:")
    print("-" * 30)
    print(f"{'Mean Value:':<15} {np.mean(V):>8.3f}")
    print(f"{'Std Dev:':<15} {np.std(V):>8.3f}")
    print(f"{'Min Value:':<15} {np.min(V):>8.3f}")
    print(f"{'Max Value:':<15} {np.max(V):>8.3f}")
    print(f"{'Value Range:':<15} {np.max(V) - np.min(V):>8.3f}")
    
    # Solution quality
    try:
        residual = solver.compute_bellman_residual()
        print(f"\nSolution Quality:")
        print("-" * 30)
        print(f"{'Bellman Residual:':<15} {residual:>8.2e}")
        
        # if residual < 1e-6:
        #     print(f"{'Assessment:':<15} Excellent convergence")
        # elif residual < 1e-4:
        #     print(f"{'Assessment:':<15} Good convergence")
        # else:
        #     print(f"{'Assessment:':<15} Poor convergence")
            
    except Exception as e:
        print(f"Could not compute Bellman residual: {e}")
    
    # Convergence information
    if hasattr(solver, 'delta_history') and solver.delta_history:
        print(f"\nConvergence Information:")
        print("-" * 30)
        print(f"{'Iterations:':<15} {len(solver.delta_history):>8}")
        print(f"{'Final Delta:':<15} {solver.delta_history[-1]:>8.2e}")
    
    print("\n" + "="*60)
    # Monte Carlo Validation Results
    if 'results' in locals() and results:
        print(f"\nMonte Carlo Validation Results:")
        print("-" * 60)
        print(f"{'State':<12} {'Optimal':<20} {'Random':<20} {'Improvement'}")
        print("-" * 60)
        for state, desc in test_states.items():
            if state in results:
                opt_m, opt_s = results[state]
                ran_m, ran_s = baseline_results[state]
                if ran_m != 0:
                    improvement = ((opt_m - ran_m) / abs(ran_m)) * 100
                    print(f"{desc:<12} {opt_m:6.2f} ± {opt_s:5.2f}  "
                        f"{ran_m:6.2f} ± {ran_s:5.2f}  {improvement:+6.1f}%")

if __name__ == "__main__":
    # Example usage with different screen sizes and options:
    
    # Option 1: For small screens - Individual plots (recommended for laptops)
    # analyze_dp_solution(figure_size='small', individual_plots=True)
    
    # Option 2: For medium screens - Compact subplot layout (default)
    analyze_dp_solution(figure_size='medium')
    
    # Option 3: For large screens - Full subplot layout
    # analyze_dp_solution(figure_size='large')
    
    # Option 4: For presentations or 4K monitors
    # analyze_dp_solution(figure_size='xlarge')
    
    # Option 5: Save plots instead of displaying (good for any screen size)
    # analyze_dp_solution(figure_size='medium', save_plots=True, plot_dpi=150)
    
    # Option 6: Individual plots saved to files (best for small screens)
    # analyze_dp_solution(figure_size='small', individual_plots=True, save_plots=True)