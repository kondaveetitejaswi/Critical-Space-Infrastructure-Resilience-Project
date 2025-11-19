import numpy as np
from toy_mdp import ToyConstellationMDP
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')


class ADPSolver:
    """
    ADP Solver
    
    Key principles:
    - Use REWARDS (positive values to maximize)
    - All values should be positive
    - V(s) = max_a [reward(s,a) + γ·V(s')]
    - Matches DP's value iteration schema
    """
    
    def __init__(self, mdp: ToyConstellationMDP, gamma: float = 0.95, 
                 learning_rate: float = 0.05, max_iterations: int = 100):
        """
        Initialize ADP Solver with proper conventions.
        
        Note: Lower learning rate (0.05) to avoid oscillation
        with bootstrapping from initial V values.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.alpha = learning_rate
        self.max_iterations = max_iterations
        
        # Post-decision state values (should be positive, represent future reward)
        self.post_decision_values = defaultdict(float)
        
        # State values (should match or approximate DP values)
        self.V = np.zeros(mdp.nS)
        
        # Policy
        self.policy = np.zeros(mdp.nS, dtype=int)
        
        # Tracking
        self.value_history = []
        self.iteration_costs = []
        self.max_pds_value_history = []
        self.mean_pds_value_history = []
        
        print("\n✓ ADPSolver initialized")
        print(f"  - Learning rate (α): {self.alpha} (LOW for stability)")
        print(f"  - Discount factor (γ): {self.gamma}")
        print(f"  - Max iterations: {self.max_iterations}")
        print(f"  - Convention: MAXIMIZE REWARDS (positive values)")
    
    def get_post_decision_state(self, state: tuple, action: str) -> tuple:
        """
        Get post-decision state.
        Deterministic component of state after action but before randomness.
        """
        oc, sp, h, cov = state
        post_decision_states = {
            "NO_OP": "PDS_NO_OP",
            "REPLACE": "PDS_REPLACE",
            "ACTIVATE_BACKUP": "PDS_ACTIVATE_BACKUP",
            "BOOST": "PDS_BOOST"
        }
        
        if action == "NO_OP":
            pds = (oc, sp, h, cov, "PDS_NO_OP")
        elif action == "REPLACE":
            if sp > 0 and h == 0.0:
                pds = (min(oc + 1, 2), sp - 1, h, 1, "PDS_REPLACE")
            else:
                pds = (oc, sp, h, cov, "PDS_REPLACE")
        elif action == "ACTIVATE_BACKUP":
            if sp > 0 and oc < 2:
                pds = (min(oc + 1, 2), sp - 1, h, 1, "PDS_ACTIVATE_BACKUP")
            else:
                pds = (oc, sp, h, cov, "PDS_ACTIVATE_BACKUP")
        elif action == "BOOST":
            pds = (oc, sp, h, 1, "PDS_BOOST")
        else:
            pds = (oc, sp, h, cov, "PDS_NO_OP")
        
        return pds
    
    def compute_immediate_reward(self, state: tuple, action: str) -> float:
        """
        FIXED: Compute IMMEDIATE REWARD (not negated cost).
        """
        transitions = self.mdp.transition(state, action)
        
        # Expected immediate reward from taking this action
        avg_immediate_reward = sum(p * r for p, _, r in transitions)
        
        # Return as positive value (not negated!)
        return avg_immediate_reward
    
    def greedy_action_selection(self, state: tuple) -> int:
        """
        FIXED: Greedy action selection using maximization.
        
        v_t(S) = max_a [reward(S,a) + Ṽ(a)(S'_a)]
        """
        best_action_idx = 0
        best_value = -float('inf')  # Start with negative infinity for MAX
        
        for action_idx, action in enumerate(self.mdp.actions):
            immediate_reward = self.compute_immediate_reward(state, action)
            pds = self.get_post_decision_state(state, action)
            future_value = self.post_decision_values[pds]
            
            total_value = immediate_reward + future_value
            
            if total_value > best_value:  # MAXIMIZE, not minimize!
                best_value = total_value
                best_action_idx = action_idx
        
        return best_action_idx
    
    def forward_pass(self, start_state: tuple = None, max_steps: int = 50) -> dict:
        """
        Execute one forward pass (trajectory) through the MDP.
        
        Records trajectory for learning.
        """
        if start_state is None:
            start_state = (2, 1, 1.0, 1)  # Healthy constellation
        
        trajectory = {
            'states': [],
            'actions': [],
            'pds_states': [],
            'next_states': [],
            'rewards': [],
        }
        
        state = start_state
        total_reward = 0
        step = 0
        
        while step < max_steps and not self.mdp.is_terminal(state):
            # Greedy action selection
            action_idx = self.greedy_action_selection(state)
            action = self.mdp.actions[action_idx]
            
            # Compute post-decision state
            pds = self.get_post_decision_state(state, action)
            
            # Execute action and observe outcome
            transitions = self.mdp.transition(state, action)
            probs = np.array([t[0] for t in transitions])
            next_states = [t[1] for t in transitions]
            rewards = [t[2] for t in transitions]
            
            # Here for the outcome_idx we are getting the weighted
            expected_value = sum(p * (r + self.gamma * self.V[self.mdp.state_to_idx[next_state]])
                                 for p, next_state, r in transitions)
            most_probable_idx = np.argmax([p for p, _, _ in transitions])
            next_state = next_states[most_probable_idx]
            reward = rewards[most_probable_idx]
            
            # Record trajectory
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['pds_states'].append(pds)
            trajectory['next_states'].append(next_state)
            trajectory['rewards'].append(reward)
            
            total_reward += reward
            state = next_state
            step += 1
        
        trajectory['total_reward'] = total_reward
        trajectory['length'] = step
        
        return trajectory
    
    def backward_update(self, trajectory: dict):
        """
        FIXED: Proper temporal difference learning.
        
        Target value = immediate reward + γ·V(next_state)
        All values POSITIVE (matching reward convention).
        
        Update: Ṽ^n(S') = (1-α)·Ṽ^{n-1}(S') + α·target
        """
        T = len(trajectory['rewards'])
        
        # Work backwards through trajectory for better bootstrapping
        for t in range(T - 1, -1, -1):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            pds = trajectory['pds_states'][t]
            # compute expected target from model
            transitions = self.mdp.transition(state, action)
            expected_target = 0.0
            for (p, ns, r) in transitions:
                ns_idx = self.mdp.state_to_idx[ns]
                expected_target += p * (r + self.gamma * self.V[ns_idx])
            # TD update on PDS using expected target
            old_pds_value = self.post_decision_values[pds]
            new_pds_value = (1 - self.alpha) * old_pds_value + self.alpha * expected_target
            self.post_decision_values[pds] = new_pds_value
    
    def update_state_values(self):
        """
        FIXED: Update state values using maximization.
        
        V(s) = max_a [immediate_reward(s,a) + Ṽ(a)(s'_a)]
        """
        for i, state in enumerate(self.mdp.states):
            best_value = -float('inf')  # Start with -inf for MAX
            best_action = 0
            
            for action_idx, action in enumerate(self.mdp.actions):
                immediate_reward = self.compute_immediate_reward(state, action)
                pds = self.get_post_decision_state(state, action)
                future_value = self.post_decision_values[pds]
                
                value = immediate_reward + future_value
                
                if value > best_value:  # MAXIMIZE
                    best_value = value
                    best_action = action_idx
            
            self.V[i] = best_value
            self.policy[i] = best_action
    
    def check_convergence(self, iteration: int) -> bool:
        """
        Check if values are converging properly.
        
        Should see:
        - Positive values
        - Increasing/stable trends
        - Max values < 1000
        - Convergence across iterations
        """
        if not self.post_decision_values:
            return False
        
        max_abs_value = max(abs(v) for v in self.post_decision_values.values())
        mean_value = np.mean(list(self.post_decision_values.values()))
        
        # Check sanity
        if max_abs_value > 1e6:
            print(f"\n⚠️ Divergence warning: Max |PDS| = {max_abs_value:.2e}")
            return False
        
        # Positive values are good
        if mean_value < 0:
            print(f"\n⚠️ Warning: Mean PDS value is negative: {mean_value:.3f}")
            return False
        
        return True
    
    def value_iteration_adp(self, convergence_threshold=1e-6, patience=5, max_iterations=500):
        """
        Main ADP value iteration with CONVERGENCE as stopping criterion.
        
        Parameters:
        -----------
        convergence_threshold : float
            Change in mean PDS value below this triggers convergence counter
        patience : int
            Number of iterations with change < threshold before stopping
        max_iterations : int
            Maximum iterations (safety net, but convergence is primary criterion)
        """
        print("\n" + "="*70)
        print("ADP VALUE ITERATION - CONVERGENCE-BASED STOPPING")
        print("="*70)
        
        # Initialize values
        self.V = np.zeros(self.mdp.nS)
        self.post_decision_values = defaultdict(float)
        iterations_without_improvement = 0
        
        iteration = 0  # Track iteration separately (not in for loop)
        
        while iteration < max_iterations:  # Changed from for loop to while
            iteration_total_reward = 0
            
            # Multiple forward passes per iteration
            n_trajectories = 10
            
            for traj_id in range(n_trajectories):
                trajectory = self.forward_pass(max_steps=30)
                iteration_total_reward += trajectory['total_reward']
                
                # Learn from trajectory
                self.backward_update(trajectory)
            
            # Update state values
            self.update_state_values()
            
            # Record statistics
            avg_reward = iteration_total_reward / n_trajectories
            self.iteration_costs.append(avg_reward)
            
            if self.post_decision_values:
                max_pds = max(abs(v) for v in self.post_decision_values.values())
                mean_pds = np.mean(list(self.post_decision_values.values()))
            else:
                max_pds = 0
                mean_pds = 0
            
            self.max_pds_value_history.append(max_pds)
            self.mean_pds_value_history.append(mean_pds)
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                mean_v = np.mean(self.V)
                print(f"Iteration {iteration+1:3d} | Reward: {avg_reward:7.2f} | "
                    f"Max PDS: {max_pds:8.3f} | Mean PDS: {mean_pds:8.3f} | "
                    f"Mean V: {mean_v:8.3f}")
            
            # ===== CONVERGENCE CHECK (PRIMARY STOPPING CRITERION) =====
            if iteration >= 10:  # Need at least 10 iterations to check convergence
                # Compute change in mean PDS value
                recent_mean_pds = self.mean_pds_value_history[-1]
                previous_mean_pds = self.mean_pds_value_history[-2]
                
                pds_change = abs(recent_mean_pds - previous_mean_pds)
                
                # Check if change is below threshold
                if pds_change < convergence_threshold:
                    iterations_without_improvement += 1
                    
                    if (iteration + 1) % 10 == 0:
                        print(f"  → Convergence indicator: {iterations_without_improvement}/{patience} "
                            f"(change: {pds_change:.6f} < {convergence_threshold})")
                else:
                    iterations_without_improvement = 0
                    if (iteration + 1) % 10 == 0:
                        print(f"  → Improvement detected (change: {pds_change:.6f})")
                
                # STOP if converged for 'patience' iterations
                if iterations_without_improvement >= patience:
                    print(f"\n✓ CONVERGED after {iteration+1} iterations!")
                    print(f"  Mean PDS change: {pds_change:.6f} < {convergence_threshold}")
                    print(f"  Iterations without improvement: {iterations_without_improvement}")
                    break
            
            iteration += 1
        
        # ===== FINAL RESULTS =====
        print("\n" + "="*70)
        print(f"ADP COMPLETED")
        print("="*70)
        print(f"Total iterations: {iteration+1}")
        print(f"Stopping criterion: {'CONVERGENCE' if iterations_without_improvement >= patience else 'MAX ITERATIONS'}")
        print(f"Final mean state value: {np.mean(self.V):.3f}")
        print(f"State value range: [{np.min(self.V):.3f}, {np.max(self.V):.3f}]")
        print(f"Unique post-decision states: {len(self.post_decision_values)}")
        print("="*70 + "\n")
        
        return self.V, self.policy
    def plot_convergence(self):
        """Plot convergence metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(self.iteration_costs))
        
        # Plot 1: Trajectory reward
        ax1.plot(iterations, self.iteration_costs, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Iteration', fontweight='bold')
        ax1.set_ylabel('Average Trajectory Reward', fontweight='bold')
        ax1.set_title('Trajectory Reward Convergence', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Max PDS value
        ax2.plot(iterations, self.max_pds_value_history, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Iteration', fontweight='bold')
        ax2.set_ylabel('Max |PDS Value|', fontweight='bold')
        ax2.set_title('Post-Decision State Value Range', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean PDS value
        ax3.plot(iterations, self.mean_pds_value_history, 'g-', linewidth=2, marker='^', markersize=4)
        ax3.set_xlabel('Iteration', fontweight='bold')
        ax3.set_ylabel('Mean PDS Value', fontweight='bold')
        ax3.set_title('Average Post-Decision State Value', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: State value distribution
        ax4.hist(self.V, bins=15, alpha=0.7, edgecolor='black', color='steelblue')
        ax4.axvline(np.mean(self.V), color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(self.V):.2f}')
        ax4.set_xlabel('State Value', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Final State Value Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

    def create_faceted_heatmaps(self):
        fig, axes = plt.subplots(1, len(self.mdp.allowed_health), figsize = (16, 5))

        action_to_num = {action:idx for idx, action in enumerate(self.mdp.actions)}

        for h_idx, h in enumerate(self.mdp.allowed_health):
            ax = axes[h_idx] if len(self.mdp.allowed_health) > 1 else axes

            policy_matrix = np.full((len(self.mdp.op_counts), 2), -1, dtype = int)

            for state in self.mdp.states:
                oc, sp, state_h, cov = state
                if state_h == h and oc in self.mdp.op_counts:
                    state_idx = self.mdp.state_to_idx[state]
                    action_idx = self.policy[state_idx]
                    oc_idx = list(self.mdp.op_counts).index(oc)
                    policy_matrix[oc_idx, sp] = action_idx
            im = ax.imshow(policy_matrix, cmap = 'tab10', aspect = 'auto', vmin = 0, vmax = len(self.mdp.actions)-1)

            ax.set_xlabel('Spare Availability', fontsize = 11, fontweight = 'bold')
            ax.set_ylabel('Operational Count', fontsize = 11, fontweight = 'bold')
            ax.set_title(f'ADP Policy Heatmap (Health={h})', fontsize=12, fontweight='bold')

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Spare Sat: Not Available', 'Spare Sat: Available'])
            ax.set_yticks(range(len(self.mdp.op_counts)))
            ax.set_yticklabels(self.mdp.op_counts)

            for i in range(len(self.mdp.op_counts)):
                for j in range(2):
                    if policy_matrix[i, j] >=0:
                        action_name = self.mdp.actions[policy_matrix[i, j]]
                        short_name = action_name.replace('ACTIVATE_BACKUP', 'AB').replace('REPLACE','R').replace('BOOST', 'B').replace('NO_OP', 'NO')
                        ax.text(j, i, short_name, ha='center', va='center', fontsize=8, fontweight='bold', 
                                color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))
            
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(len(self.mdp.actions)))
        cbar.set_label('Action', rotation=270, labelpad=20, fontweight='bold')
        cbar.ax.set_yticklabels(self.mdp.actions, fontsize=9)

        plt.suptitle('ADP Policy Across Health Levels\n(Operational Count vs Spares)',
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        return fig

    def create_faceted_heatmaps_constellation_level(self):
        fig, axes = plt.subplots(1, len(self.mdp.allowed_health), figsize=(16, 5))
        
        # Handle single health level case
        if len(self.mdp.allowed_health) == 1:
            axes = [axes]

        action_to_num = {action: idx for idx, action in enumerate(self.mdp.actions)}

        for h_idx, h in enumerate(self.mdp.allowed_health):
            ax = axes[h_idx]
            
            # Create matrix with correct dimensions for spares
            n_spares = len(self.mdp.spares)
            n_ops = len(self.mdp.op_counts)
            policy_matrix = np.full((n_ops, n_spares), -1, dtype=int)

            for state in self.mdp.states:
                oc, sp, state_h, cov = state
                if state_h == h and oc in self.mdp.op_counts:
                    state_idx = self.mdp.state_to_idx[state]
                    action_idx = self.policy[state_idx]
                    oc_idx = list(self.mdp.op_counts).index(oc)
                    sp_idx = list(self.mdp.spares).index(sp)  # ← Use spares list
                    policy_matrix[oc_idx, sp_idx] = action_idx

            im = ax.imshow(policy_matrix, cmap='tab10', aspect='auto', 
                        vmin=0, vmax=len(self.mdp.actions)-1)

            ax.set_xlabel('Spare Availability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Operational Count', fontsize=11, fontweight='bold')
            ax.set_title(f'ADP Policy Heatmap (Health={h})', fontsize=12, fontweight='bold')

            # Set x-axis labels based on actual spare counts
            ax.set_xticks(range(n_spares))
            spare_labels = [f'Spares: {sp}' for sp in self.mdp.spares]
            ax.set_xticklabels(spare_labels, rotation=45, ha='right')
            
            ax.set_yticks(range(n_ops))
            ax.set_yticklabels(self.mdp.op_counts)

            # Add text annotations
            for i in range(n_ops):
                for j in range(n_spares):
                    if policy_matrix[i, j] >= 0:
                        action_name = self.mdp.actions[policy_matrix[i, j]]
                        short_name = (action_name
                                    .replace('ACTIVATE_BACKUP', 'AB')
                                    .replace('REPLACE', 'R')
                                    .replace('BOOST', 'B')
                                    .replace('NO_OP', 'NO'))
                        ax.text(j, i, short_name, ha='center', va='center', 
                            fontsize=8, fontweight='bold',
                            color='white', 
                            bbox=dict(boxstyle="round,pad=0.2", 
                                    facecolor="black", alpha=0.5))

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(len(self.mdp.actions)))
        cbar.set_label('Action', rotation=270, labelpad=20, fontweight='bold')
        cbar.ax.set_yticklabels(self.mdp.actions, fontsize=9)

        plt.suptitle('ADP Policy Across Health Levels\n(Operational Count vs Spares)',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        return fig

    def create_comprehensive_analysis(self, figure_size = 'medium', save_plots = False, plot_dpi = 100):
        size_options = {
            'small': (12, 10),
            'medium': (16, 12),
            'large': (20, 16),
            'xlarge': (24, 20)
        }

        figsize = size_options.get(figure_size, (16, 12))
        base_font_size = 8 if figsize[0] <= 12 else (10 if figsize[0] <= 16 else 12)

        plt.rcParams.update({
            'font.size': base_font_size,
            'axes.titlesize': base_font_size + 2,
            'axes.labelsize': base_font_size,
            'legend.fontsize': base_font_size - 1,
            'xtick.labelsize': base_font_size - 1,
            'ytick.labelsize': base_font_size - 1
        })
        
        fig = plt.figure(figsize=figsize)

        # 1. Policy Heatmap
        ax1 = plt.subplot(2, 2, 1)
        policy_matrix = np.full((len(self.mdp.op_counts), 2, len(self.mdp.allowed_health)), -1)
        
        for state in self.mdp.states:
            oc, sp, h, _ = state
            if oc in self.mdp.op_counts and h in self.mdp.allowed_health:
                idx = self.mdp.state_to_idx[state]
                action_idx = self.policy[idx]
                oc_idx = list(self.mdp.op_counts).index(oc)
                h_idx = list(self.mdp.allowed_health).index(h)
                policy_matrix[oc_idx, sp, h_idx] = action_idx
        
        # Show policy for healthy systems (h=1)
        healthy_policy = policy_matrix[:, :, 1]
        im1 = ax1.imshow(healthy_policy, cmap='tab10', aspect='auto')
        ax1.set_xlabel('Spare Status')
        ax1.set_ylabel('Operational Count')
        ax1.set_title('ADP Policy Heatmap (Healthy Systems)')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['No Spares', 'Has Spares'])
        ax1.set_yticks(range(len(self.mdp.op_counts)))
        ax1.set_yticklabels(self.mdp.op_counts)
        
        label_fontsize = max(6, base_font_size - 2)
        for i in range(len(self.mdp.op_counts)):
            for j in range(2):
                if healthy_policy[i, j] >= 0:
                    action_name = self.mdp.actions[int(healthy_policy[i, j])]
                    ax1.text(j, i, action_name, ha='center', va='center', 
                            fontsize=label_fontsize, fontweight='bold', color='white')
                    
        # 2. Policy Action Distribution
        ax3 = plt.subplot(2, 2, 2)
        action_counts = defaultdict(int)
        for state_idx, action_idx in enumerate(self.policy):
            action = self.mdp.actions[action_idx]
            action_counts[action] += 1
        
        actions_list = list(action_counts.keys())
        counts_list = list(action_counts.values())
        percentages = [count/len(self.policy)*100 for count in counts_list]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(actions_list)))
        bars = ax3.bar(actions_list, percentages, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Percentage of States (%)')
        ax3.set_title('ADP Policy Action Distribution')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
        # 3. State Value Heatmap
        ax4 = plt.subplot(2, 2, 3)
        value_matrix = np.zeros((len(self.mdp.op_counts), 2))
        for oc_idx, oc in enumerate(self.mdp.op_counts):
            for sp in [0, 1]:
                values_for_combo = []
                for h in self.mdp.allowed_health:
                    for t in [0, 1]:
                        state = (oc, sp, h, t)
                        if state in self.mdp.state_to_idx:
                            values_for_combo.append(self.V[self.mdp.state_to_idx[state]])
                if values_for_combo:
                    value_matrix[oc_idx, sp] = np.mean(values_for_combo)
        
        im4 = ax4.imshow(value_matrix, cmap='viridis', aspect='auto')
        ax4.set_xlabel('Spare Status')
        ax4.set_ylabel('Operational Count')
        ax4.set_title('ADP Average State Values Heatmap')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['No Spares', 'Has Spares'])
        ax4.set_yticks(range(len(self.mdp.op_counts)))
        ax4.set_yticklabels(self.mdp.op_counts)
        
        cbar = plt.colorbar(im4, ax=ax4)
        cbar.set_label('Average State Value')
        
        for i in range(len(self.mdp.op_counts)):
            for j in range(2):
                ax4.text(j, i, f'{value_matrix[i, j]:.1f}', 
                        ha='center', va='center', fontweight='bold', 
                        fontsize=label_fontsize)
    
        if figsize[0] <= 12:
            plt.tight_layout(pad = 1.0)
            suptitle_y = 0.95
            suptitle_size = base_font_size + 2
        else:
            plt.tight_layout(pad=2.0)
            suptitle_y = 0.325
            suptitle_size = base_font_size + 4
        
        plt.suptitle('ADP Solution Analysis', 
                    fontsize=suptitle_size, y=1, fontweight='bold')
        
        if save_plots:
            filename = f'ADP_Analysis_{figure_size}.png'
            plt.savefig(filename, dpi=plot_dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"\nPlots saved as: {filename}")
            plt.show()
            plt.close()
        else:
            plt.show()

    def create_comprehensive_analysis_constellation_level(self, figure_size='medium', save_plots=True, plot_dpi=100):
        """Create comprehensive analysis plots with dynamic dimensions"""
        
        size_options = {
            'small': (12, 10),
            'medium': (16, 12),
            'large': (20, 16),
            'xlarge': (24, 20)
        }

        figsize = size_options.get(figure_size, (16, 12))
        base_font_size = 8 if figsize[0] <= 12 else (10 if figsize[0] <= 16 else 12)

        plt.rcParams.update({
            'font.size': base_font_size,
            'axes.titlesize': base_font_size + 2,
            'axes.labelsize': base_font_size,
            'legend.fontsize': base_font_size - 1,
            'xtick.labelsize': base_font_size - 1,
            'ytick.labelsize': base_font_size - 1
        })
        
        fig = plt.figure(figsize=figsize)
        label_fontsize = max(6, base_font_size - 2)

        # Get actual dimensions from MDP
        n_ops = len(self.mdp.op_counts)
        n_spares = len(self.mdp.spares)
        n_health = len(self.mdp.allowed_health)

        # ============ SUBPLOT 1: Policy Heatmap ============
        ax1 = plt.subplot(2, 2, 1)
        
        # Create matrix with CORRECT dimensions
        policy_matrix = np.full((n_ops, n_spares, n_health), -1, dtype=int)
        
        for state in self.mdp.states:
            oc, sp, h, _ = state
            try:
                oc_idx = list(self.mdp.op_counts).index(oc)
                sp_idx = list(self.mdp.spares).index(sp)
                h_idx = list(self.mdp.allowed_health).index(h)
                
                state_idx = self.mdp.state_to_idx[state]
                action_idx = self.policy[state_idx]
                
                policy_matrix[oc_idx, sp_idx, h_idx] = action_idx
            except (ValueError, KeyError):
                continue
        
        # Show policy for healthiest systems
        healthy_policy = policy_matrix[:, :, -1]  # Last index = max health
        im1 = ax1.imshow(healthy_policy, cmap='tab10', aspect='auto', 
                        vmin=0, vmax=len(self.mdp.actions)-1)
        
        ax1.set_xlabel('Spare Count', fontweight='bold')
        ax1.set_ylabel('Operational Count', fontweight='bold')
        ax1.set_title('ADP Policy (Healthy Systems)', fontweight='bold')
        
        ax1.set_xticks(range(n_spares))
        ax1.set_xticklabels([str(sp) for sp in self.mdp.spares])
        ax1.set_yticks(range(n_ops))
        ax1.set_yticklabels([str(oc) for oc in self.mdp.op_counts])
        
        # Add action labels
        for i in range(n_ops):
            for j in range(n_spares):
                if healthy_policy[i, j] >= 0:
                    action_name = self.mdp.actions[int(healthy_policy[i, j])]
                    short = action_name[:2].upper()  # First 2 chars
                    ax1.text(j, i, short, ha='center', va='center', 
                            fontsize=label_fontsize, fontweight='bold', 
                            color='white', bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor='black', alpha=0.7))
        
        # ============ SUBPLOT 2: Action Distribution ============
        ax2 = plt.subplot(2, 2, 2)
        action_counts = defaultdict(int)
        
        for action_idx in self.policy:
            action = self.mdp.actions[action_idx]
            action_counts[action] += 1
        
        actions_list = list(action_counts.keys())
        counts_list = list(action_counts.values())
        percentages = [100 * c / len(self.policy) for c in counts_list]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(actions_list)))
        bars = ax2.bar(range(len(actions_list)), percentages, color=colors, 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.set_ylabel('Percentage of States (%)', fontweight='bold')
        ax2.set_title('Policy Action Distribution', fontweight='bold')
        ax2.set_xticks(range(len(actions_list)))
        ax2.set_xticklabels(actions_list, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=label_fontsize)
        
        # ============ SUBPLOT 3: State Value Heatmap ============
        ax3 = plt.subplot(2, 2, 3)
        value_matrix = np.zeros((n_ops, n_spares))
        
        for i, oc in enumerate(self.mdp.op_counts):
            for j, sp in enumerate(self.mdp.spares):
                values = []
                for h in self.mdp.allowed_health:
                    for cov in [0, 1]:
                        state = (oc, sp, h, cov)
                        if state in self.mdp.state_to_idx:
                            idx = self.mdp.state_to_idx[state]
                            values.append(self.V[idx])
                
                value_matrix[i, j] = np.mean(values) if values else 0
        
        im3 = ax3.imshow(value_matrix, cmap='viridis', aspect='auto')
        ax3.set_xlabel('Spare Count', fontweight='bold')
        ax3.set_ylabel('Operational Count', fontweight='bold')
        ax3.set_title('Average State Values', fontweight='bold')
        
        ax3.set_xticks(range(n_spares))
        ax3.set_xticklabels([str(sp) for sp in self.mdp.spares])
        ax3.set_yticks(range(n_ops))
        ax3.set_yticklabels([str(oc) for oc in self.mdp.op_counts])
        
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Value', fontweight='bold')
        
        # Add value labels
        for i in range(n_ops):
            for j in range(n_spares):
                ax3.text(j, i, f'{value_matrix[i, j]:.1f}', 
                        ha='center', va='center', fontweight='bold',
                        fontsize=label_fontsize, color='white' if value_matrix[i, j] < value_matrix.max()/2 else 'black')
        
        # ============ SUBPLOT 4: Convergence ============
        ax4 = plt.subplot(2, 2, 4)
        iterations = range(len(self.iteration_costs))
        
        ax4.plot(iterations, self.iteration_costs, 'b-', linewidth=2, 
                marker='o', markersize=4, label='Trajectory Reward')
        ax4.set_xlabel('Iteration', fontweight='bold')
        ax4.set_ylabel('Average Reward', fontweight='bold')
        ax4.set_title('Training Convergence', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.suptitle('ADP Solution Analysis', fontsize=base_font_size + 4, 
                    fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_plots:
            filename = f'8 constellation ADP_Analysis_{figure_size}.png'
            plt.savefig(filename, dpi=plot_dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"\n✓ Plots saved as: {filename}")
            plt.show()
            plt.close()
        else:
            plt.show()

def run_properly_fixed_adp():
    """Run the ADP"""
    print("\n" + "#"*70)
    print("# ADP IMPLEMENTATION")
    print("#"*70)
    
    mdp = ToyConstellationMDP()
    solver = ADPSolver(mdp, gamma=0.95, learning_rate=0.05, max_iterations=1000)
    
    V, policy = solver.value_iteration_adp()
    
    # plot faceted heatmaps
    print("\n Plotting Heatmaps")
    fig = solver.create_faceted_heatmaps()
    fig.savefig("ADP_Heatmaps.png",dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.show() 


    
    # Create comprehensive analysis
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS")
    print("="*70)
    solver.create_comprehensive_analysis(figure_size='medium', save_plots=True, plot_dpi=150)


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
    solver, V, policy = run_properly_fixed_adp()