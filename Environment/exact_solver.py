import numpy as np
from toy_mdp import ToyConstellationMDP
import matplotlib.pyplot as plt
from collections import defaultdict

class ExactDPSolver:
    def __init__(self, mdp: ToyConstellationMDP, gamma: float = 0.95, max_iterations: int = 1000):
        self.mdp = mdp
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.V = np.zeros(mdp.nS)
        self.policy = np.zeros(mdp.nS, dtype=int)
    
    def value_iteration(self, theta: float = 1e-6):
        """Value iteration with convergence tracking"""
        iteration = 0
        self.value_history = []  # Initialize history trackers
        self.delta_history = []
        
        while True:
            delta = 0
            for i, s in enumerate(self.mdp.states):
                old_value = self.V[i]
                
                # Simple value update
                q_values = []
                for a in self.mdp.actions:
                    q = 0
                    for prob, next_state, reward in self.mdp.transition(s, a):
                        next_idx = self.mdp.state_to_idx[next_state]
                        q += prob * (reward + self.gamma * self.V[next_idx])
                    q_values.append(q)
                
                self.V[i] = max(q_values)
                self.policy[i] = np.argmax(q_values)
                delta = max(delta, abs(old_value - self.V[i]))
        
            # Track convergence metrics
            self.value_history.append(np.mean(self.V))
            self.delta_history.append(delta)
            
            if delta < theta:
                print(f"Converged after {iteration} iterations (δ={delta:.6f})")
                break
                
            iteration += 1
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, δ={delta:.6f}")
        
        return self.V, self.policy
    
    def print_policy(self):
        """Print human-readable policy"""
        for i, s in enumerate(self.mdp.states):
            print(f"State {s}: best action = {self.mdp.actions[self.policy[i]]}, value = {self.V[i]:.2f}")
    
    def plot_state_values(self):
        """Plot state values for analysis"""
        plt.figure(figsize=(12, 6))
        states = list(range(self.mdp.nS))
        plt.plot(states, self.V, 'b-', label='State Values')
        plt.xlabel('State Index')
        plt.ylabel('Value')
        plt.title('Value Function Across States')
        plt.grid(True)
        plt.legend()
        plt.savefig('state_values.png')
        plt.close()
    
    def analyze_policy_consistency(self):
        """Analyze action selection consistency"""
        policy_analysis = defaultdict(list)
        
        for i, s in enumerate(self.mdp.states):
            action = self.mdp.actions[self.policy[i]]
            oc, sp, h, cov = s
            state_desc = f"Op:{oc} Sp:{sp} H:{h} C:{cov}"
            policy_analysis[action].append(state_desc)
        
        print("\nPolicy Analysis:")
        print("================")
        for action, states in policy_analysis.items():
            print(f"\n{action}:")
            for state in states:
                print(f"  - {state}")
    
    def plot_convergence_metrics(self, value_history, delta_history):
        """Plot convergence metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Value convergence
        iterations = range(len(value_history))
        ax1.plot(iterations, value_history)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Value')
        ax1.set_title('Value Function Convergence')
        ax1.grid(True)
        
        # Delta convergence
        ax2.plot(iterations, delta_history)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Max Value Change (δ)')
        ax2.set_title('Convergence Rate')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('convergence_metrics.png')
        plt.close()
    
    def validate_policy(self):
        """Validate policy decisions for critical states"""
        test_states = [
            (2, 1, 0.0, 1),  # Just attacked
            (2, 1, 0.5, 1),  # Partially recovered
            (2, 1, 1.0, 1),  # Fully healthy
            (1, 1, 0.0, 0),  # Degraded with spares
            (2, 0, 0.0, 0),  # Multiple degraded, no spares
        ]
        
        print("\nPolicy Validation")
        print("="*50)
        
        for state in test_states:
            idx = self.mdp.state_to_idx[state]
            action = self.mdp.actions[self.policy[idx]]
            value = self.V[idx]
            
            print(f"\nState Analysis: Op:{state[0]} Sp:{state[1]} H:{state[2]:.1f} C:{state[3]}")
            print(f"Selected Action: {action}")
            print(f"State Value: {value:.2f}")
            
            # Show expected outcomes
            transitions = self.mdp.transition(state, action)
            for prob, next_state, reward in transitions:
                print(f"  {prob*100:.0f}% -> Health:{next_state[2]:.1f}, R={reward:.1f}")
    
    def analyze_optimal_policy(self):
        """Detailed analysis of the optimal policy"""
        # Group states by action
        action_states = defaultdict(list)
        for i, s in enumerate(self.mdp.states):
            action = self.mdp.actions[self.policy[i]]
            action_states[action].append((s, self.V[i]))
        
        print("\nOptimal Policy Analysis")
        print("="*50)
        
        # Analyze each action
        for action, states in action_states.items():
            print(f"\n{action} Action Selected for:")
            states.sort(key=lambda x: x[1], reverse=True)  # Sort by value
            for state, value in states:
                oc, sp, h, cov = state
                print(f"  State (Op:{oc} Sp:{sp} H:{h} C:{cov}) - Value: {value:.2f}")
                
                # Show transition probabilities
                transitions = self.mdp.transition(state, action)
                print("  Possible outcomes:")
                for prob, next_state, reward in transitions:
                    print(f"    {prob*100:.0f}% -> {next_state}, R={reward:.1f}")
    
    def plot_convergence_analysis(self, value_history, delta_history):
        """Plot comprehensive convergence analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot value convergence
        iterations = range(len(value_history))
        ax1.plot(iterations, value_history, 'b-', label='Average Value')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average State Value')
        ax1.set_title('Value Function Convergence')
        ax1.grid(True)
        ax1.legend()
        
        # Plot delta convergence (log scale)
        ax2.plot(iterations, delta_history, 'r-', label='Delta')
        ax2.set_yscale('log')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Max Value Change (δ)')
        ax2.set_title('Convergence Rate (Log Scale)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()  # Changed from plt.savefig()

    def plot_state_value_heatmap(self):
        """Create heatmap of state values using matplotlib"""
        # Reshape values for operational count vs spares
        op_counts = self.mdp.op_counts
        spares = self.mdp.spares
        
        healthy_covered = np.zeros((len(op_counts), len(spares)))
        degraded_uncovered = np.zeros((len(op_counts), len(spares)))
        
        for i, oc in enumerate(op_counts):
            for j, sp in enumerate(spares):
                # Get values for healthy, covered state
                state_healthy = (oc, sp, 1, 1)
                state_degraded = (oc, sp, 0, 0)
                
                idx_healthy = self.mdp.state_to_idx[state_healthy]
                idx_degraded = self.mdp.state_to_idx[state_degraded]
                
                healthy_covered[i, j] = self.V[idx_healthy]
                degraded_uncovered[i, j] = self.V[idx_degraded]
        
        # Create heatmaps using matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot healthy & covered states
        im1 = ax1.imshow(healthy_covered, cmap='viridis', aspect='auto')
        ax1.set_title('State Values (Healthy & Covered)')
        ax1.set_xlabel('Spare Satellites')
        ax1.set_ylabel('Operational Satellites')
        # Add value annotations
        for i in range(len(op_counts)):
            for j in range(len(spares)):
                text = ax1.text(j, i, f'{healthy_covered[i, j]:.1f}',
                              ha="center", va="center", color="w")
        # Set ticks
        ax1.set_xticks(range(len(spares)))
        ax1.set_yticks(range(len(op_counts)))
        ax1.set_xticklabels(spares)
        ax1.set_yticklabels(op_counts)
        plt.colorbar(im1, ax=ax1)
        
        # Plot degraded & uncovered states
        im2 = ax2.imshow(degraded_uncovered, cmap='viridis', aspect='auto')
        ax2.set_title('State Values (Degraded & Uncovered)')
        ax2.set_xlabel('Spare Satellites')
        ax2.set_ylabel('Operational Satellites')
        # Add value annotations
        for i in range(len(op_counts)):
            for j in range(len(spares)):
                text = ax2.text(j, i, f'{degraded_uncovered[i, j]:.1f}',
                              ha="center", va="center", color="w")
        # Set ticks
        ax2.set_xticks(range(len(spares)))
        ax2.set_yticks(range(len(op_counts)))
        ax2.set_xticklabels(spares)
        ax2.set_yticklabels(op_counts)
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()  # Changed from plt.savefig()
    
    def compute_bellman_residual(self):
        """Compute Bellman residual to verify convergence"""
        max_residual = 0
        for i, s in enumerate(self.mdp.states):
            # Current value
            v = self.V[i]
            
            # Compute TV (Bellman operator)
            max_q = float('-inf')
            for a in self.mdp.actions:
                q = 0
                for prob, next_state, reward in self.mdp.transition(s, a):
                    next_idx = self.mdp.state_to_idx[next_state]
                    q += prob * (reward + self.gamma * self.V[next_idx])
                max_q = max(max_q, q)
            
            # Update maximum residual
            residual = abs(max_q - v)
            max_residual = max(max_residual, residual)
        
        return max_residual
    
    def evaluate_policy(self):
        """Evaluate current policy independently"""
        V_pi = np.zeros(self.mdp.nS)
        theta = 1e-6
        
        while True:
            delta = 0
            for i, s in enumerate(self.mdp.states):
                v = V_pi[i]
                
                # Get action from current policy
                a = self.mdp.actions[self.policy[i]]
                
                # Compute value under policy
                new_v = 0
                for prob, next_state, reward in self.mdp.transition(s, a):
                    next_idx = self.mdp.state_to_idx[next_state]
                    new_v += prob * (reward + self.gamma * V_pi[next_idx])
                
                V_pi[i] = new_v
                delta = max(delta, abs(v - new_v))
        
            if delta < theta:
                break
    
        return V_pi

    def run_random_policy_rollout(self, start_state, n_episodes=1000, max_steps=100):
        """Run Monte Carlo rollouts with uniformly random action selection"""
        returns = []
        
        for _ in range(n_episodes):
            state = start_state
            total_reward = 0
            discount = 1.0
            
            for step in range(max_steps):
                if self.mdp.is_terminal(state):
                    break
                
                # Select action uniformly at random
                action = np.random.choice(self.mdp.actions)
                
                # Sample transition
                transitions = self.mdp.transition(state, action)
                probs = [t[0] for t in transitions]
                next_states = [t[1] for t in transitions]
                rewards = [t[2] for t in transitions]
                
                # Sample next state and reward
                idx = np.random.choice(len(transitions), p=probs)
                state = next_states[idx]
                reward = rewards[idx]
                
                total_reward += discount * reward
                discount *= self.gamma
            
            returns.append(total_reward)
        
        return np.mean(returns), np.std(returns)