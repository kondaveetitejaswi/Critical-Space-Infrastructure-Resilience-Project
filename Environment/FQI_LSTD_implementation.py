import numpy as np
from typing import Tuple, Dict, List
from Environment.original_toy_mdp import ToyConstellationMDP
import time

######## Basis Function ########
class BasisFunctions:

    def __init__(self, mdp):
        self.mdp = mdp
        self.n_features = self._compute_feature_dimension()

    def _compute_feature_dimension(self):
        # Breakdown:
        # - Bias: 1
        # - Linear terms (oc, sp, h, cov): 4
        # - Quadratic terms (oc², sp², h², cov²): 4
        # - Interaction terms (oc*sp, oc*h, oc*cov, sp*h, h*cov): 5
        # - Action one-hot encoding: len(actions)
        # Total: 1 + 4 + 4 + 5 + len(actions) = 14 + len(actions)
        return 14 + len(self.mdp.actions)
    
    def extract_features(self, state: Tuple, action: str = None) -> np.ndarray:
        """
        Extract basis function features for state (and optionally action).
        For Q(s,a) approximation, always include both state and action components.
        
        IMPORTANT: action should ALWAYS be provided for function approximation.
        If action is None, we still create action features (all zeros for baseline).
        """
        oc, sp, h, cov = state

        # Normalize state components
        oc_norm = oc / 2.0
        sp_norm = sp / 1.0
        h_norm = h
        cov_norm = cov

        features = []

        # Bias term (1 feature)
        features.append(1.0)

        # Linear terms (4 features)
        features.extend([oc_norm, sp_norm, h_norm, cov_norm])

        # Quadratic terms (4 features)
        features.extend([
            oc_norm ** 2,
            sp_norm ** 2,
            h_norm ** 2,
            cov_norm ** 2
        ])

        # Interaction terms (5 features)
        features.extend([
            oc_norm * sp_norm,
            oc_norm * h_norm,
            oc_norm * cov_norm,
            sp_norm * h_norm,
            h_norm * cov_norm
        ])

        # One-hot encode action (ALWAYS include, even if None)
        # This should be 4 features (one per action)
        action_features = np.zeros(len(self.mdp.actions))
        if action is not None:
            action_idx = self.mdp.actions.index(action)
            action_features[action_idx] = 1.0
        features.extend(action_features.tolist())

        # Total: 1 + 4 + 4 + 5 + len(actions) = 14 + len(actions)
        result = np.array(features, dtype=np.float64)
        
        # Safety check
        if len(result) != self.n_features:
            raise ValueError(f"Feature dimension mismatch! Expected {self.n_features}, "
                           f"got {len(result)}. Action: {action}, State: {state}")
        
        return result

    
    def get_feature_matrix(self, states: List[Tuple], actions: List[str] = None) -> np.ndarray:
        '''
        Create feature matrix for a batch of states (and optionally actions).
        '''
        if actions is None:
            return np.array([self.extract_features(s) for s in states])
        else:
            return np.array([self.extract_features(s, a) for s, a in zip(states, actions)])


######## FITTED Q_ITERATION SOLVER ########

class FittedQIteration:
    """
    Fitted Q iteration (FQI) using known model transitions.

    Following the paper's approach:
    - Uses regression to approximate Q_function
    - Computes target values using expected Bellman Backup with known P(s'|s,a)
    - Iteratively updates Q_approximation until convergence
    """

    def __init__(self, mdp: ToyConstellationMDP, gamma: float = 0.95,
                 lambda_reg: float = 1e-3, max_iters: int = 100, tol: float = 1e-4):
        self.mdp = mdp
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.max_iter = max_iters
        self.tol = tol

        self.basis = BasisFunctions(mdp)

        # Initialize Q_function parameters (theta for each action)
        self.n_features = self.basis.n_features
        self.theta = {action: np.zeros(self.n_features) for action in mdp.actions}

        # Convergence tracking
        self.iteration_history = []
        self.theta_history = []
        self.convergence_deltas = []

    def compute_expected_target(self, state: Tuple, action: str) -> float:
        '''
        Compute expected target value using known transition model:
        y = E[r + γ * max_a' Q(s', a'; θ)]

        This is the key difference from model-free: we use P(s'|s,a) explicitly
        '''
        expected_value = 0.0
        transitions = self.mdp.transition(state, action)

        for prob, next_state, reward in transitions:
            # Compute max_a' Q(s', a')
            q_values_next = []
            for next_action in self.mdp.actions:
                # FIXED: Always include action in feature extraction
                phi_next = self.basis.extract_features(next_state, next_action)
                q_next = np.dot(self.theta[next_action], phi_next)
                q_values_next.append(q_next)

            max_q_next = max(q_values_next) 

            # Expected Bellman backup
            expected_value += prob * (reward + self.gamma * max_q_next)

        return expected_value
    
    def fit_iteration(self) -> float:
        '''
        Perform one iteration of FQI:
        1. Compute target values for all (s, a) pairs using current Q
        2. Solve ridge regression to update theta for each action
        3. Return maximum parameter change for convergence check
        '''
        theta_old = {a: self.theta[a].copy() for a in self.mdp.actions}

        # For each action, collect training data and fit
        max_delta = 0.0
        for action in self.mdp.actions:
            features_list = []
            targets_list = []

            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    continue

                # FIXED: Always include action when extracting features
                phi = self.basis.extract_features(state, action)
                features_list.append(phi)

                # Compute target using expected Bellman backup
                target = self.compute_expected_target(state, action)
                targets_list.append(target)

            Phi = np.array(features_list)
            y = np.array(targets_list)

            # Ridge regression to update theta
            A = Phi.T @ Phi + self.lambda_reg * np.eye(self.n_features)
            b = Phi.T @ y

            try:
                self.theta[action] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix for action {action}, using pseudoinverse")
                self.theta[action] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Track maximum parameter change across actions
            delta_action = np.max(np.abs(self.theta[action] - theta_old[action]))
            if delta_action > max_delta:
                max_delta = delta_action

        return max_delta

    def solve(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        '''
        Run FQI until convergence.
        Return: (theta parameters dict, derived policy)
        '''
        print("\n" + "=" * 60)
        print("Fitted Q-Iteration")
        print("=" * 60)
        print(f"Parameters: γ={self.gamma}, λ={self.lambda_reg}, tol={self.tol}")
        print(f"Feature dimension: {self.n_features}")
        print(f"State Space size: {self.mdp.nS}")
        print("-" * 60)

        for iteration in range(self.max_iter):
            delta = self.fit_iteration()
            self.convergence_deltas.append(delta)

            if iteration % 10 == 0:
                print(f"Iteration {iteration:4d}: ΔΘ = {delta:.6f}")

            # Check convergence
            if delta < self.tol:
                print(f"\nConverged after {iteration} iterations (ΔΘ = {delta:.6f})")
                break

        policy = self.extract_policy()

        print(f"FQI training complete.")
        print("=" * 60 + "\n")

        return self.theta, policy

    def extract_policy(self) -> np.ndarray:
        '''
        Extract greedy policy from learned Q_function:
        π(s) = argmax_a Q(s, a; Θ)
        '''
        policy = np.zeros(self.mdp.nS, dtype=int)

        for state_idx, s in enumerate(self.mdp.states):
            q_values = []
            for action in self.mdp.actions:
                # FIXED: Always include action
                phi = self.basis.extract_features(s, action)
                q = np.dot(self.theta[action], phi)
                q_values.append(q)

            policy[state_idx] = np.argmax(q_values)

        return policy

    def get_q_value(self, state: Tuple, action: str) -> float:
        '''Get Q-value for a specific state-action pair'''
        # FIXED: Always include action
        phi = self.basis.extract_features(state, action)
        return float(np.dot(self.theta[action], phi))

    def get_value_function(self) -> np.ndarray:
        '''
        Compute state-value function V[s] = max_a Q[s,a] for all states
        '''
        V = np.zeros(self.mdp.nS)

        for state_idx, state in enumerate(self.mdp.states):
            q_values = [self.get_q_value(state, a) for a in self.mdp.actions]
            V[state_idx] = max(q_values)

        return V    


########## LSTD-BASED APPROXIMATE POLICY ITERATION ##########

class LSTD_API:
    """
    Approximate Policy Iteration using Least Squares Temporal Difference (LSTD).

    Following the referenced paper's methodology:
    - Policy Evaluation: LSTD computes V^π(s; w) using known transitions
    - Policy Improvement: Greedy policy update using one-step lookahead
    - Alternates until policy converges
    
    NOTE: For LSTD, we use state-only features (no action encoding) since we're
    learning a state-value function V(s), not Q(s,a).
    """

    def __init__(self, mdp: ToyConstellationMDP, gamma: float = 0.95, lambda_reg: float = 1e-3,
                 max_iters: int = 1000, tol: float = 1e-4):
        self.mdp = mdp
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.max_iter = max_iters
        self.tol = tol

        self.basis = BasisFunctions(mdp)
        # For state-value function, we use state-only features
        # State features: 1 bias + 4 linear + 4 quadratic + 5 interactions = 14 features
        self.n_features = 14  # State features only, no action encoding

        # Initialize policy randomly
        self.policy = np.zeros(self.mdp.nS, dtype=int)

        # Initialize value function parameters
        self.w = np.zeros(self.n_features)

        # Convergence tracking
        self.iteration_history = []
        self.w_history = []
        self.convergence_deltas = []

    def _extract_state_features(self, state: Tuple) -> np.ndarray:
        """Extract state-only features (no action component)"""
        oc, sp, h, cov = state

        # Normalize state components
        oc_norm = oc / 2.0
        sp_norm = sp / 1.0
        h_norm = h
        cov_norm = cov

        features = []

        # Bias term
        features.append(1.0)

        # Linear terms
        features.extend([oc_norm, sp_norm, h_norm, cov_norm])

        # Quadratic terms
        features.extend([
            oc_norm ** 2,
            sp_norm ** 2,
            h_norm ** 2,
            cov_norm ** 2
        ])

        # Interaction terms
        features.extend([
            oc_norm * sp_norm,
            oc_norm * h_norm,
            oc_norm * cov_norm,
            sp_norm * h_norm,
            h_norm * cov_norm
        ])

        return np.array(features)

    def lstd_policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        '''
        LSTD Policy Evaluation using known model transitions.

        Solves: (Φ^T(Φ - γΦ'))w = Φ^T R
        where expectations are computed using P(s'|s,π(s))
        
        This is the core LSTD algorithm from the paper.
        '''
        A = np.zeros((self.n_features, self.n_features))
        b = np.zeros(self.n_features)

        for state_idx, state in enumerate(self.mdp.states):
            if self.mdp.is_terminal(state):
                continue

            # Get action from current policy
            action_idx = policy[state_idx]
            action = self.mdp.actions[action_idx]

            # Extract STATE-ONLY features (no action component)
            phi = self._extract_state_features(state)

            # Compute expected next-state features and reward
            expected_phi_next = np.zeros(self.n_features)
            expected_reward = 0.0

            transitions = self.mdp.transition(state, action)
            for prob, next_state, reward in transitions:
                # Use state-only features for next state
                phi_next = self._extract_state_features(next_state)
                expected_phi_next += prob * phi_next
                expected_reward += prob * reward

            # Update LSTD matrices
            A += np.outer(phi, phi - self.gamma * expected_phi_next)
            b += phi * expected_reward

        # Add regularization
        A += self.lambda_reg * np.eye(self.n_features)

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in LSTD, using pseudoinverse")
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        return w

    def policy_improvement(self, w: np.ndarray) -> np.ndarray:
        '''
        Greedy policy improvement using one-step lookahead with known transitions:
        π'(s) = argmax_a E[r + γV(s'; w)]
        '''
        new_policy = np.zeros(self.mdp.nS, dtype=int)

        for state_idx, state in enumerate(self.mdp.states):
            if self.mdp.is_terminal(state):
                new_policy[state_idx] = 0  # No action for terminal states
                continue
            
            action_values = []
            for action in self.mdp.actions:
                expected_value = 0.0
                transitions = self.mdp.transition(state, action)

                for prob, next_state, reward in transitions:
                    # Use state-only features for value estimation
                    phi_next = self._extract_state_features(next_state)
                    v_next = np.dot(w, phi_next)
                    expected_value += prob * (reward + self.gamma * v_next)

                action_values.append(expected_value)

            new_policy[state_idx] = np.argmax(action_values)

        return new_policy

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Run API with LSTD until policy convergence.
        Returns: (weights w, converged policy)
        '''
        print("\n" + "=" * 60)
        print("APPROXIMATE POLICY ITERATION (LSTD)")
        print("=" * 60)
        print(f"Parameters: γ={self.gamma}, λ={self.lambda_reg}, tol={self.tol}")
        print(f"Feature dimension: {self.n_features}")
        print(f"State Space size: {self.mdp.nS}")
        print("-" * 60)

        for iteration in range(self.max_iter):
            # Policy Evaluation using LSTD
            w_new = self.lstd_policy_evaluation(self.policy)

            # Policy Improvement
            policy_new = self.policy_improvement(w_new)

            # Check policy convergence
            policy_change = np.sum(policy_new != self.policy)
            self.convergence_deltas.append(policy_change)

            # Track weight change for diagnostics
            w_change = np.max(np.abs(w_new - self.w))

            if iteration % 5 == 0:
                print(f"Iteration {iteration:4d}: Policy changes = {policy_change}, Δw = {w_change:.6f}")
            
            self.w = w_new
            self.policy = policy_new

            if policy_change == 0:
                print(f"\nPolicy converged after {iteration} iterations.")
                break

        print(f"API-LSTD training complete.")
        print("=" * 60 + "\n")

        return self.w, self.policy
        
    def get_value_function(self) -> np.ndarray:
        '''
        Compute state-value function V(s;w) = φ(s)^T w for all states
        '''
        V = np.zeros(self.mdp.nS)
        for state_idx, state in enumerate(self.mdp.states):
            phi = self._extract_state_features(state)
            V[state_idx] = np.dot(self.w, phi)

        return V
        

# ======================================================
############# Evaluation Framework ###############
# ======================================================

class ADPEvaluator:
    '''
    Evaluation framework comparing ADP methods against DP baseline
    
    Implements metrics from experimental protocol:
    - RMSE between V-ADP and V*
    - Policy match ratio
    - Monte Carlo Rollout Evaluation
    '''

    def __init__(self, mdp: ToyConstellationMDP, V_optimal: np.ndarray,
                 policy_optimal: np.ndarray, gamma: float = 0.95):
        self.mdp = mdp
        self.V_optimal = V_optimal
        self.policy_optimal = policy_optimal
        self.gamma = gamma

    def compute_rmse(self, V_approx: np.ndarray) -> float:
        return np.sqrt(np.mean((V_approx - self.V_optimal) ** 2))
    
    def compute_policy_match_ratio(self, policy_approx: np.ndarray) -> float:
        matches = np.sum(policy_approx == self.policy_optimal)
        return matches / len(policy_approx)

    def monte_carlo_evaluation(self, policy: np.ndarray, n_episodes: int = 1000,
                                max_steps: int = 100, seed: int = 42) -> Tuple[float, float]:
        '''
        Evaluate policy using Monte Carlo rollouts.
        Returns: (mean return, standard error)
        '''
        np.random.seed(seed)
        returns = []

        # Use same initial states for fair comparison
        initial_states = [
            (2, 1, 1.0, 1),
            (1, 1, 0.5, 1),
            (1, 0, 0.0, 0)
        ]

        episodes_per_state = n_episodes // len(initial_states)

        for init_state in initial_states:
            for _ in range(episodes_per_state):
                total_return = self._run_episode(init_state, policy, max_steps)
                returns.append(total_return)

        mean_return = np.mean(returns)
        std_error = np.std(returns) / np.sqrt(len(returns))

        return mean_return, std_error
    
    def _run_episode(self, start_state: Tuple, policy: np.ndarray, max_steps: int) -> float:
        '''Run single episode and compute discounted return'''
        state = start_state
        total_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            if self.mdp.is_terminal(state):
                break
            
            # Get action from policy
            state_idx = self.mdp.state_to_idx[state]
            action_idx = policy[state_idx]
            action = self.mdp.actions[action_idx]

            # Sample transition
            transitions = self.mdp.transition(state, action)
            probs = [t[0] for t in transitions]
            next_states = [t[1] for t in transitions]
            rewards = [t[2] for t in transitions]

            idx = np.random.choice(len(transitions), p=probs)
            state = next_states[idx]
            reward = rewards[idx]

            total_return += discount * reward
            discount *= self.gamma

        return total_return
    
    def generate_comparison_report(self, results: Dict) -> None:
        '''
        Generate comprehensive comparison report
        results: Dict with keys 'DP', 'FQI', 'API-LSTD'
        '''
        print("\n" + "=" * 70)
        print("ADP Evaluation Report")
        print("=" * 70)
        print(f"\n{'Algorithm':<12} {'RMSE':<12} {'Policy Match':<15} {'MC Return':<15} {'Iterations':<12}")
        print("-" * 70)

        for algo_name, data in results.items():
            print(f"{algo_name:<12} {data['rmse']:<12.4f} {data['match_ratio']:<15.2%} "
                  f"{data['mc_return'][0]:<7.2f}±{data['mc_return'][1]:<6.2f} {data['iterations']:<12}")
        
        print("=" * 70 + "\n")