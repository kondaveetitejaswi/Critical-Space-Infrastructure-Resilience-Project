from dataclasses import dataclass
from typing import Tuple
import numpy as np

# =====================================================
# STATE
# =====================================================

@dataclass(frozen=True)
class GNSSState:
    sats_per_plane: Tuple[int, ...]
    spares: int
    health: float
    age: int

    def total_operational(self):
        return sum(self.sats_per_plane)


# =====================================================
# MDP WITH ADP CONTROL
# =====================================================

class GNSSConstellationMDP:

    def __init__(self,
                 n_planes=6,
                 sats_per_plane=6,
                 max_spares=10):

        # Geometry
        self.n_planes = n_planes
        self.capacity_per_plane = sats_per_plane
        self.max_sats = n_planes * sats_per_plane
        self.max_spares = max_spares
        self.min_planes_required = 4

        # Aging & Failure
        self.health_decay = 0.02
        self.max_age = 50

        # Weibull PH (sophisticated failure model)
        self.weibull_k = 4.5        # shape (wear-out regime)
        self.weibull_eta = 60.0     # scale (design life in steps)

        # Covariate coefficients for hazard
        self.beta_health = 2.0
        self.beta_geom = 1.5
        self.beta_dop = 1.0

        # Action costs
        self.action_costs = {
            "NO_OP": 0,
            "LAUNCH_1": 10,
            "ACTIVATE_SPARE": 2,
            "RETIRE_SAT": 1,
            "REBALANCE_PLANE": 3
        }

        self.actions = list(self.action_costs.keys())

    # =================================================
    # ACTION MODEL
    # =================================================

    def weakest_plane(self, planes):
        """Return index of plane with fewest satellites"""
        return int(np.argmin(planes))

    def strongest_plane(self, planes):
        """Return index of plane with most satellites"""
        return int(np.argmax(planes))

    def apply_action(self, state, action):
        """Apply deterministic action, return new planes config and spares"""
        
        planes = list(state.sats_per_plane)
        spares = state.spares

        if action == "ACTIVATE_SPARE" and spares > 0:
            w = self.weakest_plane(planes)
            if planes[w] < self.capacity_per_plane:
                planes[w] += 1
                spares -= 1

        elif action == "REBALANCE_PLANE":
            s = self.strongest_plane(planes)
            w = self.weakest_plane(planes)
            if planes[s] > planes[w]:
                planes[s] -= 1
                planes[w] += 1

        elif action == "RETIRE_SAT":
            s = self.strongest_plane(planes)
            if planes[s] > 0:
                planes[s] -= 1
                spares += 1

        elif action == "LAUNCH_1":
            spares = min(spares + 1, self.max_spares)

        # NO_OP does nothing

        return tuple(planes), spares

    # =================================================
    # KPI MODEL (Geometry & Health Aware)
    # =================================================

    def compute_kpis(self, planes, health):
        """
        Compute coverage, DOP, and CNO from state.
        Returns: (coverage, dop, cno)
        - coverage: 1 if service available (≥4 planes with ≥3 sats each), 0 otherwise
        - dop: Dilution of Precision (1-10)
        - cno: Carrier-to-Noise ratio in dB
        """
        
        total = sum(planes)

        # CNO model: degrades with health, affected by shadowing
        cno = 45 - 8*(1 - health) + np.random.normal(0, 1.5)

        # Visibility probability (logistic model based on CNO threshold)
        p_vis = 1 / (1 + np.exp(-(cno - 38)))

        # Sample visible satellites
        visible = np.random.binomial(total, p_vis)

        # Coverage: at least 4 planes with ≥3 satellites each
        planes_ok = sum(1 for p in planes if p >= 3)
        coverage = 1 if planes_ok >= self.min_planes_required else 0

        # DOP: depends on number of visible satellites
        if visible < 4:
            dop = 10.0
        else:
            dop = np.clip(
                6.0 / np.sqrt(visible) + np.random.normal(0, 0.5),
                1.0, 10.0
            )

        return coverage, dop, cno

    # =================================================
    # FAILURE MODEL (Proportional Hazard - Weibull)
    # =================================================

    def compute_failure_probability(self, state, dop):
        """
        Compute per-satellite failure probability using Cox proportional hazard model.
        Baseline: Weibull hazard with age.
        Covariates: health degradation, geometric imbalance, DOP stress.
        """
        
        age = state.age
        health = state.health

        # Weibull baseline hazard
        if age == 0:
            baseline = 0.0
        else:
            baseline = (
                self.weibull_k / self.weibull_eta
            ) * (age / self.weibull_eta) ** (self.weibull_k - 1)

        # Geometric imbalance (std dev of satellites across planes)
        geom_imbalance = np.std(state.sats_per_plane) if len(state.sats_per_plane) > 0 else 0.0

        # Linear predictor from covariates
        linear = (
            self.beta_health * (1.0 - health) +
            self.beta_geom * geom_imbalance +
            self.beta_dop * (dop / 10.0)
        )

        # Proportional hazard
        hazard = baseline * np.exp(linear)

        # Convert hazard to failure probability
        p_fail = np.clip(1.0 - np.exp(-hazard), 0.0, 0.9)

        return p_fail

    def apply_failures(self, planes, p_fail):
        """Apply stochastic failures to constellation"""
        return tuple(
            max(n - np.random.binomial(n, p_fail), 0)
            for n in planes
        )

    # =================================================
    # SINGLE SIMULATION STEP
    # =================================================

    def sample_next_state(self, state, action):
        """
        Execute one step: action → KPI → failures → degradation → next_state + cost
        
        Returns: (next_state, cost)
        - cost includes action cost + service penalties + quality penalties
        """
        
        # Step 1: Apply action (deterministic)
        planes, spares = self.apply_action(state, action)

        # Step 2: Compute KPIs before failure
        coverage, dop, cno = self.compute_kpis(planes, state.health)

        # Step 3: Compute and apply failures (stochastic)
        p_fail = self.compute_failure_probability(state, dop)
        planes = self.apply_failures(planes, p_fail)

        # Step 4: Health degradation (stochastic)
        health = max(
            state.health - self.health_decay + np.random.normal(0, 0.01),
            0.0
        )

        # Step 5: Age increment
        age = min(state.age + 1, self.max_age)

        # Create next state
        next_state = GNSSState(planes, spares, health, age)

        # Step 6: Compute cost
        cost = self.action_costs[action]

        # Service penalty: huge cost if coverage lost
        if coverage == 0:
            cost += 200.0

        # Quality penalties
        if dop > 3:
            cost += 5.0 * (dop - 3)

        if cno < 38:
            cost += 2.0 * (38 - cno)

        return next_state, cost

    # =================================================
    # MONTE CARLO ADP TRANSITION
    # =================================================

    def transition(self, state, action, n_samples=20):
        """
        Monte Carlo aggregation of next states.
        
        Runs sample_next_state n_samples times, aggregates outcomes.
        Returns: list of (probability, next_state, reward) tuples
        
        Args:
            state: current GNSSState
            action: action string
            n_samples: number of Monte Carlo samples (default 20)
            
        Returns:
            List of (prob, next_state, reward) where reward = -cost
        """
        
        outcomes = {}

        for _ in range(n_samples):
            ns, cost = self.sample_next_state(state, action)
            reward = -cost  # Negative cost = reward

            # Aggregate identical outcomes
            if ns not in outcomes:
                outcomes[ns] = [0, reward]
            outcomes[ns][0] += 1

        # Convert counts to probabilities
        transitions = []
        for ns, (count, r) in outcomes.items():
            prob = count / n_samples
            transitions.append((prob, ns, r))

        return transitions

    # =================================================
    # ADP CONTROL POLICY
    # =================================================

    def greedy_action(self, state):
        """
        Greedy policy: evaluate all actions, pick one with lowest expected cost.
        
        For each action:
        1. Get Monte Carlo distribution of next states
        2. Compute expected cost E[cost] = -E[reward]
        3. Pick action with minimum expected cost
        
        Args:
            state: current GNSSState
            
        Returns:
            action string with lowest expected cost
        """
        
        best_action = None
        best_cost = float("inf")

        for action in self.actions:
            transitions = self.transition(state, action)

            # Expected cost
            expected_cost = 0.0
            for prob, next_state, reward in transitions:
                expected_cost += prob * (-reward)

            if expected_cost < best_cost:
                best_cost = expected_cost
                best_action = action

        return best_action

    # =================================================
    # SIMULATION
    # =================================================

    def run_simulation(self, policy, initial, steps=50, verbose=True):
        """
        Simulate constellation control over multiple steps.
        
        Args:
            policy: "NO_OP" (do nothing) or "GREEDY" (ADP greedy)
            initial: initial GNSSState
            steps: number of simulation steps
            verbose: print state at each step
            
        Yields/Prints trajectory of (state, action, kpis)
        """
        
        state = initial
        trajectory = []

        for t in range(steps):

            # Select action
            if policy == "NO_OP":
                action = "NO_OP"
            elif policy == "GREEDY":
                action = self.greedy_action(state)
            else:
                raise ValueError(f"Unknown policy: {policy}")

            # Execute step
            transitions = self.transition(state, action)

            # Sample next state from distribution
            probs = [x[0] for x in transitions]
            idx = np.random.choice(len(transitions), p=probs)
            _, state, reward = transitions[idx]

            # Compute KPIs for logging
            coverage, dop, cno = self.compute_kpis(
                state.sats_per_plane,
                state.health
            )

            if verbose:
                print(
                    f"t={t:2d} | action={action:15s} | "
                    f"planes={state.sats_per_plane} | "
                    f"health={state.health:.2f} | "
                    f"cov={coverage} | dop={dop:.2f} | cno={cno:.1f}"
                )

            trajectory.append({
                'time': t,
                'action': action,
                'state': state,
                'coverage': coverage,
                'dop': dop,
                'cno': cno,
                'reward': reward
            })

        return trajectory


# =====================================================
# RUN SIMULATION
# =====================================================

if __name__ == "__main__":

    mdp = GNSSConstellationMDP()

    # Healthy initial state: full constellation
    initial = GNSSState(
        sats_per_plane=(6, 6, 6, 6, 6, 6),
        spares=10,
        health=1.0,
        age=0
    )

    print("=" * 100)
    print("NO-OP POLICY (Passive, no maintenance)")
    print("=" * 100)
    mdp.run_simulation("NO_OP", initial, steps=20)

    print("\n" + "=" * 100)
    print("GREEDY ADP POLICY (Active maintenance)")
    print("=" * 100)
    mdp.run_simulation("GREEDY", initial, steps=20)