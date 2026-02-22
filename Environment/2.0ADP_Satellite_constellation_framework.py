from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class ConstellationState:
    sats_per_plane: Tuple[int, ...]
    spares: int
    health: float
    age: int

    def total_operational(self):
        return sum(self.sats_per_plane)
    
    def discretize_health(self, h):
        return round(h, 1)  # 0.0, 0.1, ... , 1.0

    def discretize_age(self, age):
        return min(age, self.max_age)

        
class ConstellationMDP:

    def __init__(self, 
                 n_planes = 6, 
                 sats_per_plane = 6,
                 max_spares = 10
                 ):
        '''
        WE are considering 6 satellite planes with 6 satellites in each plane.
        The constellation has a maximum of 10 spare satellites that is higher than any existing GNSS constellation
        The total of 36 operational satellite and 10 spare satellites are modelled in the current constellation setting.
        '''
        
        #geometry
        self.n_planes = n_planes
        self.capacity_per_plane = sats_per_plane
        self.max_spares = max_spares

        # minimum planes with geometry
        self.min_planes_required = 4

        #aging & failure
        self.health_decay = 0.02
        self.p0 = 0.002
        self.alpha = 0.015
        self.beta = 0.4
        self.max_age = 50

        #costs
        self.action_costs = {
            "NO_OP": 0,
            "LAUNCH_1": 10.0,
            "ACTIVATE_SPARE": 2.0,
            "RETIRE_SAT": 1.0,
            "REBALANCE_PLANE": 3.0
        }

        self.service_penalty = 100.0

        self.actions = list(self.action_costs.keys())


    # Failure Model

    def failure_probability(self, health, age):
        hazard = self.p0 * (1 + self.alpha * age) * (1 + self.beta * (1 - health))
        return min(max(1 - np.exp(-hazard), 0.0), 0.25)
    
    def is_terminal(self, state):
        return sum(state.sats_per_plane) == 0

    
    def apply_failures(self, planes, p_fail):
        new_planes = []
        for n in planes:
            failures = np.random.binomial(n, p_fail)
            new_planes.append(max(n - failures, 0))
        return tuple(new_planes)
    

    def service_available(self, planes):
        planes_with_geometry = sum(1 for n in planes if n >= 3)
        return planes_with_geometry >= self.min_planes_required
    
    def weakest_plane(self, planes):
        return np.argmin(planes)
    
    def strongest_planes(self, planes):
        return int(np.argmax(planes))
    
    def apply_action(self, state:ConstellationState, action: str):

        planes = list(state.sats_per_plane)
        spares = state.spares

        if action =="ACTIVATE_SPARE" and spares >0:
            weakest = self.weakest_plane(planes)
            planes[weakest] +=1
            spares -=1

        elif action== "RETIRE_SAT":
            strongest = self.strongest_planes(planes)
            if planes[strongest]>0:
                planes[strongest]-=1
                spares +=1

        elif action == "LAUNCH_1":
            spares = min(spares +1, self.max_spares)

        elif action == "REBALANCE_PLANE":
            strongest = self.strongest_planes(planes)
            weakest = self.weakest_plane(planes)
            if planes[strongest]> planes[weakest]+1:
                planes[strongest]-=1
                planes[weakest]+=1

        return tuple(planes), spares
    

    # KPI models

    def evaluate_cno(self, health):
        nominal = 45.0
        loss = 10.0 * ( 1 - health)
        noise = np.random.normal(0, 1.5)
        return nominal - loss + noise
    
    def visibility_probability(self, cno):
        return 1 / (1 + np.exp(-(cno - 38)))
    
    def sample_visible(self, planes, p_vis):
        total = sum(planes)
        return np.random.binomial(total, p_vis)
    
    def evaluate_coverage(self, n_visible):
        return 1 if n_visible >= 4 else 0
    
    def evaluate_dop(self, n_visible):
        if n_visible <4:
            return 10.0
        
        return np.clip(6 / np.sqrt(n_visible) + np.random.normal(0, 0.5), 1, 10.0)
    
    def compute_kpis(self, state: ConstellationState):
        cno = self.evaluate_cno(state.health)
        p_vis = self.visibility_probability(cno)
        n_visible = self.sample_visible(state.sats_per_plane, p_vis)
        coverage = self.evaluate_coverage(n_visible)
        dop = self.evaluate_dop(n_visible)

        return coverage, dop, cno
    def discretize_health(self, h):
        return round(h, 1)  # 0.0, 0.1, ... , 1.0

    def discretize_age(self, age):
        return min(age, self.max_age)

    def transition(self, state, action, n_samples=20): #remove the actions first, just a simple transitoion without the action adn then observe.
        # natural degradation and the action are not working together. 

        outcomes = {}

        for _ in range(n_samples):
            next_state, cost = self.sample_next_state(state, action)

            reward = -cost   # ADP solver maximizes reward

            key = next_state
            if key not in outcomes:
                outcomes[key] = [0, reward]

            outcomes[key][0] += 1

        transitions = []
        for ns, (count, r) in outcomes.items():
            prob = count / n_samples
            transitions.append((prob, ns, r))

        return transitions
    def sample_next_state(self, state, action):

        planes, spares = self.apply_action(state, action)

        p_fail = self.failure_probability(state.health, state.age)
        planes = self.apply_failures(planes, p_fail)

        new_health = self.discretize_health(max(state.health - self.health_decay, 0.0))
        new_age = self.discretize_age(state.age + 1)

        next_state = ConstellationState(planes, spares, new_health, new_age)


        total_cost = 0.0

        # Operational action cost
        total_cost += self.action_costs[action]

        # Compute performance metrics (KPIs)
        coverage, dop, cno = self.compute_kpis(next_state)

        # Service outage penalty (catastrophic)
        if coverage == 0:
            total_cost += 200.0

        # Geometry degradation penalty
        # good GNSS DOP ≈ 1–3, bad > 6
        if dop > 3:
            total_cost += 5.0 * (dop - 3)

        # Signal quality penalty
        # threshold ~38 dB-Hz (typical tracking limit)
        if cno < 38:
            total_cost += 2.0 * (38 - cno)


        return next_state, total_cost



    
    # ADP policy

    def greedy_action(self, state: ConstellationState):

        best_action = None
        best_value = float('inf')

        for action in self.actions:

            transitions = self.transition(state, action)

            # expected cost = - expected reward
            expected_cost = 0.0
            for prob, next_state, reward in transitions:
                expected_cost += prob * (-reward)

            if expected_cost < best_value:
                best_value = expected_cost
                best_action = action

        return best_action

    
    def run_simulation(self, policy_name, mdp, initial_state, steps = 50):
        state = initial_state
        total_cost = 0.0
        service_failures = 0

        for t in range(steps):
            print("Its working till here")
            if policy_name == "NO_OP":
                action = "NO_OP"
            else:
                action = mdp.greedy_action(state)

            transitions = mdp.transition(state, action)

            # sample one outcome according to probabilities
            probs = [t[0] for t in transitions]
            idx = np.random.choice(len(transitions), p=probs)

            _, state, reward = transitions[idx]

            total_cost += -reward   # reward = -cost


            coverage, dop, cno = self.compute_kpis(state)

            if coverage == 0:
                service_failures += 1

            print(f"Year {t+1} | Action:{action:15} | Planes:{state.sats_per_plane} "
              f"| Health:{state.health:.2f} | Coverage:{coverage} | DOP:{dop:.2f} | C/N0:{cno:.2f}")

        print(f"\nTotal cost: {total_cost}")
        print(f"Service Failures: {service_failures}\n")

if __name__ == "__main__":
    mdp = ConstellationMDP()
    initial_state = ConstellationState(sats_per_plane=(6, 6, 6, 6, 6, 6), spares=10, health=1.0, age=0)

    print("=== Running NO_OP Policy ===")
    mdp.run_simulation("NO_OP", mdp, initial_state)

    print("=== Running Greedy Policy ===")
    mdp.run_simulation("GREEDY", mdp, initial_state)