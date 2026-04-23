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
# MDP
# =====================================================

class GNSSConstellationMDP:

    def __init__(self,
                 n_planes=6,
                 sats_per_plane=6,
                 max_spares=10):

        self.n_planes = n_planes
        self.capacity_per_plane = sats_per_plane
        self.max_spares = max_spares
        self.min_planes_required = 4

        # degradation
        self.health_decay = 0.02
        self.max_age = 50

        # Weibull PH
        self.weibull_k = 4.5
        self.weibull_eta = 60

        self.beta_health = 2.0
        self.beta_geom = 1.5
        self.beta_dop = 1.0

        # costs
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
        return int(np.argmin(planes))

    def strongest_plane(self, planes):
        return int(np.argmax(planes))

    def apply_action(self, state, action):

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

        return tuple(planes), spares

    # =================================================
    # FAILURE MODEL (PH-WEIBULL)
    # =================================================

    def compute_failure_probability(self, state, dop):

        age = state.age
        health = state.health

        if age == 0:
            baseline = 0
        else:
            baseline = (
                self.weibull_k / self.weibull_eta
            ) * (age / self.weibull_eta) ** (self.weibull_k - 1)

        geom_imbalance = np.std(state.sats_per_plane)

        linear = (
            self.beta_health * (1 - health)
            + self.beta_geom * geom_imbalance
            + self.beta_dop * dop / 10
        )

        hazard = baseline * np.exp(linear)

        return np.clip(1 - np.exp(-hazard), 0, 0.9)

    def apply_failures(self, planes, p_fail):
        return tuple(
            max(n - np.random.binomial(n, p_fail), 0)
            for n in planes
        )

    # =================================================
    # KPI MODEL (GEOMETRY AWARE)
    # =================================================

    def compute_kpis(self, planes, health):

        total = sum(planes)

        cno = 45 - 8*(1-health) + np.random.normal(0,1.5)

        p_vis = 1/(1+np.exp(-(cno-38)))

        visible = np.random.binomial(total, p_vis)

        planes_ok = sum(p >= 3 for p in planes)
        coverage = 1 if planes_ok >= self.min_planes_required else 0

        if visible < 4:
            dop = 10
        else:
            dop = np.clip(
                6/np.sqrt(visible)+np.random.normal(0,0.5),
                1,10
            )

        return coverage, dop, cno

    # =================================================
    # SINGLE SIMULATION STEP
    # =================================================

    def sample_next_state(self, state, action):

        # ACTION
        planes, spares = self.apply_action(state, action)

        # KPI estimate before failure
        coverage, dop, cno = self.compute_kpis(
            planes, state.health
        )

        # FAILURE
        p_fail = self.compute_failure_probability(state, dop)
        planes = self.apply_failures(planes, p_fail)

        # DEGRADATION
        health = max(
            state.health - self.health_decay +
            np.random.normal(0,0.01), 0
        )

        age = min(state.age+1, self.max_age)

        next_state = GNSSState(planes, spares, health, age)

        # COST
        cost = self.action_costs[action]

        if coverage == 0:
            cost += 200

        if dop > 3:
            cost += 5*(dop-3)

        if cno < 38:
            cost += 2*(38-cno)

        return next_state, cost

    # =================================================
    # MONTE CARLO ADP TRANSITION
    # =================================================

    def transition(self, state, action, n_samples=20):

        outcomes = {}

        for _ in range(n_samples):

            ns, cost = self.sample_next_state(state, action)
            reward = -cost

            outcomes.setdefault(ns,[0,reward])
            outcomes[ns][0]+=1

        transitions=[]

        for ns,(count,r) in outcomes.items():
            transitions.append((count/n_samples,ns,r))

        return transitions

    # =================================================
    # ADP CONTROL POLICY
    # =================================================

    def greedy_action(self,state):

        best=None
        best_cost=float("inf")

        for a in self.actions:

            transitions=self.transition(state,a)

            expected=0
            for p,ns,r in transitions:
                expected+=p*(-r)

            if expected<best_cost:
                best_cost=expected
                best=a

        return best

    # =================================================
    # SIMULATION
    # =================================================

    def run_simulation(self,policy,initial,steps=50):

        state=initial

        for t in range(steps):

            action="NO_OP" if policy=="NO_OP" \
                else self.greedy_action(state)

            transitions=self.transition(state,action)

            probs=[x[0] for x in transitions]
            idx=np.random.choice(len(transitions),p=probs)

            _,state,_=transitions[idx]

            cov,dop,cno=self.compute_kpis(
                state.sats_per_plane,
                state.health
            )

            print(
                f"t={t} | action={action} | "
                f"planes={state.sats_per_plane} | "
                f"health={state.health:.2f} | "
                f"cov={cov} | dop={dop:.2f}"
            )


# =====================================================
# RUN
# =====================================================

mdp=GNSSConstellationMDP()

initial=GNSSState(
    sats_per_plane=(6,6,6,6,6,6),
    spares=10,
    health=1.0,
    age=0
)

print("NO OP POLICY")
mdp.run_simulation("NO_OP",initial)

print("\nGREEDY ADP POLICY")
mdp.run_simulation("GREEDY",initial)