from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# GNSS State definition

@dataclass(frozen=True)
class GNSSState:
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



# GNSS Constellation MDP

class GNSSConstellationMDP:

    def __init__(self,
        n_planes = 6,
        sats_per_plane = 6,
        max_spares = 10):

        # geometry
        self.n_planes = n_planes
        self.capacity_per_plane = sats_per_plane
        self.max_sats = n_planes * sats_per_plane
        self.max_spares = max_spares
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


        # --- Weibull PH parameters ---
        self.weibull_k = 4.5        # shape (wear-out regime)
        self.weibull_eta = 60.0     # scale (design life in steps)

        # Covariate coefficients (tune carefully!)
        self.beta_health = 2.0
        self.beta_dop = 1.0
        self.beta_cov = 2.5
        self.beta_cno = 1.5
 
    def is_terminal(self, state: GNSSState):
        return state.N_operational == 0
    
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
    
    def apply_action(self, state:GNSSState, action: str):
        planes = list(state.sats_per_plane)
        spares = state.spares

        if action == "ACTIVATE_SPARE" and spares > 0:
            weakest = self.weakest_plane(planes)
            if planes[weakest] < self.capacity_per_plane:
                planes[weakest] += 1
                spares -= 1
        elif action == "RETIRE_SAT":
            strongest = self.strongest_planes(planes)
            if planes[strongest] > 0:
                planes[strongest] -= 1
                spares += 1

        elif action == "LAUNCH_1":
            spares = min(spares + 1, self.max_spares)

        elif action == "REBALANCE_PLANE":
            strongest = self.strongest_planes(planes)
            weakest = self.weakest_plane(planes)
            if planes[strongest] > 0 and planes[weakest] < self.capacity_per_plane:
                planes[strongest] -= 1
                planes[weakest] += 1

        elif action == "NO_OP":
            pass

        return tuple(planes), spares
    
    
    # PDS computation

    def post_decision_state(self, state: GNSSState, action: str):

        Nop, Ns, h, cov, dop, cno, age = state.to_tuple()

        if action == "LAUNCH_1":
            Ns = min(Ns + 1, self.max_spares)

        elif action == "ACTIVATE_SPARE" and Ns > 0:
            Nop += min(Nop + 1, self.max_sats)
            Ns -= 1

        elif action == "RETIRE_SAT" and Nop > 0:
            Nop -= 1        

        '''
        Here we are omitting the other functions because the effects of te removed actions are 
        undeterministic and needs to finish the simulation step to be computed.
        '''

        age = min(age + 1, max(self.age_bins))

        return (Nop, Ns, h, age)

    # Failure probability computation

    def compute_failure_probability(self, state: GNSSState):

        age = state.age_bin
        mean_health = state.mean_health_bin / 10.0
        dop = state.dop_bin / 10.0
        coverage = state.four_cov_bin / 10.0
        cno = state.cno_bin / 10.0

        # Weibull baseline hazard
        if age ==0:
            baseline_hazard = 0.0
        else:
            k = self.weibull_k
            eta = self.weibull_eta
            baseline_hazard = (k / eta) * (age / eta) ** (k - 1)

        # Covariates
        x_health = 1.0 - mean_health
        x_dop = dop
        x_cov = 1.0 - coverage
        x_cno = 1.0 - cno

        linear_term = (
            self.beta_health * x_health +
            self.beta_dop * x_dop +
            self.beta_cov * x_cov +
            self.beta_cno * x_cno
        )

        stress_multiplier = np.exp(linear_term)

        # Proportional hazard
        hazard = baseline_hazard * stress_multiplier

        p_fail = 1 - np.exp(-hazard)

        return p_fail
    # KPI Models (analytical, simulator-ready)

    def evaluate_cno_db(self, mean_health):
        CNO_nominal = 45.0
        health_loss = 8.0 * (1 - mean_health)
        shadowing = np.random.normal(0, 1.5) # In actual simulator we need to replace this with the actual link budet

        return CNO_nominal - health_loss + shadowing
    
    def visibility_probability(self, cno_db):
        CNO_treshold = 38.0
        margin = cno_db - CNO_treshold

        #Monte carlo compressed according to the dissertation
        return 1 / (1 + np.exp(-margin))

    def sample_visibility_satellites(self, Nop, p_vis):
        return np.random.binomial(Nop, p_vis)
    
    def evaluate_coverage(self, N_vis):
        return 1 if N_vis >= 4 else 0
    
    def evaluate_dop(self, N_vis):
        if N_vis < 4:
            return 10
        
        base_dop = 6.0 / np.sqrt(N_vis)
        noise = np.random.normal(0, 0.5)
        return np.clip(base_dop + noise, 1.0, 10.0)

    # Environment Transition (stochastic physics + failures)

    def environment_transition(self, pds):

        Nop, Ns, h_bin, age = pds

        # -----------------------------
        # 1️⃣ Health degradation
        # -----------------------------
        mean_health = h_bin / 10.0
        degradation_noise = np.random.normal(0, 0.01)

        new_health = np.clip(
            mean_health - self.health_decay + degradation_noise,
            0.0, 1.0
        )
        new_h_bin = int(round(new_health * 10))

        # -----------------------------
        # 2️⃣ KPI evaluation (before failures)
        # -----------------------------
        cno_db = self.evaluate_cno_db(new_health)
        cno_bin = int(np.clip(round((cno_db - 30) / 20 * 10), 0, 10))

        p_vis = self.visibility_probability(cno_db)
        N_vis = self.sample_visibility_satellites(Nop, p_vis)

        coverage = self.evaluate_coverage(N_vis)
        cov_bin = int(coverage * 10)

        dop_value = self.evaluate_dop(N_vis)
        dop_bin = int(np.clip(round(dop_value), 0, 10))

        # -----------------------------
        # 3️⃣ Build temporary state for PH
        # -----------------------------
        temp_state = GNSSState(
            N_operational=Nop,
            N_spare=Ns,
            mean_health_bin=new_h_bin,
            four_cov_bin=cov_bin,
            dop_bin=dop_bin,
            cno_bin=cno_bin,
            age_bin=age
        )

        # -----------------------------
        # 4️⃣ Proportional Hazard failure
        # -----------------------------
        p_fail = self.compute_failure_probability(temp_state)
        failures = np.random.binomial(Nop, p_fail)

        Nop_next = max(Nop - failures, 0)

        age_next = min(age, max(self.age_bins))

        # -----------------------------
        # 5️⃣ Return next state
        # -----------------------------
        return GNSSState(
            Nop_next,
            Ns,
            new_h_bin,
            cov_bin,
            dop_bin,
            cno_bin,
            age_next
        )
    # Reward function

    def coverage_reward(self, cov_bin):
        cov = cov_bin / 10.0
        return cov ** 2  #strong penalization for low coverage
    
    def dop_penalty(self, dop_bin):
        dop = max(dop_bin /10.0, 0.1)
        return dop **2
    
    def cno_reward(self, cno_bin):
        cno = cno_bin / 10.0
        return np.tanh(2.0 * cno) 
    
    def action_cost_penalty(self, action):
        max_cost = max(self.action_cost.values())
        return self.action_cost[action] / max_cost
    
    def reward(self, state: GNSSState, action: str):

        r_cov = self.coverage_reward(state.four_cov_bin)
        r_dop = self.dop_penalty(state.dop_bin)
        r_cno = self.cno_reward(state.cno_bin)
        r_action = self.action_cost_penalty(action)

        total_reward = self.w1 * r_cov - self.w2 * r_dop + self.w3 * r_cno - self.w4 * r_action

        return total_reward

    def transition(self, state: GNSSState, action: str):

        pds = self.post_decision_state(state, action)
        next_state = self.environment_transition(pds)

        if next_state.four_cov_bin == 0:
            r = -10.0

        else:
            r = self.reward(next_state, action)

        return [(1.0, next_state, r)]

    # Terminal condition (optional)


    
mdp = GNSSConstellationMDP()

# # Healthy GNSS-like initial state
# state = GNSSState(
#     N_operational=24,
#     N_spare=2,
#     mean_health_bin=10,
#     four_cov_bin=10,
#     dop_bin=2,
#     cno_bin=8,
#     age_bin=0
# )

# print("Initial state:", state)

# for action in mdp.actions:
#     result = mdp.transition(state, action)[0]
#     prob, next_state, reward = result

#     print("\nAction:", action)
#     print("Next state:", next_state)
#     print("Reward:", round(reward, 3))

state = GNSSState(
    N_operational= GNSSConstellationMDP().max_sats,
    N_spare= GNSSConstellationMDP().max_spares,
    mean_health_bin=10,
    four_cov_bin=10,
    dop_bin=0,
    cno_bin=10,
    age_bin=GNSSConstellationMDP().age_bins[0]
)

print("\nFailure evolution test:")
for t in range(51):
    _, state, _ = mdp.transition(state, "NO_OP")[0]
    print(f"Time step {t}: Operational={state.N_operational}, Age={state.age_bin}, health={state.mean_health_bin/10.0:.2f}, cov={state.four_cov_bin/10.0:.2f}, dop={state.dop_bin/10.0:.2f}, cno={state.cno_bin/10.0:.2f}")