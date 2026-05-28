import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy

from main_constellation_with_skyfield_packages_3 import (
    SkyfieldPhysicsEngine,
    SkyfieldADPSolver,
    GPS_PARAMS
)

# ===============================
# SCALED PHYSICS (ONLY THIS IN CLASS)
# ===============================

class ScaledPhysics(SkyfieldPhysicsEngine):

    def __init__(self, params, attack_scale=1.0):
        super().__init__(params)
        self.attack_scale = attack_scale

    def _apply_attacks(self, sats, iteration):

        sats, jammer_loc, persist, attack_log = \
            super()._apply_attacks(sats, iteration)

        # ---- SCALE EFFECTS ----
        if self.attack_scale > 1.0:

            # 1) Increase probability of extra attack
            if np.random.rand() < 0.2 * self.attack_scale:

                active = [i for i, s in enumerate(sats) if s.health > 0.5]

                if active:
                    num_targets = int(1 + self.attack_scale)   # more satellites affected
                    targets = np.random.choice(active, min(len(active), num_targets), replace=False)

                    for idx in targets:
                        s = sats[idx]
                        sats[idx] = s.__class__(
                            s.sat_name, s.satellite_obj,
                            max(s.health - 0.6, 0.01),  # stronger degradation
                            s.age_days, s.orbital_plane,
                            s.launch_countdown, True
                        )

                    attack_log.append(f"[SCALED] Multi-sat degradation ({len(targets)})")

            # 2) Increase jammer persistence
            if persist > 0:
                persist = int(persist * self.attack_scale)

        return sats, jammer_loc, persist, attack_log


# ===============================
# FUNCTIONS (OUTSIDE CLASS)
# ===============================

def run_policy(adp, physics, steps=150):

    state = physics.download_initial_state()
    kpis = physics.compute_kpis(state)

    coverage = []
    action_count = 0

    for step in range(1, steps + 1):

        kpis = physics.compute_kpis(state)

        macro = adp.get_macro_state(state, kpis)
        action = adp.greedy_action(macro, state)
        if action != "NO_OP":
            action_count += 1

        state, _, kpis, _, _ = physics.step_physics(state, action, iteration=step)

        coverage.append(kpis['coverage_pct'])

    return np.array(coverage), action_count


def compute_summary(runs, actions):

    arr = np.array(runs)

    return {
        "avg_coverage": np.mean(arr),
        "min_coverage": np.min(arr),
        "std": np.std(arr),
        "collapse_rate": np.mean(arr.min(axis=1) < 90),
        "avg_actions": np.mean(actions)
    }


# ===============================
# EXPERIMENTS
# ===============================

def attack_intensity_experiment(adp):

    scales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = {}

    for scale in scales:

        physics = ScaledPhysics(GPS_PARAMS, attack_scale=scale)

        runs = []
        actions =[]

        for _ in range(10):
            cov, act = run_policy(adp, physics)
            runs.append(cov)
            actions.append(act)

        results[scale] = compute_summary(runs, actions)

    print("\n=== ATTACK INTENSITY ===")
    for k, v in results.items():
        print(f"Scale {k}: {v}")

    return results


def spare_capacity_experiment(adp):

    capacities = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    results = {}

    for cap in capacities:

        from copy import deepcopy
        params = deepcopy(GPS_PARAMS)
        params.spare_capacity = cap

        physics = SkyfieldPhysicsEngine(params)

        runs = []
        actions = []

        for _ in range(10):
            cov, act = run_policy(adp, physics)
            runs.append(cov)
            actions.append(act)

        results[cap] = compute_summary(runs, actions)

    print("\n=== SPARE CAPACITY ===")
    for k, v in results.items():
        print(f"Capacity {k}: {v}")

    return results


# ===============================
# PLOTTING
# ===============================

def plot_sensitivity(results, title):

    x = list(results.keys())
    y = [results[k]['avg_coverage'] for k in x]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel("Parameter")
    plt.ylabel("Average Coverage")
    plt.savefig(f"{title}.png")
    plt.show()

def plot_spare_curve(results):

    caps = sorted(results.keys())

    avg = [results[c]['avg_coverage'] for c in caps]
    minv = [results[c]['min_coverage'] for c in caps]
    actions = [results[c]['avg_actions'] for c in caps]

    plt.figure(figsize=(8,5))
    plt.plot(caps, avg, label='Avg Coverage')
    plt.plot(caps, minv, label='Min Coverage')
    plt.xlabel("Spare Capacity")
    plt.ylabel("Coverage")
    plt.legend()
    plt.title("Spare Capacity Sensitivity")
    plt.savefig("Spare_Capacity_Sensitivity.png")
    plt.show()

    # 🔥 THIS is the key insight plot
    plt.figure(figsize=(8,5))
    plt.plot(caps, actions, marker='o')
    plt.xlabel("Spare Capacity")
    plt.ylabel("Avg Actions Taken")
    plt.title("Policy Aggressiveness vs Capacity")
    plt.savefig("Policy_Aggressiveness_vs_Capacity.png")
    plt.show()


# ===============================
# MAIN (ONLY HERE)
# ===============================

if __name__ == "__main__":

    physics = SkyfieldPhysicsEngine(GPS_PARAMS)
    adp = SkyfieldADPSolver(physics)

    with open("resilience_output/ADP_policy_3.pkl", "rb") as f:
        adp.post_decision_values = pickle.load(f)
    print("Loaded ADP policy with", len(adp.post_decision_values), "entries")
    attack_results = attack_intensity_experiment(adp)
    print("attack_results:", attack_results)
    spare_results = spare_capacity_experiment(adp)
    print("spare_results:", spare_results)


    plot_sensitivity(attack_results, "Attack Intensity vs Coverage")
    plt.savefig("Attack_Intensity_Sensitivity.png")
    plot_sensitivity(spare_results, "Spare Capacity vs Coverage")
    plt.savefig("Spare_Capacity_Sensitivity.png")
    plot_spare_curve(spare_results)
    plt.savefig("Spare_Capacity_Detailed.png")