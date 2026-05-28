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

def run_no_op_policy(physics, steps=150):

    state = physics.download_initial_state()
    kpis = physics.compute_kpis(state)

    coverage = []
    action_count = 0  # always 0

    for step in range(1, steps + 1):

        kpis = physics.compute_kpis(state)

        action = "NO_OP"

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

def attack_intensity_experiment_with_baseline(adp):

    scales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    results_adp = {}
    results_noop = {}

    for scale in scales:

        physics = ScaledPhysics(GPS_PARAMS, attack_scale=scale)

        runs_adp = []
        actions_adp = []

        runs_noop = []
        actions_noop = []

        for _ in range(10):

            # 🔥 SAME RANDOMNESS FOR FAIR COMPARISON
            seed = np.random.randint(0, 100000)

            np.random.seed(seed)
            cov_adp, act_adp = run_policy(adp, physics)

            np.random.seed(seed)
            cov_noop, act_noop = run_no_op_policy(physics)

            runs_adp.append(cov_adp)
            actions_adp.append(act_adp)

            runs_noop.append(cov_noop)
            actions_noop.append(act_noop)

        results_adp[scale] = compute_summary(runs_adp, actions_adp)
        results_noop[scale] = compute_summary(runs_noop, actions_noop)

    print("\n=== ATTACK INTENSITY (ADP vs NO_OP) ===")

    for scale in scales:
        print(f"\nScale {scale}:")
        print(f"  ADP   -> {results_adp[scale]}")
        print(f"  NO_OP -> {results_noop[scale]}")

    return results_adp, results_noop


def spare_capacity_experiment_with_baseline(adp):

    capacities = [2,3,4,5,6,7,8,9,10,11,12]

    results_adp = {}
    results_noop = {}

    for cap in capacities:

        params = deepcopy(GPS_PARAMS)
        params.spare_capacity = cap

        physics = SkyfieldPhysicsEngine(params)

        runs_adp = []
        actions_adp = []

        runs_noop = []
        actions_noop = []

        for _ in range(10):

            seed = np.random.randint(0, 100000)

            np.random.seed(seed)
            cov_adp, act_adp = run_policy(adp, physics)

            np.random.seed(seed)
            cov_noop, act_noop = run_no_op_policy(physics)

            runs_adp.append(cov_adp)
            actions_adp.append(act_adp)

            runs_noop.append(cov_noop)
            actions_noop.append(act_noop)

        results_adp[cap] = compute_summary(runs_adp, actions_adp)
        results_noop[cap] = compute_summary(runs_noop, actions_noop)

    print("\n=== SPARE CAPACITY (ADP vs NO_OP) ===")

    for cap in sorted(results_adp.keys()):
        print(f"\nCapacity {cap}:")
        print(f"  ADP   -> {results_adp[cap]}")
        print(f"  NO_OP -> {results_noop[cap]}")

    return results_adp, results_noop
# ===============================
# PLOTTING
# ===============================
def plot_attack_baseline(results_adp, results_noop):

    scales = sorted(results_adp.keys())

    adp_avg = [results_adp[s]['avg_coverage'] for s in scales]
    noop_avg = [results_noop[s]['avg_coverage'] for s in scales]

    adp_min = [results_adp[s]['min_coverage'] for s in scales]
    noop_min = [results_noop[s]['min_coverage'] for s in scales]

    # ---- Average ----
    plt.figure(figsize=(8,5))
    plt.plot(scales, adp_avg, marker='o', label='ADP Avg')
    plt.plot(scales, noop_avg, linestyle='--', label='NO_OP Avg')
    plt.xlabel("Attack Scale")
    plt.ylabel("Average Coverage")
    plt.title("Attack Intensity vs Coverage (ADP vs NO_OP)")
    plt.legend()
    plt.savefig("Attack_Baseline_Avg.png")
    plt.show()

    # ---- Minimum ----
    plt.figure(figsize=(8,5))
    plt.plot(scales, adp_min, marker='o', label='ADP Min')
    plt.plot(scales, noop_min, linestyle='--', label='NO_OP Min')
    plt.xlabel("Attack Scale")
    plt.ylabel("Minimum Coverage")
    plt.title("Attack Intensity vs Worst Case (ADP vs NO_OP)")
    plt.legend()
    plt.savefig("Attack_Baseline_Min.png")
    plt.show()

def plot_baseline_comparison(results_adp, results_noop):

    caps = sorted(results_adp.keys())

    adp_avg = [results_adp[c]['avg_coverage'] for c in caps]
    noop_avg = [results_noop[c]['avg_coverage'] for c in caps]

    adp_min = [results_adp[c]['min_coverage'] for c in caps]
    noop_min = [results_noop[c]['min_coverage'] for c in caps]

    plt.figure(figsize=(8,5))
    plt.plot(caps, adp_avg, label="ADP Avg Coverage", marker='o')
    plt.plot(caps, noop_avg, label="NO_OP Avg Coverage", linestyle='--')
    plt.xlabel("Spare Capacity")
    plt.ylabel("Average Coverage")
    plt.title("ADP vs NO_OP (Average)")
    plt.legend()
    plt.savefig("Baseline_Avg_Comparison.png")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(caps, adp_min, label="ADP Min Coverage", marker='o')
    plt.plot(caps, noop_min, label="NO_OP Min Coverage", linestyle='--')
    plt.xlabel("Spare Capacity")
    plt.ylabel("Minimum Coverage")
    plt.title("ADP vs NO_OP (Worst Case)")
    plt.legend()
    plt.savefig("Baseline_Min_Comparison.png")
    plt.show()

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
    print("Loaded pre-trained ADP policy.")
    print(f"Spare Capacity: {physics.params.spare_capacity}")
    spare_adp, spare_noop = spare_capacity_experiment_with_baseline(adp)
    attack_adp, attack_noop = attack_intensity_experiment_with_baseline(adp)

    plot_spare_curve(spare_adp)
    plt.savefig("Spare_Capacity_Sensitivity.png")
    plot_baseline_comparison(spare_adp, spare_noop)
    plt.savefig("Baseline_Spare_Capacity_Comparison.png")

    plot_attack_baseline(attack_adp, attack_noop)
    plt.savefig("Baseline_Attack_Intensity_Comparison.png")
