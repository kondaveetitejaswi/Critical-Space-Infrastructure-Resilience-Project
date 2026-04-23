import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

from main_constellation_with_skyfield_packages import (
    SkyfieldPhysicsEngine,
    SkyfieldADPSolver,
    GPS_PARAMS
)

# ===============================
# SCENARIO CLASSES (ROBUST DESIGN)
# ===============================

class JammingScenario:
    def __init__(self):
        self.attack_steps = sorted([
            np.random.randint(10, 40),
            np.random.randint(50, 90),
            np.random.randint(100, 140)
        ])
        self.active_until = -1

    def __call__(self, state, step):
        if step in self.attack_steps:
            self.active_until = step + 5

        if step <= self.active_until:
            return state.__class__(
                satellites=state.satellites,
                current_time=state.current_time,
                spares_available=state.spares_available,
                active_jammer_location=(0.0, 0.0),
                jammer_persistence_steps=1,
                threat_phase="HIGH",
                pending_launches=state.pending_launches
            )
        return state


class CascadeScenario:
    def __init__(self):
        self.attack_steps = sorted([
            np.random.randint(20, 50),
            np.random.randint(60, 100)
        ])

    def __call__(self, state, step):
        if step in self.attack_steps:
            sats = list(state.satellites)

            affected = np.random.choice(len(sats), 6, replace=False)

            for i in affected:
                s = sats[i]
                sats[i] = s.__class__(
                    s.sat_name, s.satellite_obj,
                    0.0, s.age_days,
                    s.orbital_plane,
                    s.launch_countdown,
                    False
                )

            return state.__class__(
                satellites=tuple(sats),
                current_time=state.current_time,
                spares_available=state.spares_available,
                active_jammer_location=None,
                jammer_persistence_steps=0,
                threat_phase="HIGH",
                pending_launches=state.pending_launches
            )

        return state


class HighThreatScenario:
    def __call__(self, state, step):
        return state.__class__(
            satellites=state.satellites,
            current_time=state.current_time,
            spares_available=state.spares_available,
            active_jammer_location=state.active_jammer_location,
            jammer_persistence_steps=state.jammer_persistence_steps,
            threat_phase="HIGH",
            pending_launches=state.pending_launches
        )


# ===============================
# POLICY EXECUTION
# ===============================

def run_trained_policy(adp, physics, initial_state, steps=150, scenario_fn=None):

    state = initial_state
    kpis = physics.compute_kpis(state)

    logs = []

    for step in range(1, steps + 1):

        if scenario_fn:
            state = scenario_fn(state, step)

        # 🔴 FIX: recompute KPIs after scenario
        kpis = physics.compute_kpis(state)

        macro = adp.get_macro_state(state, kpis)
        action = adp.greedy_action(macro, state)

        next_state, reward, next_kpis, atk, atk_log = \
            physics.step_physics(state, action, iteration=step)

        logs.append({
            "step": step,
            "coverage": next_kpis['coverage_pct'],
            "pdop": next_kpis['avg_pdop'],
            "cn0": next_kpis['avg_cn0_dbhz'],
            "action": action
        })

        state = next_state
        kpis = next_kpis

    return pd.DataFrame(logs)


# ===============================
# PLOTTING
# ===============================

def plot_curve(df, title, save_path, vlines=None, hline=None):

    plt.figure(figsize=(6,4))
    plt.plot(df['step'], df['coverage'])

    if vlines:
        for v in vlines:
            plt.axvline(v, linestyle='--')

    if hline:
        plt.axhline(hline, linestyle=':')

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":

    physics = SkyfieldPhysicsEngine(GPS_PARAMS)

    # ✅ FIXED initialization
    adp = SkyfieldADPSolver(physics)

    policy_path = 'C:\\Users\\tejas\\Masters Thesis\\Critical-Space-Infrastructure-Resilience-Project\\resilience_output\\ADP_policy.pkl'

    with open(policy_path, 'rb') as f:
        adp.post_decision_values = pickle.load(f)

    print("Loaded policy states:", len(adp.post_decision_values))

    save_dir = "scenario_outputs"
    os.makedirs(save_dir, exist_ok=True)

    num_runs = 10

    for run in range(1, num_runs + 1):

        print(f"\n===== RUN {run} =====")

        initial_state = physics.download_initial_state()

        # NEW scenario objects per run
        jam = JammingScenario()
        cas = CascadeScenario()
        high = HighThreatScenario()

        # ---- Jamming ----
        df_jam = run_trained_policy(adp, physics, initial_state, scenario_fn=jam)
        plot_curve(df_jam, "Jamming", 
                   os.path.join(save_dir, f"threat_1_run_{run}.png"),
                   vlines=jam.attack_steps, hline=95)

        # ---- Cascade ----
        df_cas = run_trained_policy(adp, physics, initial_state, scenario_fn=cas)
        plot_curve(df_cas, "Cascade", 
                   os.path.join(save_dir, f"threat_2_run_{run}.png"),
                   vlines=cas.attack_steps, hline=90)

        # ---- High Threat ----
        df_high = run_trained_policy(adp, physics, initial_state, scenario_fn=high)
        plot_curve(df_high, "High Threat", 
                   os.path.join(save_dir, f"threat_3_run_{run}.png"),
                   hline=90)


    # ===============================
    # COMBINED VISUALIZATION
    # ===============================

    fig, axes = plt.subplots(num_runs, 3, figsize=(12, 3*num_runs))

    for run in range(1, num_runs + 1):
        for threat in range(1, 4):

            img_path = os.path.join(save_dir, f"threat_{threat}_run_{run}.png")
            img = mpimg.imread(img_path)

            axes[run-1, threat-1].imshow(img)
            axes[run-1, threat-1].axis('off')

            if run == 1:
                axes[run-1, threat-1].set_title(
                    ["Jamming", "Cascade", "High Threat"][threat-1]
                )

    plt.tight_layout()
    plt.show()