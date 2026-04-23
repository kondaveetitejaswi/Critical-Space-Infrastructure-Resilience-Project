import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import Environment
import matplotlib.image as mpimg

from main_constellation_with_skyfield_packages import ConstellationParameters, SkyfieldSatelliteState, SkyfieldConstellationState, SkyfieldPhysicsEngine, SkyfieldADPSolver, GPS_PARAMS

import pickle
import os

policy_path = 'C:\\Users\\tejas\\Masters Thesis\\Critical-Space-Infrastructure-Resilience-Project\\resilience_output\\ADP_policy.pkl'

adp = SkyfieldADPSolver(None)  

def run_trained_policy(adp, physics, initial_state, steps=150, scenario_fn=None):
    state = initial_state
    kpis = physics.compute_kpis(state)

    logs = []

    for step in range(1, steps + 1):

        # 🔹 Apply forced scenario BEFORE decision
        if scenario_fn:
            state = scenario_fn(state, step)

        macro = adp.get_macro_state(state, kpis)
        action = adp.greedy_action(macro, state)

        next_state, reward, next_kpis, atk, atk_log = \
            physics.step_physics(state, action, iteration=step)

        logs.append({
            "step": step,
            "coverage": next_kpis['coverage_pct'],
            "pdop": next_kpis['avg_pdop'],
            "cn0": next_kpis['avg_cn0_dbhz'],
            "action": action,
            "attack": atk,
            "attack_log": "; ".join(atk_log) if atk_log else ""
        })

        state = next_state
        kpis = next_kpis

    return pd.DataFrame(logs)

def jamming_scenario(state, step):
    if step == 20:
        return state.__class__(
            satellites=state.satellites,
            current_time=state.current_time,
            spares_available=state.spares_available,
            active_jammer_location=(0.0, 0.0),  # Equator
            jammer_persistence_steps=5,
            threat_phase=state.threat_phase,
            pending_launches=state.pending_launches
        )
    return state

def analyze_jamming(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['step'], df['coverage'], label='Coverage')
    plt.axvline(20, color='red', linestyle='--', label='Jamming Start')
    plt.axhline(95, linestyle=':', label='Recovery Threshold')
    plt.legend()
    plt.title("Jamming Attack Recovery")
    plt.xlabel("Step")
    plt.ylabel("Coverage (%)")
    plt.show()

    # Recovery time
    after_attack = df[df['step'] > 20]
    recovery = after_attack[after_attack['coverage'] >= 95]

    if not recovery.empty:
        recovery_step = recovery.iloc[0]['step']
        print("Recovery Time:", recovery_step - 20)
    else:
        print("No recovery")

def cascade_scenario(state, step):
    if step == 20:
        sats = list(state.satellites)

        # Kill multiple satellites
        for i in range(min(6, len(sats))):
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
            threat_phase=state.threat_phase,
            pending_launches=state.pending_launches
        )
    return state

def analyze_cascade(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['step'], df['coverage'])
    plt.axvline(20, color='red', linestyle='--', label='Cascade')
    plt.axhline(90, linestyle=':', label='Failure Threshold')
    plt.legend()
    plt.title("Cascade Failure Response")
    plt.show()

    min_cov = df['coverage'].min()
    print("Minimum Coverage:", min_cov)

    if min_cov < 90:
        print("⚠️ System entered critical state")
    else:
        print("✅ System remained stable")

def high_threat_scenario(state, step):
    return state.__class__(
        satellites=state.satellites,
        current_time=state.current_time,
        spares_available=state.spares_available,
        active_jammer_location=state.active_jammer_location,
        jammer_persistence_steps=state.jammer_persistence_steps,
        threat_phase="HIGH",
        pending_launches=state.pending_launches
    )

def analyze_high_threat(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['step'], df['coverage'], label='Coverage')
    plt.axhline(90, linestyle=':', label='Target')
    plt.legend()
    plt.title("High Threat Phase Robustness")
    plt.show()

    print("Average Coverage:", df['coverage'].mean())
    print("Minimum Coverage:", df['coverage'].min())

def plot_high(df, save_path):
    plt.figure(figsize=(6,4))
    plt.plot(df['step'], df['coverage'])
    plt.axhline(90, linestyle=':')
    plt.title("High Threat")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_jamming(df, save_path):
    plt.figure(figsize=(6,4))
    plt.plot(df['step'], df['coverage'])
    plt.axvline(20, linestyle='--')
    plt.axhline(95, linestyle=':')
    plt.title("Jamming")
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cascade(df, save_path):
    plt.figure(figsize=(6,4))
    plt.plot(df['step'], df['coverage'])
    plt.axvline(20, linestyle='--')
    plt.axhline(90, linestyle=':')
    plt.title("Cascade")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    physics = SkyfieldPhysicsEngine(GPS_PARAMS)

    adp = SkyfieldADPSolver(physics)

    # Load trained policy
    policy_path = 'C:\\Users\\tejas\\Masters Thesis\\Critical-Space-Infrastructure-Resilience-Project\\resilience_output\\ADP_policy.pkl'
    with open(policy_path, 'rb') as f:
        adp.post_decision_values = pickle.load(f)

    save_dir = "scenario_outputs"
    os.makedirs(save_dir, exist_ok=True)

    num_runs = 10

    for run in range(1, num_runs + 1):

        print(f"\n===== RUN {run} =====")

        # IMPORTANT: new initial state every run
        initial_state = physics.download_initial_state()

        # ---- Jamming ----
        df_jam = run_trained_policy(adp, physics, initial_state,
                                   scenario_fn=jamming_scenario)

        plot_jamming(df_jam, os.path.join(save_dir, f"threat_1_run_{run}.png"))

        # ---- Cascade ----
        df_cascade = run_trained_policy(adp, physics, initial_state,
                                       scenario_fn=cascade_scenario)

        plot_cascade(df_cascade, os.path.join(save_dir, f"threat_2_run_{run}.png"))

        # ---- High Threat ----
        df_high = run_trained_policy(adp, physics, initial_state,
                                    scenario_fn=high_threat_scenario)

        plot_high(df_high, os.path.join(save_dir, f"threat_3_run_{run}.png"))


    fig, axes = plt.subplots(10, 3, figsize=(12, 30))

    for run in range(1, 11):

        for threat in range(1, 4):

            img_path = os.path.join(save_dir, f"threat_{threat}_run_{run}.png")

            img = mpimg.imread(img_path)

            axes[run-1, threat-1].imshow(img)
            axes[run-1, threat-1].axis('off')

            if run == 1:
                title = ["Jamming", "Cascade", "High Threat"][threat-1]
                axes[run-1, threat-1].set_title(title)

    plt.tight_layout()
    plt.show()