import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from main_constellation_with_skyfield_packages_3 import (
    SkyfieldPhysicsEngine,
    SkyfieldADPSolver,
    GPS_PARAMS
)

# ===============================
# SCENARIOS
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
            self.active_until = step + 10

        if step <= self.active_until:
            return state.__class__(
                satellites=state.satellites,
                current_time=state.current_time,
                spares_available=state.spares_available,
                active_jammer_location=(0.0, 0.0),
                jammer_persistence_steps=2,
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

            affected = np.random.choice(len(sats), 8, replace=False)

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
    def __init__(self):
        self.attack_steps = sorted(
            np.random.choice(range(10, 140), size=5, replace=False)
        )
        self.active_until = -1

    def __call__(self, state, step):

        if step in self.attack_steps:
            self.active_until = step + 6

        if step <= self.active_until:
            return state.__class__(
                satellites=state.satellites,
                current_time=state.current_time,
                spares_available=state.spares_available,
                active_jammer_location=(np.random.uniform(-30,30),
                                        np.random.uniform(-60,60)),
                jammer_persistence_steps=2,
                threat_phase="HIGH",
                pending_launches=state.pending_launches
            )

        return state


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

        kpis = physics.compute_kpis(state)

        macro = adp.get_macro_state(state, kpis)
        action = adp.greedy_action(macro, state)

        next_state, reward, next_kpis, atk, atk_log = \
            physics.step_physics(state, action, iteration=step)

        logs.append({
            "step": step,
            "coverage": next_kpis['coverage_pct']
        })

        state = next_state
        kpis = next_kpis

    return pd.DataFrame(logs)


# ===============================
# METRICS
# ===============================

def compute_metrics(df):

    coverage = df['coverage'].values

    mean_cov = np.mean(coverage)
    min_cov = np.min(coverage)
    std_cov = np.std(coverage)

    recovery_time = -1
    below = np.where(coverage < 95)[0]

    if len(below) > 0:
        start = below[0]
        for i in range(start, len(coverage)):
            if coverage[i] >= 95:
                recovery_time = i - start
                break

    return mean_cov, min_cov, std_cov, recovery_time


def summarize(metrics_list):

    arr = np.array(metrics_list)

    return {
        "mean_cov": np.mean(arr[:,0]),
        "min_cov": np.mean(arr[:,1]),
        "std_cov": np.mean(arr[:,2]),
        "recovery": np.mean(arr[:,3][arr[:,3] >= 0])
    }


# ===============================
# PLOTTING
# ===============================

def save_run_plot(df, title, save_path, vlines=None, hline=None):

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


def plot_mean_std(all_runs, title):

    arr = np.array(all_runs)

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    steps = np.arange(len(mean))

    plt.figure(figsize=(8,5))
    plt.plot(steps, mean, label="Mean Coverage")
    plt.fill_between(steps, mean-std, mean+std, alpha=0.3)

    plt.axhline(90, linestyle=':')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Coverage (%)")
    plt.legend()
    plt.show()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    physics = SkyfieldPhysicsEngine(GPS_PARAMS)
    adp = SkyfieldADPSolver(physics)

    with open("resilience_output/ADP_policy_3.pkl", "rb") as f:
        adp.post_decision_values = pickle.load(f)
    print("Loaded policy states:", len(adp.post_decision_values))

    save_dir = "scenario_outputs"
    os.makedirs(save_dir, exist_ok=True)

    num_runs = 10

    all_jam, all_cas, all_high = [], [], []
    metrics_jam, metrics_cas, metrics_high = [], [], []

    for run in range(num_runs):

        print(f"\n===== RUN {run} =====")

        initial_state = physics.download_initial_state()

        jam = JammingScenario()
        cas = CascadeScenario()
        high = HighThreatScenario()

        # ---- Jamming ----
        df_jam = run_trained_policy(adp, physics, initial_state, scenario_fn=jam)
        save_run_plot(df_jam, "Jamming",
                      os.path.join(save_dir, f"threat_1_run_{run+1}.png"),
                      vlines=jam.attack_steps, hline=95)

        # ---- Cascade ----
        df_cas = run_trained_policy(adp, physics, initial_state, scenario_fn=cas)
        save_run_plot(df_cas, "Cascade",
                      os.path.join(save_dir, f"threat_2_run_{run+1}.png"),
                      vlines=cas.attack_steps, hline=90)

        # ---- High Threat ----
        df_high = run_trained_policy(adp, physics, initial_state, scenario_fn=high)
        save_run_plot(df_high, "High Threat",
                      os.path.join(save_dir, f"threat_3_run_{run+1}.png"),
                      vlines=high.attack_steps, hline=90)

        all_jam.append(df_jam['coverage'].values)
        all_cas.append(df_cas['coverage'].values)
        all_high.append(df_high['coverage'].values)

        metrics_jam.append(compute_metrics(df_jam))
        metrics_cas.append(compute_metrics(df_cas))
        metrics_high.append(compute_metrics(df_high))


    print("\n===== SUMMARY =====")
    print("Jamming:", summarize(metrics_jam))
    print("Cascade:", summarize(metrics_cas))
    print("High Threat:", summarize(metrics_high))

    plot_mean_std(all_jam, "Jamming (Mean ± Std)")
    plt.savefig(os.path.join(save_dir, "jamming_mean_std.png"))
    plot_mean_std(all_cas, "Cascade (Mean ± Std)")
    plt.savefig(os.path.join(save_dir, "cascade_mean_std.png"))

    plot_mean_std(all_high, "High Threat (Mean ± Std)")
    plt.savefig(os.path.join(save_dir, "high_threat_mean_std.png"))