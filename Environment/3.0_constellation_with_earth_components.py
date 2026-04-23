from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
from collections import defaultdict

# Constants and Earth models

Earth_radius_km = 6371.0
constellation_altitude_km = 20200.0
min_elevation_angle_deg = 5.0
good_pdop_threshold = 5.0
excellent_pdop_threshold = 3.0
grid_lat_step = 10
grid_lon_step = 10

# State
@dataclass(frozen=True)
class GNSSState:
    sats_per_plane: Tuple[int, ...]
    spares: int
    health: float
    age: int

    def total_operational(self):
        return sum(self.op_counts)
    
# Earth aware coverage model
class EarthCoverageModel:
    def __init__(self, n_planes = 6, sats_per_plane = 6):
        self.n_planes = n_planes
        self.sats_per_plane = sats_per_plane

        self.inclination = 55.0
        self.plane_separation = 360.0 / n_planes

        self.lat_grid = np.arange(-90, 91, grid_lat_step)
        self.lon_grid = np.arange(-180, 180, grid_lon_step)
        self.n_grid_cells = len(self.lat_grid) * len(self.lon_grid)

    
    def _estimate_elevation_angle(self, lat_ground, lon_ground, plane_lon, inclination):
        lon_diff = abs(lon_ground - plane_lon)
        if lon_diff > 180:
            lon_diff = 360 - lon_diff

        lat_diff = abs(lat_ground - inclination)

        angular_distance = np.sqrt(lon_diff**2 + lat_diff**2)
        elevation_angle = 90 - angular_distance
        return elevation_angle
    
    def _elevation_to_visibility(self, elevation, health):
        if elevation < min_elevation_angle_deg:
            return 0.05

        visibility = (elevation - min_elevation_angle_deg) / (90 - min_elevation_angle_deg)
        visibility *= health
        return visibility
    
    def compute_pdop_at_location(self, n_visible, elevation_angles):
        if n_visible < 4:
            return 10.0
        
        base_pdop = 10.0 / np.sqrt(n_visible)
        geometry_factor = 1.0 + 0.1 * np.random.normal(0, 1)

        pdop = np.clip(base_pdop * geometry_factor, 1.0, 10.0)
        return pdop
    

    def compute_visible_sats(self, lat, lon, planes, health):
        n_visible = 0
        total_sats = sum(planes)
        for plane_idx, n_sats in enumerate(planes):
            plane_lon = plane_idx * self.plane_separation

            elevation = self._estimate_elevation_angle(lat, lon, plane_lon, self.inclination)

            visibility_prob = self._elevation_to_visibility(elevation, health)

            expected_visible = n_sats * visibility_prob
            n_visible += expected_visible

        return n_visible
    
    def compute_grid_coverage(self, planes, health):
        adequate_cells = 0
        good_cells = 0
        poor_cells = 0

        lat_coverage = defaultdict(lambda: {'adequate': 0, 'total':0})

        for lat in self.lat_grid:
            for lon in self.lon_grid:

                n_visible = self.compute_visible_sats(lat, lon, planes, health)

                if n_visible < 4:
                    poor_cells += 1
                    coverage_type = 'poor'

                else:
                    pdop = self.compute_pdop_at_location(n_visible, [])

                    if pdop <= excellent_pdop_threshold:
                        good_cells += 1
                        coverage_type = 'good'

                    elif pdop <= good_pdop_threshold:
                        adequate_cells += 1
                        coverage_type = 'adequate'

                    else:
                        coverage_type = 'degraded'

                lat_band = int(lat // grid_lat_step)
                lat_coverage[lat_band]['total'] += 1
                if coverage_type in ['good', 'adequate']:
                    lat_coverage[lat_band]['adequate'] += 1

        total_cells = self.n_grid_cells
        overall_coverage_pct = 100.0 * (adequate_cells + good_cells) / total_cells

        good_coverage_pct = 100.0 * good_cells / total_cells
        return {
            "overall_coverage_pct": overall_coverage_pct,
            "good_coverage_pct": good_coverage_pct,
            "adequate_cells": adequate_cells,
            "good_cells": good_cells,
            "poor_cells": poor_cells,
            "lat_coverage": dict(lat_coverage),
            "n_visible_avg": sum(planes) * 0.7
        }
    
# MDP with modified KPI coverage
class GNSSConstellationMDP:
    def __init__(self,
                n_planes = 6, 
                sats_per_plane = 6,
                max_spares = 10):
        
        self.n_planes = n_planes
        self.capacity_per_plane = sats_per_plane
        self.max_sats = n_planes * sats_per_plane
        self.max_spares = max_spares

        self.min_coverage_pct = 95.0
        self.min_good_coverage_pct = 60.0

        self.health_decay = 0.02
        self.max_age = 50

        self.weibull_k = 4.5
        self.weibull_eta = 60

        self.beta_health = 2.0
        self.beta_geom = 1.5
        self.beta_dop = 1.0

        self.action_costs = {
            "NO_OP": 0,
            "LAUNCH_1": 10,
            "ACTIVATE_SPARE": 2,
            "RETIRE_SAT": 1,
            "REBALANCE_PLANE": 3
        }

        self.actions = list(self.action_costs.keys())

        self.earth_model = EarthCoverageModel(n_planes, sats_per_plane)


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
            if planes[s] > 0 and planes[w] < self.capacity_per_plane:
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
    
    def compute_kpis(self, planes, health):
        coverage_dict = self.earth_model.compute_grid_coverage(planes, health)

        overall_coverage_pct = coverage_dict["overall_coverage_pct"]
        good_coverage_pct = coverage_dict["good_coverage_pct"]

        service_available = 1 if overall_coverage_pct >= self.min_coverage_pct else 0

        degradation_factor = (
            0.6 * (overall_coverage_pct / 100.0) +
            0.4 * (good_coverage_pct / 100.0)
        )

        return overall_coverage_pct, good_coverage_pct, service_available, degradation_factor
    
    def compute_failure_probability(self, state, degradation):

        age = state.age
        health = state.health

        if age == 0:
            baseline = 0.0
        else:
            baseline = (self.weibull_k / self.self.weibull_eta) * (age / self.weibull_eta) ** (self.weibull_k - 1)

        geom_imbalance = np.std(state.sats_per_plane) if len(state.sats_per_plane) > 0 else 0.0 

        linear = (
            self.beta_health * (1.0 - health) +
            self.beta_geom * geom_imbalance +
            self.beta_dop * (1.0 - degradation)
        )

        hazard = baseline * np.exp(linear)
        failure_prob = 1 - np.exp(-hazard)
        return failure_prob
    
    def apply_failures(self, planes, p_fail):
        return tuple(
            max(n - np.random.binomial(n, p_fail), 0) for n in planes
        )
    

    def sample_next_state(self, state, action):
        planes, spares = self.apply_action(state, action)

        cov_pct, good_cov_pct, service_avail, quality = self.compute_kpis(planes, state.health)

        p_fail = self.compute_failure_probability(state, quality)
        planes = self.apply_failures(planes, p_fail)

        health = max(state.health + np.random.normal(0, 0.01), 0.0)

        age = min(state.age + 1, self.max_age)

        next_state = GNSSState(planes, spares, health, age)

        cost = self.action_costs[action]

        if service_avail == 0:
            cost += 20

        coverage_gap = max(0, self.min_coverage_pct - cov_pct)
        cost += 2.0 * coverage_gap

        good_coverage_gap = max(0, self.min_good_coverage_pct - good_cov_pct)
        cost += 1.0 * good_coverage_gap

        return next_state, cost, {
            "coverage_pct": cov_pct,
            "good_coverage_pct": good_cov_pct,
            "service_available": service_avail,
            "quality": quality
        }
    
    def transition(self, state, action, n_samples = 20):
        outcomes = {}

        for _ in range(n_samples):
            ns, cost, kpis = self.sample_next_state(state, action)
            reward = -cost

            if ns not in outcomes:
                outcomes[ns] = [0, reward, kpis]
            outcomes[ns][0] += 1

        transitions = []
        for ns, (count, r, kpis) in outcomes.items():
            prob = count / n_samples
            transitions.append((prob, ns, r, kpis))

        return transitions
    

    def greedy_action(self, state):
        best_action = None
        best_cost = float("inf")

        for action in self.actions:
            transitions = self.transition(state, action)

            expected_cost = 0.0
            for prob, _, reward, _ in transitions:
                expected_cost += prob * (-reward)

            if expected_cost < best_cost:
                best_cost = expected_cost
                best_action = action

        return best_action
    
    def run_simulation(self, policy, initial, steps = 50, verbose = True):
        state = initial
        trajectory = []

        for t in range(steps):

            if policy == "NO_OP":
                action = "NO_OP"
            elif policy == "GREEDY":
                action = self.greedy_action(state)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            transitions = self.transition(state, action)

            probs = [x[0] for x in transitions]
            idx = np.random.choice(len(transitions), p=probs)
            _, next_state, reward, kpis = transitions[idx]

            if verbose:
                print(
                    f"t={t:2d} | action={action:15s} | "
                    f"planes={state.sats_per_plane} | "
                    f"health={state.health:.2f} | "
                    f"coverage={kpis['coverage_pct']:5.1f}% | "
                    f"good_cov={kpis['good_coverage_pct']:5.1f}% | "
                    f"service={'ON' if kpis['service_available'] else 'OFF'}"
                )

            trajectory.append({
                'time': t,
                'action': action,
                'state': state,
                'coverage_pct': kpis['coverage_pct'],
                'good_coverage_pct': kpis['good_coverage_pct'],
                'service_available': kpis['service_available'],
                'quality': kpis['quality'],
                'reward': reward
            })

        return trajectory
    
# Simulation running
if __name__ == "__main__":

    mdp = GNSSConstellationMDP()

    initial = GNSSState(
        sats_per_plane=(6, 6, 6, 6, 6, 6),
        spares=10,
        health=1.0,
        age=0
    )

    print("=" * 120)
    print("NO-OP POLICY (Passive, no maintenance)")
    print("=" * 120)
    traj_noop = mdp.run_simulation("NO_OP", initial, steps=20)

    print("\n" + "=" * 120)
    print("GREEDY ADP POLICY (Active maintenance)")
    print("=" * 120)
    traj_greedy = mdp.run_simulation("GREEDY", initial, steps=20)
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    
    avg_cov_noop = np.mean([t['coverage_pct'] for t in traj_noop])
    avg_good_cov_noop = np.mean([t['good_coverage_pct'] for t in traj_noop])
    
    avg_cov_greedy = np.mean([t['coverage_pct'] for t in traj_greedy])
    avg_good_cov_greedy = np.mean([t['good_coverage_pct'] for t in traj_greedy])
    
    print(f"NO-OP:    Avg Coverage = {avg_cov_noop:.1f}%, Avg Good Coverage = {avg_good_cov_noop:.1f}%")
    print(f"GREEDY:   Avg Coverage = {avg_cov_greedy:.1f}%, Avg Good Coverage = {avg_good_cov_greedy:.1f}%")

