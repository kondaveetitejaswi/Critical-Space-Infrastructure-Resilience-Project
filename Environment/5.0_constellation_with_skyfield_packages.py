import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from collections import defaultdict
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from skyfield.api import load, wgs84

# =====================================================
# 1. PARAMETERS & STATE DEFINITIONS
# =====================================================

@dataclass
class ConstellationParameters:
    name: str
    celestrak_url: str
    min_satellites_for_fix: int = 4
    min_elevation_angle_deg: float = 5.0
    spare_capacity: int = 5
    design_life_years: float = 12.0
    #attack proababilities
    jamming_prob: float = 0.05
    spoofing_prob: float = 0.03

GPS_PARAMS = ConstellationParameters(
    name="GPS",
    celestrak_url='https://celestrak.org/NORAD/elements/gps-ops.txt'
)

@dataclass(frozen=True)
class SkyfieldSatelliteState:
    sat_name: str
    satellite_obj: object
    health: float
    age_days: float

@dataclass(frozen=True)
class SkyfieldConstellationState:
    satellites: Tuple[SkyfieldSatelliteState, ...]
    current_time: object
    spares_available: int
    active_jammer_location: Tuple[float, float] = None 
    
    def get_operational_count(self) -> int:
        return sum(1 for sat in self.satellites if sat.health > 0.5)
    
    def get_avg_health(self) -> float:
        return np.mean([s.health for s in self.satellites]) if self.satellites else 0.0

# =====================================================
# 2. PHYSICS & KPI ENGINE (COVERAGE, PDOP, C/N0)
# =====================================================

class SkyfieldPhysicsEngine:
    def __init__(self, params: ConstellationParameters):
        self.params = params
        self.ts = load.timescale()
        # Coarser grid for ADP training speed (30-degree increments)
        self.lat_grid = np.arange(-90, 91, 30)
        self.lon_grid = np.arange(-180, 180, 30)
        self.grid_cells = [(lat, lon) for lat in self.lat_grid for lon in self.lon_grid]
        
    def download_initial_state(self) -> SkyfieldConstellationState:
        sats_data = load.tle_file(self.params.celestrak_url)
        states = tuple(SkyfieldSatelliteState(s.name, s, 1.0, 0.0) for s in sats_data)
        return SkyfieldConstellationState(states, self.ts.now(), self.params.spare_capacity)

    def compute_kpis(self, state: SkyfieldConstellationState) -> Dict:
        """Calculates Coverage, Exact PDOP, and Carrier-to-Noise Ratio"""
        adequate_cells, good_cells = 0, 0
        total_cells = len(self.grid_cells)
        global_pdop_list, global_cn0_list = [], []
        
        t = state.current_time
        active_sats = [s for s in state.satellites if s.health > 0.5]

        #jammer settings
        jammer_lat, jammer_lon = state.active_jammer_location if state.active_jammer_location else (None, None)
        jammer_power_v = 1000.0

        for lat, lon in self.grid_cells:
            ground_station = wgs84.latlon(lat, lon)
            H_matrix = []
            cn0_local_list = []

            J_received_dbw = -999.0
            if jammer_lat is not None:
                dist_to_jammer = np.sqrt((lat - jammer_lat) ** 2 + (lon - jammer_lon) ** 2) * 111000.0
                if dist_to_jammer < 1000000:
                    L_jammer = 20 * np.log10(4 * np.pi * dist_to_jammer * 1.57542e9 / 299792458.0)
                    J_received_dbw = jammer_power_v - L_jammer
            
            for sat_state in active_sats:
                difference = sat_state.satellite_obj - ground_station
                topocentric = difference.at(t)
                alt, az, dist = topocentric.altaz()
                
                if alt.degrees >= self.params.min_elevation_angle_deg:
                    # 1. Geometric DOP Calculation (Design Matrix H)
                    el_rad = alt.radians
                    az_rad = az.radians
                    # Standard transformation to line-of-sight unit vector
                    H_matrix.append([-np.cos(el_rad)*np.sin(az_rad), 
                                     -np.cos(el_rad)*np.cos(az_rad), 
                                     -np.sin(el_rad), 1.0])
                    
                    # 2. C/N0 Calculation using Friis Equation
                    d_m = dist.m
                    f = 1575.42e6  # GPS L1 Frequency
                    c = 299792458.0 # Speed of light
                    # Free Space Path Loss
                    Ls = 20 * np.log10(4 * np.pi * d_m * f / c)
                    EIRP = 26.8 # Typical GPS EIRP in dBW
                    kT = -204.0 # Thermal noise density in dBW/Hz
                    C_received_dbw = EIRP - Ls 
                    
                    # calculate J/S if jammer active
                    if J_received_dbw > -999:
                        effective_noise = kT + J_received_dbw
                        cn0 = C_received_dbw - effective_noise
                    else:
                        cn0 = C_received_dbw - kT

                    if cn0 > 28.0:
                        el_rad, az_rad = alt.radians, az.radians
                        # H_matrix.append([-np.cos(el_rad)*np.sin(az_rad), -np.cos(el_rad)*np.cos(az_rad), -np.sin(el_rad), 1.0])
                        cn0_local_list.append(cn0)
            
            # KPI Aggregation per cell
            n_visible = len(H_matrix)
            cell_pdop = 99.9
            
            if n_visible >= self.params.min_satellites_for_fix:
                H = np.array(H_matrix)
                try:
                    Q = np.linalg.inv(H.T @ H) # Covariance matrix
                    cell_pdop = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
                except np.linalg.LinAlgError:
                    pass
                    
            global_pdop_list.append(cell_pdop)
            if cn0_local_list: global_cn0_list.extend(cn0_local_list)
            
            # Fixed Coverage check
            if n_visible >= self.params.min_satellites_for_fix:
                adequate_cells += 1
                if cell_pdop < 6.0: good_cells += 1

        cov_pct = (adequate_cells / total_cells) * 100.0
        return {
            'coverage_pct': cov_pct,
            'avg_pdop': np.mean(global_pdop_list),
            'avg_cn0_dbhz': np.mean(global_cn0_list) if global_cn0_list else 0.0
        }
    
    def get_valid_Actions(self, state:SkyfieldConstellationState) -> List[str]:
        valid_actions = ["NO_OP"]
        sats = state.satellites
        spares = state.spares_available
        if any(0.0 < s.health < 0.4 for s in sats):
            valid_actions.append("RETIRE_SATELLITE")
        if spares>0 and any(s.health <= 0.0 for s in sats):
            valid_actions.append("ACTIVATE_SPARE")
        if spares < self.params.spare_capacity:
            valid_actions.append("LAUNCH_SATELLITE")
        if spares == 0 and any(s.health <= 0.0 for s in sats) and sum(1 for s in sats if s.health > 0.8) > 5:
            valid_actions.append("REBALANCE_ORBITS")

        return valid_actions
    
    def apply_action(self, state:SkyfieldConstellationState, action: str) -> Tuple[list, int, float, float]:
        sats = list(state.satellites)
        spares = state.spares_available
        cost = 0.0
        cov_boost = 0.0

        if action == "RETIRE_SATELLITE":
            candidates = [i for i, s in enumerate(sats) if 0.0 < s.health < 0.4]
            if candidates:
                worst_idx = min(candidates, key = lambda i: sats[i].health)
                s = sats[worst_idx]
                sats[worst_idx] = SkyfieldSatelliteState(s.sat_name, s.satellite_obj, 0.0, s.age_days)
                cost = 5.0
                print(f" Retiring Satellite: {s.sat_name} with health {s.health:.2f}")

        elif action == "ACTIVATE_SPARE":
            dead_indices = [i for i, s in enumerate(sats) if s.health <= 0.0]
            if dead_indices and spares > 0:
                best_idx = dead_indices[0] 
                s = sats[best_idx]
                sats[best_idx] = SkyfieldSatelliteState(s.sat_name, s.satellite_obj, 1.0, 0.0)
                spares -= 1
                cost = 20.0
                print(f" Activating Spare for Satellite: {s.sat_name} | Spares left: {spares}")
        elif action == "LAUNCH_SATELLITE":
            spares = min(spares +1, self.params.spare_capacity)
            cost = 150.0
            print(f" Launching New Satellite | Spares available: {spares}")
        elif action == "REBALANCE_ORBITS":
            healthy_indices = [i for i, s in enumerate(sats) if s.health > 0.8]
            if healthy_indices:
                donor_idx = healthy_indices[0]
                s = sats[donor_idx]
                sats[donor_idx] = SkyfieldSatelliteState(s.sat_name, s.satellite_obj, s.health - 0.3, s.age_days)
                cost = 50.0
                cov_boost = 10.0
                print(f" Rebalancing Orbits: Donor {s.sat_name} health reduced to {sats[donor_idx].health:.2f} | Coverage boost: {cov_boost}%")
        return sats, spares, cost, cov_boost
           


    def step_physics(self, state: SkyfieldConstellationState, action: str, delta_days: float = 1.0) -> Tuple[SkyfieldConstellationState, float, bool]:
        """Applies action, attacks, natural degradation, and advances time."""
        sats, spares, action_cost, cov_boost = self.apply_action(state, action)
        jammer_loc = None
        under_attack = False
        
        new_time = self.ts.tt_jd(state.current_time.tt + delta_days)
        decay = (0.05 / (self.params.design_life_years * 365)) * delta_days

        for i, s in enumerate(sats):
            if s.health >0:
                new_health = max(s.health - decay - abs(np.random.normal(0, 0.005)), 0.0)
                sats[i] = SkyfieldSatelliteState(s.sat_name, s.satellite_obj, new_health, s.age_days + delta_days)
        
        if np.random.rand() < self.params.jamming_prob:
            j_lat = np.random.choice(self.lat_grid)
            j_lon = np.random.choice(self.lon_grid)
            jammer_loc = (j_lat, j_lon)
            under_attack = True
            print(f"Jamming Attack at Lat: {j_lat}, Lon: {j_lon}")

        if np.random.rand() < self.params.spoofing_prob:
            num_spoofed = np.random.randint(1, 4)
            spoof_targets = np.random.choice(len(sats), num_spoofed, replace=False)
            for idx in spoof_targets:
                s = sats[idx]
                sats[idx] =SkyfieldSatelliteState(s.sat_name, s.satellite_obj, 0.1, s.age_days)
            under_attack = True
            print(f"Spoofing Attack on Satellites: {[sats[idx].sat_name for idx in spoof_targets]}")


        new_state = SkyfieldConstellationState(tuple(sats), new_time, spares, active_jammer_location = jammer_loc)
        
        # Compute Reward (Maximization Target)
        kpis = self.compute_kpis(new_state)
        
        final_cov = min(kpis['coverage_pct'] + cov_boost, 100.0)
        # final_cov = kpis['coverage_pct'] + cov_boost
        # Reward = Big bonus for coverage + slight bonus for signal quality - maintenance cost
        reward = (kpis['coverage_pct'] * 10.0) - action_cost
        
        if final_cov < 90.0:
            reward -= 500

        kpis['coverage_pct'] = final_cov

        return new_state, reward, kpis,under_attack

# =====================================================
# 3. ADP SOLVER (FORWARD PASS & BACKWARD TD UPDATE)
# =====================================================

class SkyfieldADPSolver:
    def __init__(self, physics: SkyfieldPhysicsEngine):
        self.physics = physics
        self.actions = ["NO_OP", "ACTIVATE_SPARE", "LAUNCH_SATELLITE", "RETIRE_SATELLITE", "REBALANCE_ORBITS"]
        self.gamma = 0.95
        self.alpha = 0.1 # Learning rate
        self.post_decision_values = defaultdict(float) # V(S^a)
        
    def get_macro_state(self, state: SkyfieldConstellationState, coverage_pct: float, avg_cn0 =0) -> tuple:
        """Discretizes the continuous physics universe for Tabular ADP."""
        oc = state.get_operational_count()
        sp = state.spares_available
        h_level = round(state.get_avg_health() * 5) # Discretize health to 0-5
        cov_status = 1 if coverage_pct >= 90.0 else 0
        threat_active = 1 if state.active_jammer_location else 0
        cn0_level = 0 if avg_cn0 < 30 else (1 if avg_cn0 < 40 else 2)
        return (oc, sp, h_level, cov_status, threat_active, cn0_level)

    def get_post_decision_state(self, macro_state: tuple, action: str) -> tuple:
        """Deterministic transition mapping based on ADP rules"""
        oc, sp, h, cov, threat, cn0 = macro_state
        if action == "ACTIVATE_SPARE" and sp > 0:
            return (oc + 1, sp - 1, h, cov, "PDS_ACTIVATE")
        elif action == "LAUNCH_SATELLITE":
            return (oc, sp + 1, h, cov, "PDS_LAUNCH")
        return (oc, sp, h, cov, "PDS_NO_OP")

    def greedy_action(self, macro_state: tuple, physical_state: SkyfieldConstellationState) -> str:
        """v_t(S) = argmax_a [V(PDS(S, a))]"""
        valid_actions = self.physics.get_valid_Actions(physical_state)
        best_actions = "NO_OP"
        best_val = -float('inf')
        for act in valid_actions:
            pds = self.get_post_decision_state(macro_state, act)
            val = self.post_decision_values[pds]
            if val > best_val:
                best_val = val
                best_actions = act
        return best_actions

    def run_adp_training(self, initial_state: SkyfieldConstellationState, iterations: int = 5):
        print("\n" + "="*70)
        print("ADP TRAINING: FORWARD PASS & BACKWARD TD UPDATES")
        print("="*70)
        
        # Initial KPI check to establish base coverage
        base_kpis = self.physics.compute_kpis(initial_state)
        print(f"Initial System Check -> Coverage: {base_kpis['coverage_pct']:.1f}%, C/N0: {base_kpis['avg_cn0_dbhz']:.1f} dB-Hz, PDOP: {base_kpis['avg_pdop']:.1f}")

        for it in range(iterations):
            state = initial_state
            kpis = base_kpis
            trajectory = []
            total_reward = 0
            
            # --- FORWARD PASS (Generate Trajectory) ---
            for step in range(10): 
                macro_state = self.get_macro_state(state, kpis['coverage_pct'])
                
                # Epsilon-greedy exploration
                if np.random.rand() < 0.2: action = np.random.choice(self.physics.get_valid_Actions(state))
                else: action = self.greedy_action(macro_state, state)
                
                pds = self.get_post_decision_state(macro_state, action)
                
                # Execute in physical environment
                next_state, reward, next_kpis, attack_flag = self.physics.step_physics(state, action)
                next_macro = self.get_macro_state(next_state, next_kpis['coverage_pct'])
                
                trajectory.append((macro_state, action, pds, reward, next_macro))
                total_reward += reward
                state = next_state
                kpis = next_kpis
                
            print(f"Iteration {it+1:2d} | Trajectory Reward: {total_reward:6.1f} | Final Cov: {kpis['coverage_pct']:.1f}% | Final PDOP: {kpis['avg_pdop']:.1f}")

            # --- BACKWARD PASS (TD Learning on Post-Decision States) ---
            # V(PDS_t) = (1-α) * V(PDS_t) + α * (Reward_t + γ * max_a V(PDS_{t+1}))
            for t in range(len(trajectory) - 1, -1, -1):
                macro_state, action, pds, reward, next_macro = trajectory[t]
                
                # Get max future value from next state
                future_value = max(self.post_decision_values[self.get_post_decision_state(next_macro, a)] for a in self.actions)
                
                target = reward + self.gamma * future_value
                current_v = self.post_decision_values[pds]
                self.post_decision_values[pds] = (1 - self.alpha) * current_v + self.alpha * target

# =====================================================
# EXECUTION
# =====================================================
if __name__ == "__main__":
    physics = SkyfieldPhysicsEngine(GPS_PARAMS)
    
    print("Downloading live Ephemeris data...")
    initial_state = physics.download_initial_state()
    
    adp = SkyfieldADPSolver(physics)
    adp.run_adp_training(initial_state, iterations=500
                         )
    
    print("\nADP Learning Complete. Learned Values for Post Decision States:")
    for pds, val in sorted(adp.post_decision_values.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"PDS State {pds}: Value = {val:.2f}")