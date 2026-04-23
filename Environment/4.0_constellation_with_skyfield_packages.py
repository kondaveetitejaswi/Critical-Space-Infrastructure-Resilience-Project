"""
Skyfield-Powered GNSS Constellation Resilience Model
====================================================

A real-world orbital framework for modeling GNSS systems using actual TLE data.
- Downloads live constellation data via CelesTrak
- Replaces idealized circular orbits with the SGP4 propagation model
- Calculates true topocentric visibility using the WGS84 Earth model
- Maintains the generalized MDP logic for resilience analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from enum import Enum
from datetime import datetime
import ssl

# Fix for potential SSL certificate issues when downloading TLEs
ssl._create_default_https_context = ssl._create_unverified_context

from skyfield.api import load, wgs84, EarthSatellite

# =====================================================
# GNSS CONSTELLATION PARAMETERS
# =====================================================

class GNSSConstellationType(Enum):
    GPS = "GPS"
    GALILEO = "Galileo"
    GLONASS = "GLONASS"
    BEIDOU = "BeiDou"

@dataclass
class ConstellationParameters:
    """Generic parameters mapped to real-world TLE data sources."""
    name: str
    constellation_type: GNSSConstellationType
    celestrak_url: str  # URL to live orbital data
    
    # Coverage requirements
    min_satellites_for_fix: int = 4
    min_elevation_angle_deg: float = 5.0
    
    # Operational requirements
    min_coverage_percentage: float = 95.0
    min_good_coverage_percentage: float = 50.0
    
    # Maintenance parameters
    satellite_design_life_years: float = 12.0
    spare_capacity: int = 10
    
    # Resilience metrics
    target_dilution_of_precision: float = 5.0

# Pre-defined sources using CelesTrak active data
CONSTELLATION_CONFIGS = {
    GNSSConstellationType.GPS: ConstellationParameters(
        name="Global Positioning System (GPS)",
        constellation_type=GNSSConstellationType.GPS,
        celestrak_url='https://celestrak.org/NORAD/elements/gps-ops.txt',
    ),
    GNSSConstellationType.GALILEO: ConstellationParameters(
        name="Galileo",
        constellation_type=GNSSConstellationType.GALILEO,
        celestrak_url='https://celestrak.org/NORAD/elements/galileo.txt',
    ),
    GNSSConstellationType.GLONASS: ConstellationParameters(
        name="GLONASS",
        constellation_type=GNSSConstellationType.GLONASS,
        celestrak_url='https://celestrak.org/NORAD/elements/glonass.txt',
        satellite_design_life_years=10.0,
    )
}

# =====================================================
# SKYFIELD GNSS STATE
# =====================================================

@dataclass(frozen=True)
class SkyfieldSatelliteState:
    """Represents a satellite state wrapped around a Skyfield EarthSatellite object."""
    sat_name: str
    satellite_obj: EarthSatellite  # Skyfield object containing TLE/SGP4 data
    health: float
    age_days: float

@dataclass(frozen=True)
class SkyfieldConstellationState:
    """Snapshot of the constellation in astronomical time."""
    satellites: Tuple[SkyfieldSatelliteState, ...]
    params: ConstellationParameters
    current_time: object  # Skyfield Time object
    spares_available: int
    avg_constellation_health: float
    
    def operational_count(self) -> int:
        return sum(1 for sat in self.satellites if sat.health > 0)

# =====================================================
# SKYFIELD RESILIENCE MDP ENGINE
# =====================================================

class SkyfieldGNSSResilienceMDP:
    """
    MDP for GNSS constellation resilience using Skyfield physics.
    """
    
    def __init__(self, params: ConstellationParameters):
        self.params = params
        self.ts = load.timescale()
        
        # Grid setup (15-degree increments to balance speed and global accuracy)
        self.lat_grid = np.arange(-90, 91, 15)
        self.lon_grid = np.arange(-180, 180, 15)
        self.grid_cells = [(lat, lon) for lat in self.lat_grid for lon in self.lon_grid]
        
        # MDP Parameters
        self.health_decay_daily = 0.00005 / params.satellite_design_life_years
        self.weibull_k = 4.5
        self.weibull_eta = params.satellite_design_life_years * 365
        
        self.action_costs = {
            "NO_OP": 0,
            "LAUNCH_SATELLITE": 100,
            "ACTIVATE_SPARE": 5,
            "RETIRE_SATELLITE": 10,
        }
        self.actions = list(self.action_costs.keys())

    def initialize_constellation(self) -> SkyfieldConstellationState:
        """Download live TLEs and build the initial constellation state."""
        print(f"Downloading live TLE data from CelesTrak for {self.params.name}...")
        satellites_data = load.tle_file(self.params.celestrak_url)
        
        states = []
        for sat in satellites_data:
            # Initialize with slight variations in health
            states.append(SkyfieldSatelliteState(
                sat_name=sat.name,
                satellite_obj=sat,
                health=1.0 - abs(np.random.normal(0, 0.02)),
                age_days=0.0
            ))
            
        return SkyfieldConstellationState(
            satellites=tuple(states),
            params=self.params,
            current_time=self.ts.now(),
            spares_available=self.params.spare_capacity,
            avg_constellation_health=1.0
        )

    def compute_coverage(self, state: SkyfieldConstellationState) -> Dict:
        """Calculate global coverage using Skyfield's WGS84 and AltAz algorithms."""
        adequate_cells = 0
        good_cells = 0
        poor_cells = 0
        total_cells = len(self.grid_cells)
        
        t = state.current_time
        active_sats = [s for s in state.satellites if s.health > 0]
        
        if not active_sats:
            return {'overall_coverage_pct': 0.0, 'good_coverage_pct': 0.0, 'service_available': 0}

        for lat, lon in self.grid_cells:
            ground_station = wgs84.latlon(lat, lon)
            n_visible = 0.0
            
            # Compute topocentric elevation for all active satellites
            for sat_state in active_sats:
                difference = sat_state.satellite_obj - ground_station
                topocentric = difference.at(t)
                alt, az, distance = topocentric.altaz()
                
                if alt.degrees >= self.params.min_elevation_angle_deg:
                    p_vis = (alt.degrees - self.params.min_elevation_angle_deg) / (90.0 - self.params.min_elevation_angle_deg)
                    p_vis *= sat_state.health
                    n_visible += min(max(p_vis, 0.0), 0.99)
            
            # PDOP logic based on visible satellites
            if n_visible < self.params.min_satellites_for_fix:
                poor_cells += 1
            else:
                base_pdop = 10.0 / np.sqrt(n_visible)
                pdop = min(max(base_pdop, 1.0), 10.0)
                
                if pdop < 4.0:
                    good_cells += 1
                elif pdop < self.params.target_dilution_of_precision:
                    adequate_cells += 1
                else:
                    poor_cells += 1
                    
        overall_cov = 100.0 * (adequate_cells + good_cells) / total_cells
        good_cov = 100.0 * good_cells / total_cells
        
        return {
            'overall_coverage_pct': overall_cov,
            'good_coverage_pct': good_cov,
            'service_available': 1 if overall_cov >= self.params.min_coverage_percentage else 0,
            'degradation': (0.7 * (overall_cov / 100.0)) + (0.3 * (good_cov / 100.0))
        }

    def sample_next_state(self, state: SkyfieldConstellationState, action: str, delta_days: float = 0.5) -> tuple:
        """Executes one simulation step: applies action, advances real time, and calculates new state."""
        sats_list = list(state.satellites)
        spares = state.spares_available
        
        # 1. Apply Action
        if action == "ACTIVATE_SPARE" and spares > 0:
            for i, sat in enumerate(sats_list):
                if sat.health <= 0:
                    sats_list[i] = SkyfieldSatelliteState(sat.sat_name, sat.satellite_obj, 1.0, 0.0)
                    spares -= 1
                    break
        elif action == "RETIRE_SATELLITE":
            active_idx = [i for i, s in enumerate(sats_list) if s.health > 0]
            if active_idx:
                oldest = max(active_idx, key=lambda i: sats_list[i].age_days)
                sats_list[oldest] = SkyfieldSatelliteState(sats_list[oldest].sat_name, sats_list[oldest].satellite_obj, 0.0, sats_list[oldest].age_days)
                spares += 1
        elif action == "LAUNCH_SATELLITE":
            spares = min(spares + 1, self.params.spare_capacity)
            
        # 2. Advance Astronomical Time (This completely replaces the math engine)
        new_time = self.ts.tt_jd(state.current_time.tt + delta_days)
        
        # 3. Apply Health Degradation and Age
        for i, sat in enumerate(sats_list):
            if sat.health > 0:
                decay = self.health_decay_daily * delta_days
                new_health = max(sat.health - decay - abs(np.random.normal(0, 0.005)), 0.0)
                sats_list[i] = SkyfieldSatelliteState(sat.sat_name, sat.satellite_obj, new_health, sat.age_days + delta_days)
                
        # 4. Calculate KPIs with new time and health
        avg_health = np.mean([s.health for s in sats_list if s.health > 0]) if any(s.health > 0 for s in sats_list) else 0.0
        temp_state = SkyfieldConstellationState(tuple(sats_list), self.params, new_time, spares, avg_health)
        kpis = self.compute_coverage(temp_state)
        
        # 5. Calculate Cost
        cost = self.action_costs[action]
        if kpis['service_available'] == 0:
            cost += 500
        cost += max(0, self.params.min_coverage_percentage - kpis['overall_coverage_pct']) * 2.0
        
        return temp_state, cost, kpis

    def run_simulation(self, policy: str, initial_state: SkyfieldConstellationState, steps: int = 10) -> List[Dict]:
        """Run the simulation loop."""
        state = initial_state
        trajectory = []
        
        for t in range(steps):
            # Simple heuristic greedy action
            action = "NO_OP"
            if policy == "GREEDY":
                if state.avg_constellation_health < 0.85 and state.spares_available > 0:
                    action = "ACTIVATE_SPARE"
                elif state.operational_count() < (len(state.satellites) * 0.9) and state.spares_available < 3:
                    action = "LAUNCH_SATELLITE"

            next_state, _, kpis = self.sample_next_state(state, action, delta_days=0.5) # Simulating 12 hours per step
            
            print(f"Step {t+1:2d} | {self.params.name:7s} | Action: {action:16s} | "
                  f"Active Sats: {next_state.operational_count():2d}/{len(state.satellites)} | "
                  f"Health: {next_state.avg_constellation_health:.2f} | "
                  f"Coverage: {kpis['overall_coverage_pct']:5.1f}%")
            
            trajectory.append(kpis)
            state = next_state
            
        return trajectory

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("=" * 100)
    print("SKYFIELD-POWERED GNSS RESILIENCE SIMULATION")
    print("Using real-time TLE data and SGP4 Orbital Propagation")
    print("=" * 100)
    
    # Let's test GPS and Galileo using real TLE data
    for const_type in [GNSSConstellationType.GPS, GNSSConstellationType.GALILEO]:
        params = CONSTELLATION_CONFIGS[const_type]
        mdp = SkyfieldGNSSResilienceMDP(params)
        
        # Initialize uses live internet to get exact satellite positions
        initial = mdp.initialize_constellation()
        
        print(f"\n--- Running 'NO_OP' (Passive) Scenario for {params.name} ---")
        traj_noop = mdp.run_simulation("NO_OP", initial, steps=10)
        
        print(f"\n--- Running 'GREEDY' (Active Maintenance) Scenario for {params.name} ---")
        traj_greedy = mdp.run_simulation("GREEDY", initial, steps=10)
        
        avg_cov_noop = np.mean([t['overall_coverage_pct'] for t in traj_noop])
        avg_cov_greedy = np.mean([t['overall_coverage_pct'] for t in traj_greedy])
        
        print(f"\nResult: Active maintenance improved {params.name} coverage from {avg_cov_noop:.1f}% to {avg_cov_greedy:.1f}%")
        print("=" * 100)