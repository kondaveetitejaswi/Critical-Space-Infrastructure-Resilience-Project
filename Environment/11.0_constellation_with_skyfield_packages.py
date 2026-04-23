# the version with epsilon decay with exponential degradation
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
import ssl, csv, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
from skyfield.api import load, wgs84


@dataclass
class ConstellationParameters:
    name: str
    celestrak_url: str
    min_satellites_for_fix: int = 4
    min_elevation_angle_deg: float = 5.0
    spare_capacity: int = 8
    design_life_years: float = 12.0
    launch_delay_steps: int = 7

GPS_PARAMS = ConstellationParameters(
    name="GPS",
    celestrak_url='https://celestrak.org/NORAD/elements/gps-ops.txt'
)

def get_threat_phase(episode_step: int, max_steps: int = 500) -> str:
    if episode_step <= max_steps * 0.33:
        return "LOW"
    elif episode_step <= max_steps * 0.66:
        return "MEDIUM"
    else:
        return "HIGH"

@dataclass(frozen=True)
class SkyfieldSatelliteState:
    sat_name:       str
    satellite_obj:  object
    health:         float
    age_days:       float
    orbital_plane:  int   = 0
    launch_countdown: int = 0  
    is_spoofed:     bool  = False 

@dataclass(frozen=True)
class SkyfieldConstellationState:
    satellites:             Tuple
    current_time:           object
    spares_available:       int
    active_jammer_location: Optional[Tuple[float,float]] = None
    jammer_persistence_steps: int  = 0
    threat_phase:           str    = "LOW"
    pending_launches:       Tuple  = () 
    was_under_attack:       bool   = False

    def get_operational_count(self) -> int:
        return sum(1 for s in self.satellites
                   if s.health > 0.5 and s.launch_countdown == 0)

    def get_avg_health(self) -> float:
        vals = [s.health for s in self.satellites if s.health > 0]
        return float(np.mean(vals)) if vals else 0.0




class SkyfieldPhysicsEngine:
    def __init__(self, params: ConstellationParameters):
        self.params = params
        self.ts = load.timescale()
        self.lat_grid  = np.arange(-90,  91, 30)
        self.lon_grid  = np.arange(-180, 180, 30)
        self.grid_cells = [(la, lo) for la in self.lat_grid for lo in self.lon_grid]

    def download_initial_state(self) -> SkyfieldConstellationState:
        sats_data = load.tle_file(self.params.celestrak_url)
        states = [
            SkyfieldSatelliteState(
                sat_name=s.name, satellite_obj=s,
                health=1.0, age_days=0.0,
                orbital_plane=i % 6,
                launch_countdown=0, is_spoofed=False)
            for i, s in enumerate(sats_data)
        ]
        return SkyfieldConstellationState(
            satellites=tuple(states),
            current_time=self.ts.now(),
            spares_available=self.params.spare_capacity)

    def compute_kpis(self, state: SkyfieldConstellationState) -> Dict:
        adequate, good = 0, 0
        total = len(self.grid_cells)
        pdop_list, cn0_list = [], []
        t = state.current_time

        active = [s for s in state.satellites
                  if s.health > 0.5 and s.launch_countdown == 0]

        jlat, jlon = (state.active_jammer_location
                      if state.active_jammer_location else (None, None))
        jammer_power_w = 50000.0

        for lat, lon in self.grid_cells:
            gs = wgs84.latlon(lat, lon)
            H, cn0_cell = [], []

            J_dbw = -999.0
            if jlat is not None:
                d = max(np.sqrt((lat-jlat)**2+(lon-jlon)**2)*111000.0, 1.0)
                if d < 2000000:
                    Lj = 20*np.log10(4*np.pi*d*1.57542e9/299792458.0)
                    J_dbw = 10*np.log10(jammer_power_w) - Lj

            for s in active:
                top = (s.satellite_obj - gs).at(t)
                alt, az, dist = top.altaz()
                if alt.degrees >= self.params.min_elevation_angle_deg:
                    el, az_ = alt.radians, az.radians
                    H.append([-np.cos(el)*np.sin(az_),
                               -np.cos(el)*np.cos(az_),
                               -np.sin(el), 1.0])
                    Ls   = 20*np.log10(4*np.pi*dist.m*1575.42e6/299792458.0)
                    C    = 26.8 - Ls
                    kT   = -204.0
                    if J_dbw > -999:
                        noise = 10*np.log10(10**(kT/10)+10**(J_dbw/10))
                        cn0_cell.append(C - noise)
                    else:
                        cn0_cell.append(C - kT)

            nv = len(H)
            pdop = 99.9
            if nv >= self.params.min_satellites_for_fix:
                try:
                    Hm = np.array(H)
                    Q  = np.linalg.inv(Hm.T @ Hm)
                    pdop = np.sqrt(Q[0,0]+Q[1,1]+Q[2,2])
                except np.linalg.LinAlgError:
                    pass
                adequate += 1
                if pdop < 6.0:
                    good += 1

            pdop_list.append(pdop)
            cn0_list.extend(cn0_cell)

        return {
            'coverage_pct':      (adequate/total)*100.0,
            'good_coverage_pct': (good/total)*100.0,
            'avg_pdop':          float(np.mean(pdop_list)),
            'avg_cn0_dbhz':      float(np.mean(cn0_list)) if cn0_list else 0.0,
            'operational_sats':  len(active)
        }

    def get_valid_actions(self, state: SkyfieldConstellationState) -> List[str]:
        valid = ["NO_OP"]
        valid += ["ACTIVATE_SPARE", "LAUNCH_SATELLITE",
                  "RETIRE_SATELLITE", "REBALANCE_ORBITS"]
        return valid

    def apply_action(self, state, action):
        sats    = list(state.satellites)
        spares  = state.spares_available
        pending = list(state.pending_launches)
        cost, cov_boost = 0.0, 0.0

        if action == "RETIRE_SATELLITE":
            cands = [i for i,s in enumerate(sats)
                     if 0.0 < s.health < 0.4 and s.launch_countdown == 0]
            if cands:
                idx = min(cands, key=lambda i: sats[i].health)
                s = sats[idx]
                sats[idx] = SkyfieldSatelliteState(
                    s.sat_name, s.satellite_obj, 0.0, s.age_days,
                    s.orbital_plane, 0, False)
                cost = 5.0
                print(f"  [ACTION] RETIRE  -> {s.sat_name} (health {s.health:.2f})")

        elif action == "ACTIVATE_SPARE":
            dead = [i for i,s in enumerate(sats) if s.health <= 0.0]
            if dead and spares > 0:
                idx = dead[0]; s = sats[idx]
                sats[idx] = SkyfieldSatelliteState(
                    s.sat_name, s.satellite_obj, 1.0, 0.0,
                    s.orbital_plane, 0, False)
                spares -= 1; cost = 5.0
                print(f"  [ACTION] ACTIVATE SPARE -> {s.sat_name} | Pool: {spares}")

        elif action == "LAUNCH_SATELLITE":
            pending.append(self.params.launch_delay_steps)
            cost = 10.0
            print(f"  [ACTION] LAUNCH -> In-transit ({self.params.launch_delay_steps} steps) | "
                  f"In-flight: {len(pending)}")

        elif action == "REBALANCE_ORBITS":
            healthy = [i for i,s in enumerate(sats)
                       if s.health > 0.8 and s.launch_countdown == 0]
            if healthy:
                idx = healthy[0]; s = sats[idx]
                sats[idx] = SkyfieldSatelliteState(
                    s.sat_name, s.satellite_obj, s.health-0.25, s.age_days,
                    s.orbital_plane, 0, False)
                cost, cov_boost = 15.0, 8.0
                print(f"  [ACTION] REBALANCE -> Donor {s.sat_name} | Boost: {cov_boost}%")

        return sats, spares, cost, cov_boost, tuple(pending)


    def _apply_attacks(self, sats, iteration):
        phase = get_threat_phase(iteration)
        attack_log, jammer_loc, persist = [], None, 0
        active_idx = [i for i,s in enumerate(sats)
                      if s.health > 0.5 and s.launch_countdown == 0]
        roll = np.random.rand() 

        if phase == "LOW":
            if roll < 0.20 and len(active_idx) >= 1:
                n = np.random.randint(1, min(4, len(active_idx)+1))
                targets = np.random.choice(active_idx, n, replace=False)
                for idx in targets:
                    s = sats[idx]
                    sats[idx] = SkyfieldSatelliteState(
                        s.sat_name, s.satellite_obj, np.random.uniform(0.01, 0.1), s.age_days,
                        s.orbital_plane, s.launch_countdown, True)
                attack_log.append(
                    f"[LOW  ] Spoof -> {[sats[i].sat_name for i in targets]}")
            elif roll < 0.13:
                jammer_loc = (float(np.random.choice(self.lat_grid)),
                              float(np.random.choice(self.lon_grid)))
                persist = 2
                attack_log.append(f"[LOW  ] Jam @ {jammer_loc} (2 steps)")

        elif phase == "MEDIUM":
            # 15% coordinated spoof (4-6 sats), 15% persistent jam
            if roll < 0.60 and len(active_idx) >= 4:
                hi = min(6, len(active_idx)); lo = min(4, hi)
                n = np.random.randint(lo, hi+1)
                targets = np.random.choice(active_idx, n, replace=False)
                for idx in targets:
                    s = sats[idx]
                    sats[idx] = SkyfieldSatelliteState(
                        s.sat_name, s.satellite_obj, np.random.uniform(0.01, 0.1), s.age_days,
                        s.orbital_plane, s.launch_countdown, True)
                attack_log.append(
                    f"[MED  ] Coordinated Spoof -> {n} sats: "
                    f"{[sats[i].sat_name for i in targets]}")
            elif roll < 0.30:
                jammer_loc = (float(np.random.choice(self.lat_grid)),
                              float(np.random.choice(self.lon_grid)))
                persist = 3
                attack_log.append(f"[MED  ] Persistent Jam @ {jammer_loc} (3 steps)")

        elif phase == "HIGH":
            if roll < 0.08 and len(active_idx) >= 3:
                plane = np.random.randint(0, 6)
                pidx  = [i for i in active_idx if sats[i].orbital_plane == plane]
                if pidx:
                    for idx in pidx:
                        s = sats[idx]
                        sats[idx] = SkyfieldSatelliteState(
                            s.sat_name, s.satellite_obj, 0.02, s.age_days,
                            s.orbital_plane, s.launch_countdown, True)
                    attack_log.append(
                        f"[HIGH ] Orbital Plane {plane} -> {len(pidx)} sats disabled")
            elif roll < 0.186:
                degraded = [i for i,s in enumerate(sats)
                            if 0.0 < s.health < 0.4 and s.launch_countdown == 0]
                if degraded:
                    deg = np.random.choice(degraded)
                    neighbors = [i for i in active_idx
                                 if i != deg and sats[i].health > 0.45]
                    if neighbors:
                        v = np.random.choice(neighbors); s = sats[v]
                        nh = max(s.health - 0.50, 0.01)
                        sats[v] = SkyfieldSatelliteState(
                            s.sat_name, s.satellite_obj, nh, s.age_days,
                            s.orbital_plane, s.launch_countdown, True)
                        attack_log.append(
                            f"[HIGH ] Cascade -> {sats[v].sat_name} (h={nh:.2f})")
            elif roll < 0.33:
                jammer_loc = (float(np.random.choice(self.lat_grid)),
                              float(np.random.choice(self.lon_grid)))
                persist = 4
                attack_log.append(f"[HIGH ] Severe Jam @ {jammer_loc} (4 steps)")

        return sats, jammer_loc, persist, attack_log

    def is_plane_crisis(self, state) -> bool:
        planes = defaultdict(list)
        for s in state.satellites:
            planes[s.orbital_plane].append(s.health)
        
        for health_list in planes.values():
            operational_in_plane = sum(1 for h in health_list if h > 0.5)
            if operational_in_plane < 3:
                return True
        return False

    def step_physics(self, state, action, iteration=1, delta_days=7.0):
        sats, spares, action_cost, cov_boost, pending = \
            self.apply_action(state, action)

        new_time = self.ts.tt_jd(state.current_time.tt + delta_days)

        decay = (0.05 / (self.params.design_life_years * 365)) * delta_days

        # WEIBULL PHM PARAMETERS
        beta = 1.5  # Shape parameter (>1 indicates wear-out over time)
        eta = self.params.design_life_years * 365.0  # Scale parameter (design life in days)

        for i, s in enumerate(sats):
            if s.launch_countdown > 0:
                # In-transit: just tick down, no aging
                sats[i] = SkyfieldSatelliteState(
                    s.sat_name, s.satellite_obj, s.health, s.age_days,
                    s.orbital_plane, s.launch_countdown - 1, False)
                continue

            if s.health <= 0.0:
                continue

            # 1. Calculate Weibull Failure Probability
            age_days = s.age_days
            new_age = age_days + delta_days

            if age_days == 0:
                base_pf = 1.0 - np.exp(- (new_age / eta)**beta)
            else:
                base_pf = 1.0 - np.exp(- ((new_age / eta)**beta - (age_days / eta)**beta))

            # Proportional Hazard Multiplier: Stress from attacks increases failure likelihood
            phm_multiplier = 2.5 if s.is_spoofed else 1.0
            exponent = - ((new_age / eta)**beta - (age_days / eta)**beta) * phm_multiplier
            prob_failure = 1.0 - np.exp(exponent)

            # Evaluate failure event
            if np.random.rand() < prob_failure:
                new_health = 0.0  # Satellite fails due to hazard
                still_spoofed = False
            else:
                # Normal operational noise
                noise = abs(np.random.normal(0, 0.001))
                new_health = max(s.health - noise, 0.0)
                
                # 2. Stronger Repair Logic (Resilience)
                still_spoofed = s.is_spoofed
                if s.is_spoofed and new_health > 0.0:
                    # Faster recovery: +0.15 per step (repairs in ~3-4 weeks instead of months)
                    new_health = max(new_health - 0.02, 0.0)
                    still_spoofed = (new_health < 0.80)

            sats[i] = SkyfieldSatelliteState(
                s.sat_name, s.satellite_obj, new_health,
                s.age_days + delta_days, s.orbital_plane, 0, still_spoofed)

        #Tick pending launch countdowns; deliver when they reach 0
        new_pending = []
        for cd in pending:
            ticked = cd - 1
            if ticked <= 0:
                spares = min(spares + 1, self.params.spare_capacity)
                print(f"  [DELIVERY] Satellite arrived! Spare pool: {spares}")
            else:
                new_pending.append(ticked)

        # Jammer persistence from previous step
        jammer_loc  = state.active_jammer_location
        persist_rem = max(state.jammer_persistence_steps - 1, 0)
        if persist_rem == 0:
            jammer_loc = None

        # apply_attacks uses single roll -> at most one attack per step
        sats, new_jammer, new_persist, attack_log = \
            self._apply_attacks(sats, iteration)
        if new_jammer is not None:
            jammer_loc  = new_jammer
            persist_rem = new_persist

        new_state = SkyfieldConstellationState(
            satellites=tuple(sats),
            current_time=new_time,
            spares_available=spares,
            active_jammer_location=jammer_loc,
            jammer_persistence_steps=persist_rem,
            threat_phase=get_threat_phase(iteration),
            pending_launches=tuple(new_pending),
            was_under_attack = (len(attack_log) > 0))

        kpis = self.compute_kpis(new_state)
        final_cov = min(kpis['coverage_pct'] + cov_boost, 100.0)
        kpis['coverage_pct'] = final_cov


        reward = 0.0
        crisis_before = self.is_plane_crisis(state)
        crisis_after = self.is_plane_crisis(new_state)

        if len(attack_log) > 0:
            reward -= 200.0   # penalty for being attacked (system stress)

        # Detect recovery (coverage improvement after attack)
        if hasattr(self, 'prev_coverage'):
            delta_cov = final_cov - self.prev_coverage

            if len(attack_log) > 0:
                if delta_cov > 1.0:
                    reward += 300.0
                elif delta_cov < -1.0:
                    reward -= 200.0

        # store for next step
        self.prev_coverage = final_cov

        # state based rewards
        if final_cov >= 95.0:
            reward += 100.0
        elif final_cov < 90.0:
            reward -= 250.0

        if kpis['avg_pdop'] > 6.0:
            reward -=150.0

        # action evaluation
        if action == "NO_OP":
            if len(attack_log) > 0:
                reward -= 300.0

            elif crisis_before or final_cov < 92.0:
                reward -=300.0
            else:
                reward += 2.0

        elif action == "REBALANCE_ORBITS":
            if crisis_before:
                reward += 400.0
            if cov_boost > 0:
                reward += 100.0

            if not crisis_before and cov_boost <= 0:
                reward -= 100.0

        elif action == "ACTIVATE_SPARE":
            if kpis['operational_sats'] < 24 or crisis_before:
                reward += 150.0

            else: 
                reward -= 100.0

        elif action == "LAUNCH_SATELLITE":
            if spares < (self.params.spare_capacity // 2):
                reward += 100.0
            else:
                reward -= 50.0

        reward -= action_cost
        return new_state, reward, kpis, len(attack_log) > 0, attack_log


# ADP SOLVER

class SkyfieldADPSolver:
    def __init__(self, physics: SkyfieldPhysicsEngine):
        self.physics   = physics
        self.actions   = ["NO_OP","ACTIVATE_SPARE","LAUNCH_SATELLITE",
                          "RETIRE_SATELLITE","REBALANCE_ORBITS"]
        self.gamma     = 0.95
        self.alpha     = 0.1
        self.post_decision_values = defaultdict(float)
        self.training_log = []
        self.episode_log = []

    def get_macro_state(self, state, kpis) -> tuple:
        oc   = state.get_operational_count()
        sp   = min(state.spares_available, self.physics.params.spare_capacity)
        h    = round(state.get_avg_health() * 4)          # 0-4
        cov  = (0 if kpis['coverage_pct'] < 80
                else 1 if kpis['coverage_pct'] < 95 else 2)
        jam  = 1 if state.active_jammer_location else 0
        pdop = (0 if kpis['avg_pdop'] < 3
                else 1 if kpis['avg_pdop'] < 6 else 2)
        ph   = {"LOW":0,"MEDIUM":1,"HIGH":2}.get(state.threat_phase, 0)
        pend = min(len(state.pending_launches), 2)
        recent_attack = 1 if state.was_under_attack else 0

        planes = defaultdict(list)
        for s in state.satellites:
            planes[s.orbital_plane].append(s.health)

        plane_crisis = 1 if any(sum(1 for hp in h_list if hp > 0.5) < 3 for h_list in planes.values()) else 0
        return (oc, sp, h, cov, jam, pdop, ph, pend, plane_crisis, recent_attack)

    def get_post_decision_state(self, macro, action) -> tuple:
        oc,sp,h,cov,jam,pdop,ph,pend,plane_crisis,recent_attack = macro
        if   action == "ACTIVATE_SPARE"   and sp > 0:
            return (oc+1, sp-1, h, cov, jam, pdop, ph, pend, plane_crisis, recent_attack, "ACT")
        elif action == "LAUNCH_SATELLITE":
            return (oc,   sp,   h, cov, jam, pdop, ph, min(pend+1,2), plane_crisis, recent_attack, "LCH")
        elif action == "RETIRE_SATELLITE":
            return (max(oc-1,0), sp, h, cov, jam, pdop, ph, pend, plane_crisis, recent_attack, "RET")
        elif action == "REBALANCE_ORBITS":
            return (oc, sp, h, min(cov+1,2), jam, pdop, ph, pend, 0, recent_attack, "RBL")
        return (oc, sp, h, cov, jam, pdop, ph, pend, plane_crisis, recent_attack, "NOP")

    def greedy_action(self, macro, state) -> str:
        valid = self.physics.get_valid_actions(state)
        best_act, best_val = "NO_OP", -float('inf')
        for act in valid:
            v = self.post_decision_values[self.get_post_decision_state(macro, act)]
            if v > best_val:
                best_val, best_act = v, act
        return best_act

    def run_adp_training(self, initial_state, episodes=500, steps_per_episode=150, output_dir="."):
        print("\n" + "="*70)
        print("  ADP RESILIENCE FRAMEWORK - GPS CONSTELLATION")
        print("  EPISODIC TRAINING ENABLED (Pure State-Driven Reward)")
        print("="*70)

        base_kpis = self.physics.compute_kpis(initial_state)
        print(f"\nBaseline -> Cov: {base_kpis['coverage_pct']:.1f}% | "
              f"PDOP: {base_kpis['avg_pdop']:.2f} | "
              f"C/N0: {base_kpis['avg_cn0_dbhz']:.1f} dB-Hz | "
              f"Op.Sats: {base_kpis['operational_sats']}\n")

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "resilience_log.csv")

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            cw = csv.writer(f)
            cw.writerow(['episode', 'step', 'phase', 'action',
                         'coverage_pct', 'good_coverage_pct',
                         'avg_pdop', 'avg_cn0_dbhz',
                         'operational_sats', 'spares', 'pending_launches',
                         'step_reward', 'under_attack', 'attack_description'])


            for ep in range(1, episodes + 1):
                state = initial_state
                kpis = base_kpis
                trajectory = []
                episode_reward = 0.0
                self.physics.prev_coverage = base_kpis['coverage_pct']

                ep_coverage = []
                ep_pdops = []
                ep_cn0s = []
                
                decay_at_step = 0.003
                eps = max(0.1, 0.5 * np.exp(-decay_at_step * ep))

                self.training_log = []

                print(f"\n{'-'*70}")
                print(f"  > STARTING EPISODE {ep}/{episodes} (Epsilon: {eps:.2f})")
                print(f"{'-'*70}")

                for step in range(1, steps_per_episode + 1):
                    macro = self.get_macro_state(state, kpis)
                    
                    if np.random.rand() < eps:
                        action = np.random.choice(self.physics.get_valid_actions(state))
                    else:
                        action = self.greedy_action(macro, state)
                    
                    pds = self.get_post_decision_state(macro, action)

                    next_state, reward, next_kpis, atk, atk_log = \
                        self.physics.step_physics(state, action, iteration=step, delta_days=7.0)

                    next_macro = self.get_macro_state(next_state, next_kpis)
                    trajectory.append((macro, action, pds, reward, next_macro))
                    
                    episode_reward += reward

                    ep_coverage.append(next_kpis['coverage_pct'])
                    ep_pdops.append(next_kpis['avg_pdop'])
                    ep_cn0s.append(next_kpis['avg_cn0_dbhz'])
                    
                    phase = get_threat_phase(step, steps_per_episode) 
                    
                    if step % 10 == 0 or atk:
                        cov_f  = " !! LOW"  if next_kpis['coverage_pct'] < 90 else ""
                        rew_f  = " !! NEG"  if reward < 0 else ""
                        print(f"  Ep {ep:3d} | Step {step:3d} [{phase:6s}] | "
                              f"Rew: {reward:6.1f}{rew_f} | "
                              f"Cov: {next_kpis['coverage_pct']:5.1f}%{cov_f} | "
                              f"Act: {action}")
                        if atk_log:
                            for line in atk_log: print(f"    {line}")

                    cw.writerow([ep, step, phase, action,
                                 round(next_kpis['coverage_pct'], 2),
                                 round(next_kpis['good_coverage_pct'], 2),
                                 round(next_kpis['avg_pdop'], 4),
                                 round(next_kpis['avg_cn0_dbhz'], 2),
                                 next_kpis['operational_sats'],
                                 next_state.spares_available,
                                 len(next_state.pending_launches),
                                 round(reward, 2), int(atk),
                                 "; ".join(atk_log) if atk_log else "None"])

                    self.training_log.append({
                        'iteration': step, 
                        'phase': phase,
                        'coverage':  next_kpis['coverage_pct'],
                        'pdop':      next_kpis['avg_pdop'],
                        'cn0':       next_kpis['avg_cn0_dbhz'],
                        'reward':    reward, 
                        'op_sats':   next_kpis['operational_sats'],
                        'spares':    next_state.spares_available,
                        'pending':   len(next_state.pending_launches),
                        'under_attack': atk,
                        'action':    action})

                    # Advance state
                    state = next_state
                    kpis = next_kpis

                    if kpis['coverage_pct'] < 50.0:
                        print(f"  [CRITICAL FAILURE] Constellation collapsed at step {step}. Resetting episode.")
                        break

                for t in range(len(trajectory)-1, -1, -1):
                    _, _, pds, r, nm = trajectory[t]
                    future = max(
                        self.post_decision_values[self.get_post_decision_state(nm, a)]
                        for a in self.actions)
                    target = r + self.gamma * future
                    old_v  = self.post_decision_values[pds]
                    
                    self.post_decision_values[pds] = (1 - self.alpha) * old_v + self.alpha * target
                
                print(f"  [EPISODE {ep} SUMMARY] Total Reward: {episode_reward:.1f} | Final Cov: {kpis['coverage_pct']:.1f}%")

                self.episode_log.append({
                    'episode': ep,
                    'total_reward': episode_reward,
                    'avg_coverage': np.mean(ep_coverage),
                    'min_coverage': np.min(ep_coverage),
                    'avg_pdop': np.mean(ep_pdops),
                    'avg_cn0': np.mean(ep_cn0s),
                    'survival_steps': step,
                    'final_spares': next_state.spares_available})

        print(f"\n  [LOG] CSV -> {csv_path}")

        print("\n" + "="*70)
        print("  TOP 10 LEARNED POST-DECISION STATE VALUES")
        print("="*70)
        top = sorted(self.post_decision_values.items(),
                     key=lambda x: x[1], reverse=True)[:10]
        print(f"  {'PDS State':<65} {'Value':>10}")
        print(f"  {'-'*65} {'-'*10}")
        for pds, val in top:
            print(f"  {str(pds):<65} {val:>10.2f}")

        self._generate_plots(output_dir)
        return csv_path

    def _generate_plots(self, output_dir):
        log = self.episode_log
        if not log:
            print("  [PLOT] No episode data to plot.")
            return

        eps = [r['episode'] for r in log]
        rewards = [r['total_reward'] for r in log]
        avg_cov = [r['avg_coverage'] for r in log]
        min_cov = [r['min_coverage'] for r in log]
        avg_pdop = [r['avg_pdop'] for r in log]
        survival = [r['survival_steps'] for r in log]
        avg_cn0 = [r['avg_cn0'] for r in log]

        fig, axes = plt.subplots(5, 1, figsize=(16, 28), sharex=True)
        fig.suptitle('ADP Learning Progress Over 500 Episodes', 
                     fontsize=15, fontweight='bold', y=0.995)

        # Helper to setup grid
        def setup_ax(ax):
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1, max(eps)])

        # ---- Plot 1: Total Reward (The Learning Curve) ----
        ax = axes[0]; setup_ax(ax)
        roll_r = pd.Series(rewards).rolling(window=10, min_periods=1).mean() # 10-episode rolling average
        ax.plot(eps, rewards, color='#2ecc71', lw=1.0, alpha=0.6, label='Total Episode Reward')
        ax.plot(eps, roll_r, color='#27ae60', lw=2.5, label='10-Episode Rolling Avg')
        ax.axhline(0, color='black', lw=1.0, ls='--')
        ax.set_ylabel('Total Reward', fontsize=11)
        ax.set_title('Learning Curve: Total Reward per Episode', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')

        # ---- Plot 2: Coverage ----
        ax = axes[1]; setup_ax(ax)
        ax.plot(eps, avg_cov, color='#3498db', lw=2.0, label='Average Episode Coverage')
        ax.plot(eps, min_cov, color='#e74c3c', lw=1.5, ls='--', label='Minimum Coverage (Worst Drop)')
        ax.axhline(90, color='red', ls=':', lw=1.5, label='90% Target')
        ax.set_ylabel('Coverage (%)', fontsize=11)
        ax.set_ylim([40, 105])
        ax.set_title('Constellation Coverage Robustness Over Training', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')

        # ---- Plot 3: Survival & Completion ----
        ax = axes[2]; setup_ax(ax)
        ax.plot(eps, survival, color='#9b59b6', lw=2.0, label='Steps Survived (Max 500)')
        ax.axhline(150, color='purple', ls=':', lw=1.5, label='Full Episode Completion')
        ax.set_ylabel('Steps Survived', fontsize=11)
        ax.set_ylim([0, 160])
        ax.set_title('Episode Completion (Does the agent prevent early failure?)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')

        # ---- Plot 4: PDOP ----
        ax = axes[3]; setup_ax(ax)
        ax.plot(eps, avg_pdop, color='#e67e22', lw=2.0, label='Average Episode PDOP')
        ax.axhline(6.0, color='red', ls='--', lw=1.5, label='PDOP=6 Limit')
        ax.set_ylabel('Average PDOP', fontsize=11)
        ax.set_title('Geometry Quality (PDOP) Over Training', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')

        # ---- Plot 5: C/N0 ----
        ax = axes[4]; setup_ax(ax)
        ax.plot(eps, avg_cn0, color='#1abc9c', lw=2.0, label='Average Episode C/N0')
        ax.set_xlabel('Training Episode', fontsize=12)
        ax.set_ylabel('C/N0 (dB-Hz)', fontsize=11)
        ax.set_title('Signal Quality (C/N0) Under Jamming Across Episodes', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')

        plt.tight_layout(rect=[0,0,1,0.975])
        out = os.path.join(output_dir, "learning_progress_plots.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [PLOT] Saved -> {out}")

if __name__ == "__main__":
    OUTPUT_DIR = "resilience_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    physics = SkyfieldPhysicsEngine(GPS_PARAMS)
    print("Downloading live TLE ephemeris data from CelesTrak...")
    initial_state = physics.download_initial_state()
    print(f"Loaded {len(initial_state.satellites)} GPS satellites.\n")

    adp = SkyfieldADPSolver(physics)
    adp.run_adp_training(initial_state, episodes=500, steps_per_episode=150, output_dir=OUTPUT_DIR)

    print("\n" + "="*70)
    print("  RESILIENCE FRAMEWORK - COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print(f"    * resilience_log.csv   - Per-episode KPI log")
    print(f"    * resilience_plots.png - 5-panel resilience figure")
    print("="*70)   