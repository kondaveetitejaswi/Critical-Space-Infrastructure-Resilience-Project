import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
import ssl, csv, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def get_threat_phase(episode_step: int, max_steps: int = 150) -> str:
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
        sats, sp = state.satellites, state.spares_available
        if any(0.0 < s.health < 0.4 and s.launch_countdown == 0 for s in sats):
            valid.append("RETIRE_SATELLITE")
        if sp > 0 and any(s.health <= 0.0 for s in sats):
            valid.append("ACTIVATE_SPARE")
        if sp < self.params.spare_capacity and len(state.pending_launches) < 5:
            valid.append("LAUNCH_SATELLITE")
        if (sp == 0 and any(s.health <= 0.0 for s in sats)
                and sum(1 for s in sats if s.health > 0.8) > 4):
            valid.append("REBALANCE_ORBITS")
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
                spares -= 1; cost = 20.0
                print(f"  [ACTION] ACTIVATE SPARE -> {s.sat_name} | Pool: {spares}")

        elif action == "LAUNCH_SATELLITE":
            pending.append(self.params.launch_delay_steps)
            cost = 30.0
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
                cost, cov_boost = 50.0, 8.0
                print(f"  [ACTION] REBALANCE -> Donor {s.sat_name} | Boost: {cov_boost}%")

        return sats, spares, cost, cov_boost, tuple(pending)


    def _apply_attacks(self, sats, iteration):
        phase = get_threat_phase(iteration)
        attack_log, jammer_loc, persist = [], None, 0
        active_idx = [i for i,s in enumerate(sats)
                      if s.health > 0.5 and s.launch_countdown == 0]
        roll = np.random.rand() 

        if phase == "LOW":
            if roll < 0.08 and len(active_idx) >= 1:
                n = np.random.randint(1, min(4, len(active_idx)+1))
                targets = np.random.choice(active_idx, n, replace=False)
                for idx in targets:
                    s = sats[idx]
                    sats[idx] = SkyfieldSatelliteState(
                        s.sat_name, s.satellite_obj, 0.30, s.age_days,
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
            if roll < 0.15 and len(active_idx) >= 4:
                hi = min(6, len(active_idx)); lo = min(4, hi)
                n = np.random.randint(lo, hi+1)
                targets = np.random.choice(active_idx, n, replace=False)
                for idx in targets:
                    s = sats[idx]
                    sats[idx] = SkyfieldSatelliteState(
                        s.sat_name, s.satellite_obj, 0.25, s.age_days,
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
                            s.sat_name, s.satellite_obj, 0.10, s.age_days,
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
                        nh = max(s.health - 0.25, 0.10)
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
                    new_health = min(new_health + 0.15, 1.0)
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
            pending_launches=tuple(new_pending))

        kpis = self.compute_kpis(new_state)
        final_cov = min(kpis['coverage_pct'] + cov_boost, 100.0)
        kpis['coverage_pct'] = final_cov


        reward = 0.0
        
        # 1. STATE-BASED REWARDS & PENALTIES (The Baseline)
        if final_cov >= 90.0:
            reward += (final_cov - 90.0) * 5.0  
        else:
            reward -= (90.0 - final_cov) * 10.0 

        if kpis['avg_pdop'] <= 4.0:
            reward += 20.0    # Great geometry
        elif kpis['avg_pdop'] > 6.0:
            reward -= 50.0   # Severe penalty for unusable geometry (PDOP > 6)

        if kpis['avg_cn0_dbhz'] >= 30.0:
            reward += 10.0  
        elif kpis['avg_cn0_dbhz'] < 25.0:
            reward -= 20.0

        if spares == 0 and len(state.pending_launches) == 0:
            reward -= 50.0
        elif state.spares_available > 0:
            reward += 5.0

        # 2. CONTEXTUAL ACTION EVALUATION (Did the agent make a smart choice?)
        if action == "NO_OP":
            if final_cov < 90.0 or kpis['operational_sats'] < 24:
                # PUNISHMENT: Doing nothing while the constellation degrades
                reward -= 100.0 
            else:
                # REWARD: Conserving resources when the system is perfectly healthy
                reward += 15.0  

        elif action == "ACTIVATE_SPARE":
            if kpis['operational_sats'] < 24:
                # REWARD: Smart deployment to fix a gap
                reward += 80.0
            else:
                # PUNISHMENT: Wasting a spare when the constellation is already full
                reward -= 50.0  

        elif action == "LAUNCH_SATELLITE":
            if spares < self.params.spare_capacity:
                reward += 150.0
            else:
                reward -= 40.0 

        elif action == "REBALANCE_ORBITS":
            if cov_boost > 0:
                # REWARD: Rebalancing actually improved coverage
                reward += 50.0
            else:
                # PUNISHMENT: Shuffling satellites around for no tangible benefit
                reward -= 30.0


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

    def get_macro_state(self, state, kpis) -> tuple:
        oc   = state.get_operational_count()
        sp   = min(state.spares_available, self.physics.params.spare_capacity)
        h    = round(state.get_avg_health() * 4)          # 0-4
        cov  = (0 if kpis['coverage_pct'] < 80
                else 1 if kpis['coverage_pct'] < 90 else 2)
        jam  = 1 if state.active_jammer_location else 0
        pdop = (0 if kpis['avg_pdop'] < 3
                else 1 if kpis['avg_pdop'] < 6 else 2)
        ph   = {"LOW":0,"MEDIUM":1,"HIGH":2}.get(state.threat_phase, 0)
        pend = min(len(state.pending_launches), 2)
        return (oc, sp, h, cov, jam, pdop, ph, pend)

    def get_post_decision_state(self, macro, action) -> tuple:
        oc,sp,h,cov,jam,pdop,ph,pend = macro
        if   action == "ACTIVATE_SPARE"   and sp > 0:
            return (oc+1, sp-1, h, cov, jam, pdop, ph, pend, "ACT")
        elif action == "LAUNCH_SATELLITE":
            return (oc,   sp,   h, cov, jam, pdop, ph, min(pend+1,2), "LCH")
        elif action == "RETIRE_SATELLITE":
            return (max(oc-1,0), sp, h, cov, jam, pdop, ph, pend, "RET")
        elif action == "REBALANCE_ORBITS":
            return (oc, sp, h, min(cov+1,2), jam, pdop, ph, pend, "RBL")
        return (oc, sp, h, cov, jam, pdop, ph, pend, "NOP")

    def greedy_action(self, macro, state) -> str:
        valid = self.physics.get_valid_actions(state)
        best_act, best_val = "NO_OP", -float('inf')
        for act in valid:
            v = self.post_decision_values[self.get_post_decision_state(macro, act)]
            if v > best_val:
                best_val, best_act = v, act
        return best_act

    def run_adp_training(self, initial_state, episodes=200, steps_per_episode=150, output_dir="."):
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
                
                eps = max(0.05, 0.40 - (ep / episodes) * 0.35)

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
        log      = self.training_log
        iters    = [r['iteration'] for r in log]
        coverage = [r['coverage']  for r in log]
        pdop_raw = [r['pdop']      for r in log]
        cn0      = [r['cn0']       for r in log]
        rewards  = [r['reward']    for r in log]
        op_sats  = [r['op_sats']   for r in log]
        spares   = [r['spares']    for r in log]
        pending  = [r['pending']   for r in log]
        atk_it   = [r['iteration'] for r in log if r['under_attack']]

        pdop_c = [min(p, 30) for p in pdop_raw]   # clip for readability

        fig, axes = plt.subplots(5, 1, figsize=(16, 28), sharex=True)
        fig.suptitle(
            'ADP-Based GPS Constellation Resilience Framework\n'
            'Escalating Threat Scenarios - All 4 Fixes Applied',
            fontsize=15, fontweight='bold', y=0.995)

        def shade(ax):
            ax.axvspan(1,   150, alpha=0.07, color='green',  zorder=0)
            ax.axvspan(151, 350, alpha=0.07, color='orange', zorder=0)
            ax.axvspan(351, 500, alpha=0.07, color='red',    zorder=0)
            for ai in atk_it:
                ax.axvline(x=ai, color='darkred', alpha=0.04, lw=0.3)
            ax.set_xlim([1, max(iters)])
            ax.grid(True, alpha=0.25)

        # ---- Plot 1: Coverage ----
        ax = axes[0]; shade(ax)
        roll_cov = np.convolve(coverage, np.ones(15)/15, mode='same')
        ax.plot(iters, coverage,  color='#5dade2', lw=0.9, alpha=0.6, label='Coverage %')
        ax.plot(iters, roll_cov,  color='#1a5276', lw=2.0, label='15-iter Rolling Avg')
        ax.axhline(90, color='red', ls='--', lw=1.5, label='90% Threshold')
        ax.fill_between(iters, coverage, 90,
                        where=[c < 90 for c in coverage],
                        color='red', alpha=0.30, label='Coverage Deficit')
        ax.set_ylabel('Coverage (%)', fontsize=11)
        ax.set_ylim([40, 105])
        ax.set_title('Global Coverage - Attack / Drop / ADP Recovery', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)

        # ---- Plot 2: PDOP ----
        ax = axes[1]; shade(ax)
        ax.plot(iters, pdop_c,   color='#a569bd', lw=1.0, alpha=0.7, label='Avg PDOP (clip @30)')
        ax.axhline(6.0, color='red', ls='--', lw=1.5, label='PDOP=6 Threshold')
        ax.fill_between(iters, pdop_c, 6.0,
                        where=[p > 6 for p in pdop_c],
                        color='red', alpha=0.25, label='PDOP Degradation')
        ax.set_ylabel('Average PDOP', fontsize=11)
        ax.set_ylim([0, 33])
        ax.set_title('Position Dilution of Precision - Geometry Degradation & Recovery', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

        # ---- Plot 3: Reward ----
        ax = axes[2]; shade(ax)
        roll_r = np.convolve(rewards, np.ones(20)/20, mode='same')
        ax.plot(iters, rewards, color='#58d68d', lw=0.8, alpha=0.6, label='Trajectory Reward')
        ax.plot(iters, roll_r,  color='#1e8449', lw=2.2, label='20-iter Rolling Avg')
        ax.fill_between(iters, rewards, 0,
                        where=[r < 0 for r in rewards],
                        color='red', alpha=0.35, label='Failure Region (Reward < 0)')
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_ylabel('Trajectory Reward', fontsize=11)
        ax.set_title('ADP Reward Signal - Learning, Degradation & Recovery', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)

        # ---- Plot 4: Fleet Health ----
        ax = axes[3]; shade(ax)
        ax.plot(iters, op_sats, color='#e67e22', lw=1.5, label='Operational Sats')
        ax.plot(iters, spares,  color='#2980b9', lw=1.2, ls='--', label='Spare Pool')
        ax.plot(iters, pending, color='#8e44ad', lw=1.0, ls=':',  label='In-Transit (FIX1)')
        ax.axhline(24, color='orange', ls=':', lw=1.5, label='Min 24-sat baseline')
        ax.set_ylabel('Count', fontsize=11)
        ax.set_ylim([0, 38])
        ax.set_title('Fleet Health - Operational Sats, Spare Pool & Launch Pipeline (FIX 1)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)

        # ---- Plot 5: C/N0 ----
        ax = axes[4]; shade(ax)
        ax.plot(iters, cn0, color='#1abc9c', lw=1.0, alpha=0.8, label='Avg C/N0 (dB-Hz)')
        ax.axhline(35, color='orange', ls='--', lw=1.2, label='35 dB-Hz min threshold')
        ax.fill_between(iters, cn0, 35,
                        where=[c < 35 for c in cn0],
                        color='orange', alpha=0.30, label='Signal Degradation')
        ax.set_ylabel('C/N0 (dB-Hz)', fontsize=11)
        ax.set_xlabel('Training Iteration', fontsize=11)
        ax.set_title('Signal Quality - C/N0 Under Jamming (FIX 2: Single Attack/Step)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)

        # Phase legend
        patches = [
            mpatches.Patch(color='green',  alpha=0.4, label='Phase 1: LOW (1-150)'),
            mpatches.Patch(color='orange', alpha=0.4, label='Phase 2: MEDIUM (151-350)'),
            mpatches.Patch(color='red',    alpha=0.4, label='Phase 3: HIGH (351-500)'),
        ]
        fig.legend(handles=patches, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.99, 0.98),
                   title='Threat Phases', title_fontsize=9)

        plt.tight_layout(rect=[0,0,1,0.975])
        out = os.path.join(output_dir, "resilience_plots.png")
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
    adp.run_adp_training(initial_state, episodes=150, output_dir=OUTPUT_DIR)

    print("\n" + "="*70)
    print("  RESILIENCE FRAMEWORK - COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print(f"    * resilience_log.csv   - Per-episode KPI log")
    print(f"    * resilience_plots.png - 5-panel resilience figure")
    print("="*70)   