import numpy as np
from pettingzoo import AECEnv
from gymnasium.spaces import Box, Dict
from pettingzoo.utils.agent_selector import agent_selector

class SatelliteDefenseEnv(AECEnv):
    metadata = {"render_modes": ['human'], "name": "satellite_defense_v0", "is_multiagent": True}

    
    # CORE PETTINGZOO ENVIRONMENT CLASS FOR SATELLITE DEFENSE SIMULATION
    def __init__(self):
        super().__init__()

        self.agents = ["defender_0", "defender_1", "defender_2", "defender_3", "defender_4", 
                       "attacker_0", "attacker_1", "attacker_2", "attacker_3", "attacker_4"]
        self.possible_agents = self.agents[:]

        # continuous action space for each agent i.e., attacker and defender
        self.action_spaces = {
            "defender_0": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "defender_1": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "defender_2": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "defender_3": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "defender_4": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "attacker_0": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "attacker_1": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "attacker_2": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "attacker_3": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "attacker_4": Box(low=0, high=1, shape=(1,), dtype=np.float32)
        }

        # continuous observations: system health levels and radiation levels
        self.observation_spaces = {
            agent: Dict({
                "power": Box(low=0.0, high=1.0, shape=()), #normalized battery status
                "memory": Box(low=0.0, high=1.0, shape=()), #memory integrity
                "control": Box(low=0.0, high=1.0, shape=()), #control system stability
                "software_health": Box(low=0.0, high=1.0, shape=()), #system integrity post-attacks
                
                "raditaion_level": Box(low=0.0, high=100.0, shape=()), #cosmic radiation exposure
                
                "under_attack": Box(low=0.0, high=1.0, shape=()),    #0: no attack, 1:under attack

                "signal_strength": Box(low = -120.0, high = -50.0, shape=()), #satellite signal reception

                "communication_status": Box(low=0.0, high=1.0, shape=()), # 1: Fully operational, 0: no signal
                "battery_health": Box(low=0.0, high=1.0, shape=()), #long-term batter degradation; not just current charge
                "thermal_status": Box(low =-200.0, high= 200.0, shape =()), #subsystems operating within temperature ranges
                "orbit_deviation": Box(low=0.0, high=1.0, shape=()), #deviation from intended orbit path
                "neighbour_state_trust": Box(low=0.0, high=1.0, shape=()), #confidence/trust score on the shared data from neighbouring satellites
                "data_queue_size": Box(low = 0.0, high=100.0, shape=()), #reflects the varying traffic loads-congestion at 90 MB might force data offloading strategies
                "redundancy_status": Box(low = 0.0, high = 1.0, shape=()) #1: fully redundant, 0: no redundancy


            })
            for agent in self.agents
        }

        self.agent_selector = agent_selector(self.agents) #creates a turn based system for agents, acting will be in sequence
        self.agent_selection = self.agent_selector.reset() #Resets the agent selection to the first agent

        self.state = None
        self.rewards = {agent: 0.0 for agent in self.agents} #reward tracking system for each agent
        self.dones = {agent: False for agent in self.agents} #tracks if the agent finished acting in the episode``
        self.infos = {agent: {} for agent in self.agents} #holds miscellaneous information for each agent
        self.defender_risk = {agent: {} for agent in self.agents}

    def reset(self, seed=None, options= None):
        #restoring the agents
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        #applying the partial deterioration of the satellite system
        self.state = {
            "power": max(0.9, self.state.get("power", 1.0) - np.random.uniform(0.01, 0.03)),
            "memory": max(0.85, self.state.get("memory", 1.0) - np.random.uniform(0.02, 0.04)),
            "control": max(0.85, self.state.get("control", 1.0) - np.random.uniform(0.02, 0.04)),  
            "software_health": max(0.85, self.state.get("software_health", 1.0) - np.random.uniform(0.02, 0.04)),  
        
            "radiation_level": min(100.0, self.state.get("radiation_level", 0.0) + np.random.uniform(0.5, 1.5)),  

            "under_attack": 0.0,

            "signal_strength": max(-120, min(-50, self.state.get("signal_strength", -80) - np.random.uniform(1, 5))),  

            "communication_status": max(0.95, self.state.get("communication_status", 1.0) - np.random.uniform(0.01, 0.02)),  
            "battery_health": max(0.9, self.state.get("battery_health", 1.0) - np.random.uniform(0.01, 0.03)),  

            "thermal_status": max(-200, min(200, self.state.get("thermal_status", 0.0) + np.random.uniform(-5, 5))),  
            "orbit_deviation": min(5.0, self.state.get("orbit_deviation", 0.0) + np.random.uniform(0.02, 0.08)),  
            "neighbor_state_trust": max(0.5, self.state.get("neighbor_state_trust", 1.0) - np.random.uniform(0.01, 0.05)),  
            "data_queue_size": max(0, min(100, self.state.get("data_queue_size", 10.0) + np.random.uniform(-5, 5))),  
            "redundancy_status": max(0.6, self.state.get("redundancy_status", 1.0) - np.random.uniform(0.01, 0.03)),  
        
        }
        
        self._apply_environment_decay()

        self.rewards = {agent:0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.defender_risk = {agent: {} for agent in self.agents}


    def observe(self, agent): #fully observable environment
            if agent not in self.agents:
                return None
            
            obs = {}

            #Attackers focus on system weaknesses
            if "attacker" in agent:
                obs = {
                    "power": self.state["power"],
                    "memory": self.state["memory"],
                    "control": self.state["control"],
                    "software_health": self.state["software_health"],
                    "radiation_level": self.state["radiation_level"],
                    "under_attack": self.state["under_attack"],
                    "signal_strength": self.state["signal_strength"],
                    "communication_status": self.state["communication_status"],
                    "battery_health": self.state["battery_health"],
                    "thermal_status": self.state["thermal_status"],
                    "orbit_deviation": self.state["orbit_deviation"], # follows state spoofing policy where the attacker can manipulate the orbit deviaiton reading
                    "neighbor_state_trust": self.state["neighbor_state_trust"],
                    "data_queue_size": self.state["data_queue_size"], # reports a wrong data measurement for the available data queue size, hence tamper with the data queue size
                    "redundancy_status": self.state["redundancy_status"]   # FDI (False Data Injection) attack on the redundancy status;In satellites, if the telemetry 
                                                                        # says "Backup computer OK" when it’s actually failed, the on-board supervisor may switch to a dead unit when the primary fails → total system loss.
                }

            elif "defender" in agent:
                obs = {
                    "power": self.state["power"],
                    "memory": self.state["memory"],
                    "control": self.state["control"],
                    "software_health": self.state["software_health"],
                    "radiation_level": self.state["radiation_level"],
                    "under_attack": self.state["under_attack"],
                    "signal_strength": self.state["signal_strength"],
                    "communication_status": self.state["communication_status"],
                    "battery_health": self.state["battery_health"],
                    "thermal_status": self.state["thermal_status"],
                    "orbit_deviation": self.state["orbit_deviation"],
                    "neighbor_state_trust": self.state["neighbor_state_trust"],
                    "data_queue_size": self.state["data_queue_size"],
                    "redundancy_status": self.state["redundancy_status"]
                }

            return obs
        
    def step(self, action):
        agent = self.agent_selection

        if agent not in self.agents:
            return
        
        #attack or defense logic
        self._apply_action(agent, action)

        #Atttackers collaborate and share vulnerability insights
        if "attacker" in agent:
            self._update_attacker_strategy()

        #defenders collborate to enhance system resilience
        elif "defender" in agent:
            self._share_risk_alerts()
            self._adjust_defense_priorities()

        self.rewards[agent] = self._compute_reward(agent)

        self._apply_radiation_decay()

        self._update_done_status()

        self.agent_selection = self.agent_selector.next()
    

    # MAIN ACTION HANDLING METHODS 
    def _apply_action(self, agent, action):
        if "attacker" in agent:
            self._attacker_action(agent, action)

        elif "defender" in agent:
            self._defender_action(agent, action)

    def _attacker_action(self, agent, action):
        # handle attacker actions in a cont value between 0 and 1
        action_value = action[0] # extract the single cont value

        obs = self.observe(agent)

        vulnerabilities = {
            "power_attack": 1 - obs["power"],
            "memory_attack": 1 - obs["memory"],
            "control_attack": 1 - obs["control"],
            "software_attack": 1 - obs["software_health"],
            "radiation_attack": 1 - obs["radiation_level"],
            "attack_status": obs["under_attack"],
            "signal_attack": 1 - obs["signal_strength"],
            "communication_attack": 1 - obs["communication_status"],
            "battery_attack": 1 - obs["battery_health"],
            "thermal_attack": 1 - obs["thermal_status"],
            "orbit_attack": 1 - obs["orbit_deviation"],
            "neighbor_trust_attack": 1 - obs["neighbor_state_trust"],
            "data_measurement_attack": 1 - obs["data_queue_size"],
            "redundancy_attack": 1 - obs["redundancy_status"] 
        }

        # Define subsystem groups
        roots = ["power_attack", "battery_attack", "radiation_attack"]
        thermal_related = ["thermal_attack"]
        software_related = ["software_attack", "memory_attack"]
        control_related = ["control_attack"]
        communication_related = ["communication_attack", "signal_attack"]
        redundancy_related = ["redundancy_attack"]
        data_related = ["data_measurement_attack"]
        orbit_related = ["orbit_attack"]
        trust_related = ["neighbor_trust_attack"]

        #defining the realistic dependency order
        dependency_chain = [
            roots,
            thermal_related,
            software_related,
            control_related,
            communication_related,
            redundancy_related,
            data_related,
            orbit_related,
            trust_related
        ]

        attack_type = None
        threshold = 0.6

        for group in dependency_chain:
            weak_points = [(k, vulnerabilities[k]) for k in group if vulnerabilities[k] > threshold]
            if weak_points:
                weakest = max(weak_points, key =lambda x: x[1])
                attack_type = weakest[0]
                break
        if attack_type is None:
            attack_type = "control_attack"

        if obs["data_queue_size"] > 80:
            attack_type = "data_measurement_attack"
        elif obs["under_attack"]:
            attack_type = "software_attack"

        self.state["under_attack"] = 1.0  # Mark that an attack is in progress


    def _defender_action(self, agent, action):
        defense_strength = action[0] 

        obs = self.observe(agent)

        roots = ["power", "battery_health", "radiation_level"]
        thermal_related = ["thermal_status"]
        software_related = ["software_health", "memory"]
        control_related = ["control"]
        communication_related = ["communication_status", "signal_strength"]
        redundancy_related = ["redundancy_status"]
        data_related = ["data_queue_size"]
        orbit_related = ["orbit_deviation"]
        trust_related = ["neighbor_state_trust"]

        dependency_chain = [
            roots,
            thermal_related,
            software_related,
            control_related,
            communication_related,
            redundancy_related,
            data_related,
            orbit_related,
            trust_related
        ]

        weak_threshold = 0.75
        high_threshold_queue = 70
        high_threshold_orbit = 5.0

        defense_type = None

        for group in dependency_chain:
            weak_points = []
            for var in group:
                val = obs[var]
                if var == "data_queue_size" and val > high_threshold_queue:
                    weak_points.append((var, val))
                elif var == "orbit_deviation" and val > high_threshold_orbit:
                    weak_points.append((var, val))
                elif var == "radiation_level" and val > 50:
                    weak_points.append((var, val))
                else:
                    if val < weak_threshold:
                        weak_points.append((var, val))
            if weak_points:
                if group in [data_related, orbit_related, ["radiation_level"]]:
                    defense_target = max(weak_points, key=lambda x: x[1])[0]
                else:
                    defense_target = min(weak_points, key = lambda x: x[1])[0]
                break
        
        if defense_target is None:
            defense_target = "control"

        #apply defense: increase or decrease depending on the variable
        if defense_target == "radiation_level":
            self.state[defense_target] = max(0.0, self.state[defense_target] - 5 * defense_strength)
        elif defense_target == "data_queue_size":
            self.state[defense_target] = max(0.0, self.state[defense_target] - 10 * defense_strength)
        elif defense_target == "orbit_deviation":
            self.state[defense_target] = max(0.0, self.state[defense_target] - 0.5 * defense_strength)
        else:
            # Normal system health vars: boost them
            self.state[defense_target] = min(1.0, self.state[defense_target] + 0.1 * defense_strength)

        # Mark that an active defense was done
        self.state["under_attack"] = 0.0  #  reset if defender countered an attack


    def _apply_attack_decay(self, agent, action):
        attack_success = False

        base_decay = 0.2
        threshold = 0.6

        attack_params = {
            "power_attack": {
                "target": "power",
                "success_prob": 0.7 if self.state["power"] < threshold else 0.4,
                "decay": base_decay * 1.5,
                "reward": 1.5
            },
            "battery_attack": {
                "target": "battery_health",
                "success_prob": 0.7 if self.state["battery_health"] < threshold else 0.4,
                "decay": base_decay * 1.5,
                "reward": 1.5
            },
            "radiation_attack": {
                "target": "radiation_level",
                "success_prob": 0.5,
                "decay": base_decay * 2.0,
                "reward": 1.2
            },

            "thermal_attack": {
                "target": "thermal_status",
                "success_prob": 0.6 if abs(self.state["thermal_status"]) > 100 else 0.3,
                "decay": 10,
                "reward": 1.0
            },

            "software_attack": {
                "target": "software_health",
                "success_prob": 0.6 if self.state["under_attack"] else 0.2,
                "decay": base_decay,
                "reward": 1.0
            },
            "memory_attack": {
                "target": "memory",
                "success_prob": 0.6 if self.state["memory"] < threshold else 0.3,
                "decay": base_decay,
                "reward": 1.0
            },

            "control_attack": {
                "target": "control",
                "success_prob": 0.5 if self.state["control"] < threshold else 0.2,
                "decay": base_decay,
                "reward": 1.0
            },
            "communication_attack": {
                "target": "communication_status",
                "success_prob": 0.5 if self.state["communication_status"] < threshold else 0.2,
                "decay": base_decay,
                "reward": 1.0
            },
            "signal_attack": {
                "target": "signal_strength",
                "success_prob": 0.5,
                "decay": 10.0,
                "reward": 0.8
            },

            "redundancy_attack": {
                "target": "redundancy_status",
                "success_prob": 0.5 if self.state["redundancy_status"] < threshold else 0.2,
                "decay": base_decay,
                "reward": 0.8
            },
            "data_measurement_attack": {
                "target": "data_queue_size",
                "success_prob": 0.8 if self.state["data_queue_size"] > 80 else 0.4,
                "decay": 15.0,
                "reward": 1.2
            },
            "orbit_decay": {
                "target": "orbit_deviation",
                "success_prob": 0.4,
                "decay": base_decay,
                "reward": 1.0
            },
            "neighbor_trust_attack": {
                "target": "neighbor_state_trust",
                "success_prob": 0.5 if self.state["neighbor_state_trust"] < threshold else 0.2,
                "decay": base_decay,
                "reward": 0.8
            }
        }


        if action in attack_params:
            params = attack_params[action]
            attack_success = np.random_rand() < params["success_prob"]

            if attack_success:
                target = params["target"]

                if target in ["radiation_level", "thermal_status", "data_queue_size"]:
                    self.state[target] = min(self.state[target] + params["decay"], self.observation_spaces[agent][target].high)
                elif target == "signal_strength":
                    self.state[target] = max(self.state[target] - params["decay"], self.observation_spaces[agent][target].low)
                else:
                    self.state[target] = max(self.state[target] - params["decay"], 0.0)

                self.rewards[agent] += params["reward"]

                self.apply_dependency_cascade(action)


        return attack_success

    def apply_dependency_cascade(self, attack_type):

        root_cascades = {
            "power_attack": {
                "control": 0.1,
                "communication_status": 0.1
            },
            "battery_attack": {
                "power": 0.1
            },
            "radiation_attack": {
                "memory": 0.1,
                "software_health": 0.1
            }
        }

        system_cascades = {
            "thermal_attack": {
                "software_health": 0.1,
                "memory": 0.1
            },
            "control_attack": {
                "orbit_deviation": 0.1,
                "redundancy_status": 0.1
            }
        }

        cascades = root_cascades if attack_type in root_cascades else system_cascades

        if attack_type in cascades:
            for target, decay in cascades[attack_type].items():
                self.state[target] = max(self.state[target] - decay, 0.0)


    def _apply_defense_effects(self, agent, action):
        defense_params = {
            "power_defense": {
                "target": "power",
                "improvement": 0.2,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "battery_protection": {
                "target": "battery_health",
                "improvement": 0.2,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "radiation_shield": {
                "target": "radiation_level",
                "improvement": -2.0,
                "cost": 0.15,
                "success_prob": 0.7
            },
            "memory_protection": {
                "target": "memory",
                "improvement": 0.15,
                "cost": 0.1,
                "success_prob": 0.9
            },
            "software_patch": {
                "target": "software_health",
                "improvement": 0.2,
                "cost": 0.1,
                "success_prob": 0.85
            },
            "communication_boost": {
                "target": "communication_status",
                "improvement": 0.15,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "signal_boost": {
                "target": "signal_strength",
                "improvement": 5.0,
                "cost": 0.1,
                "success_prob": 0.75
            },
            "thermal_regulation": {
                "target": "thermal_status",
                "improvement": -10.0,
                "cost": 0.1,
                "success_prob": 0.9
            },
            "control_system_protection":{
                "target": "control",
                "improvement": 0.2,
                "cost": 0.15,
                "success_prob": 0.8
            },
            "orbit_correction": {
                "target": "orbit_deviation",
                "improvement": -0.1,
                "cost": 0.2,
                "success_prob": 0.7
            },

            "trust_verification": {
                "target": "neighbor_state_trust",
                "improvement": 0.15,
                "cost": 0.05,
                "success_prob": 0.9
            },
            "queue_management": {
                "target": "data_queue_size",
                "improvement": -10.0,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "redundancy_boost": {
                "target": "redundancy_status",
                "improvement": 0.2,
                "cost": 0.15,
                "success_prob": 0.85
            },

            "emergency_reboot": {
                "target": "under_attack",
                "improvement": -1.0,
                "cost": 0.3,
                "success_prob": 0.6
            }
            
        }

        if action in defense_params:
            params = defense_params[action]
            success = np.random.rand() < params["success_prob"]

            if success:
                target = params["target"]
                improvement = params["improvement"]

                if target in ["radiation_level", "thermal_status", "data_queue_size"]:
                    self.state[target] = max(
                        self.observation_spaces[agent][target].low,
                        min(self.state[target] + improvement, self.observation_spaces[agent][target].high)
                    )

                else:
                    self.state[target] = max(0.0, min(1.0, self.state[target] + improvement))

                self.rewards[agent] += 1.0
                self.state["under_attack"] = max(0.0, self.state["under_attack"] - 0.2)


                self.apply_defense_reinforcement(action)
            return success
        return False
        
    def _share_risk_alerts(self):
        """Defenders exchange vulnerability information to improve threat anticipation."""
        risk_metrics = {
            "under_attack": self.state["under_attack"],
            "radiation_risk": self.state["radiation_level"] / 100.0,
            "power_risk": 1 - self.state["power"],
            "memory_risk": 1 - self.state["memory"],
            "control_risk": 1 - self.state["control"],

            "software_risk": 1 - self.state["software_health"],
            "signal_risk": 1 - self.state["signal_strength"],

            "communication_risk": 1 - self.state["communication_status"],

            "battery_risk": 1 - self.state["battery_health"],

            "thermal_risk": abs(self.state["thermal_status"]) / 200.0,
            "orbit_risk": self.state["orbit_deviation"],
            "trust_risk": 1 - self.state["neighbor_state_trust"],
            "queue_risk": self.state["data_queue_size"] / 100.0,
            "redundancy_risk": 1 - self.state["redundancy_status"]

        }

        for defender in self.agents:
            if "defender" in defender:
                weighted_risks = {
                    "critical": max(
                        risk_metrics["power_risk"],
                        risk_metrics["software_risk"],
                        risk_metrics["under_attack"]
                    ),
                    "high": max(
                        risk_metrics["memory_risk"],
                        risk_metrics["communication_risk"],
                        risk_metrics["control_risk"]
                    ),
                    "medium": max(
                        risk_metrics["radiation_risk"],
                        risk_metrics["thermal_risk"],
                        risk_metrics["signal_risk"],
                        risk_metrics["battery_risk"]
                    ),
                    "low": max(
                        risk_metrics["orbit_risk"],
                        risk_metrics["trust_risk"],
                        risk_metrics["queue_risk"],
                        risk_metrics["redundancy_risk"]
                    )
                }

                self.defender_risk[defender] = weighted_risks


    def _adjust_defense_priorities(self):
        """Defenders allocate defensive measures dynamically based on attack patterns."""
        threat_levels = {
            # critical priority systems
            "power": {"value": self.state["power"], "critical": 0.4, "high":0.6},
            "software_health": {"value": self.state["software_health"], "critical": 0.4, "high": 0.7},
            "under_attack": {"value": self.state["under_attack"], "critical": 0.7, "high": 0.5},

            # High priority systems
            "memory": {"value": self.state["memory"], "critical": 0.3, "high": 0.6},
            "communication_status": {"value": self.state["communication_status"], "critical":0.4, "high": 0.7},
            "control": {"value": self.state["control"], "critical": 0.4, "high":0.6},

            # Medium priority systems
            "radiation_level": {"value": self.state["radiation_level"], "critical": 80, "high": 60},
            "thermal_status": {"value": abs(self.state["thermal_status"]), "critical": 150, "high":100},
            "signal_strength": {"value": abs(self.state["signal_strength"]), "critical": -100, "high": -80},
            "battery_health": {"value": self.state["battery_health"], "critical": 0.3, "high": 0.6},

            # Low priority systems
            "orbit_deviation": {"value": self.state["orbit_deviation"], "critical": 0.8, "high": 0.6},
            "neighbor_state_trust": {"value": self.state["neighbor_state_trust"], "critical": 0.3, "high": 0.6},
            "data_queue_size": {"value": self.state["data_queue_size"], "critical": 90, "high": 70},
            "redundancy_status": {"value": self.state["redundancy_status"], "critical": 0.3, "high": 0.6}
        }

        priority_groups = {
            "critical": ["power", "software_health", "under_attack"],
            "high": ["memory", "communication_status", "control"],
            "medium": ["radiation_level", "thermal_status", "signal_strength", "battery_health"],
            "low": ["orbit_deviation", "neighbor_state_trust", "data_queue_size", "redundancy_status"]
        }

        threats = {priority: [] for priority in priority_groups.key()}

        for priority, systems in priority_groups.items():
            for system in systems:
                thresh = threat_levels[system]
                if system in ["radiation_level", "thermal_status", "data_queue_size"]:
                    if thresh["value"] >= thresh["critical"]:
                        threats["critical"].append(system)
                    elif thresh["value"] >= thresh["high"]:
                        threats["high"].append(system)
                elif system == "signal_strength":
                    if thresh["value"] <= thresh["critical"]:
                        threats["critical"].append(system)
                    elif thresh["value"] <= thresh["high"]:
                        threats["high"].append(system)

                else:
                    if thresh["value"] <= thresh["critical"]:
                        threats["critical"].append(system)
                    elif thresh["value"] <= thresh["high"]:
                        threats["high"].append(system)

        if threats["critical"]:
            self.defense_priority = threats["critical"][0]
            self.defense_level = "critical"
            self.defense_targets = threats["critical"]
        elif threats["high"]:
            self.defense_priority = threats["high"][0]
            self.defense_level = "high"
            self.defense_targets = threats["high"]
        elif threats["medium"]:
            self.defense_priority = threats["medium"][0]
            self.defense_level = "medium"
            self.defense_targets = threats["medium"]
        else:
            self.defense_priority = threats["low"][0] if threats["low"] else "power"
            self.defense_level = "low"
            self.defense_targets = threats["low"]

        self.defense_info = {
            "primary_target": self.defense_priority,
            "threat_level": self.defense_level,
            "all_threats": {level: systems for level, systems in threats.items() if systems},
            "defense_targets": self.defense_targets
        }


    def apply_defense_reinforcement(self, defense_action):
        reinforcement_effects = {
            "power_defense": {
                "software_health": 0.05,
                "control": 0.05,
                "battery_health": 0.05
            },
            "battery_boost": {
                "power": 0.05,
                "thermal_status": -5.0
            },

            # Core Systems
            "radiation_shield": {
                "memory": 0.05,
                "software_health": 0.05,
                "thermal_status": -5.0
            },
            "memory_protection": {
                "software_health": 0.05,
                "redundancy_status": 0.05,
                "control": 0.03
            },
            "software_patch": {
                "memory": 0.05,
                "control": 0.05,
                "data_queue_size": -5.0
            },

            # Communication Systems
            "communication_boost": {
                "neighbor_state_trust": 0.05,
                "control": 0.05,
                "signal_strength": 2.0
            },
            "signal_boost": {
                "communication_status": 0.05,
                "neighbor_state_trust": 0.03
            },

            #Environmental Systems
            "thermal_regulation": {
                "software_health": 0.05,
                "memory": 0.05,
                "battery_health": 0.03
            },
            "orbit_correction": {
                "control": 0.05,
                "communication_status": 0.05,
                "signal_strength": 1.0
            },

            # Data and Trust Systems
            "trust_verification": {
                "communication_status": 0.05,
                "redundancy_status": 0.05,
                "data_queue_size": -3.0
            },
            "queue_management": {
                "software_health": 0.05,
                "memory": 0.05,
                "control": 0.03
            },

            # Redundancy Systems
            "redundancy_boost": {
                "control": 0.05,
                "memory": 0.05,
                "software_health": 0.03
            }
        }

        if defense_action in reinforcement_effects:
            for target, improvement in reinforcement_effects[defense_action].items():
                if target in ["thermal_status", "data_queue_size"]:
                    self.state[target] = max(
                        self.observation_spaces[self.agents[0]][target].low,
                        self.state[target] + improvement
                    )
                elif target == "signal_strength":
                    self.state[target] = min(
                        -50.0, 
                        max(-120.0, self.state[target] + improvement)
                    )
                else:
                    self.state[target] = min(1.0, self.state[target] + improvement)


    def compute_attacker_reward(self, agent):
        system_weights = {
            # critical systems
            "power": 2.0,
            "software_health": 2.0,
            "under_attack": 1.5,

            # High priority 
            "memory": 1.5,
            "communication_status": 1.5,
            "control": 1.5,

            # Medium Priority
            "radiation_level": 1.0,
            "thermal_status": 1.0,
            "signal_strength": 1.0,
            "battery_health": 1.0,

            # Low Priority
            "orbit_deviation": 0.5,
            "neighbor_state_trust": 0.5,
            "data_queue_size": 0.5,
            "redundancy_status": 0.5
        }

        attack_success = 0.0
        total_weight = 0.0

        for system, weight, in system_weights.items():
            if system in ["radiation_level", "thermal_status", "data_queue_size"]:
                impact = self.state[system] / self.observation_spaces[agent][system].high
            elif system == "signal_strength":
                impact = (self.observation_spaces[agent][system].high - self.state[system]) / \
                        (self.observation_spaces[agent][system].high - self.observation_spaces[agent][system].low)
            else:
                impact = 1.0 - self.state[system]

            attack_success += impact * weight
            total_weight += weight

        attack_success /= total_weight

        reward = attack_success * 2.0

        if self.state["under_attack"] > 0.5:
            reward += 0.5
        if self.state["power"] < 0.4 or self.state["software_health"] < 0.4:
            reward += 0.5

        if self.state["redundancy_status"] > 0.7:
            reward *= 0.8

        return np.clip(reward, -2.0, 2.0)
    
    def compute_defender_reward(self, agent):
        system_weights = {
            # Critical systems
            "power": 2.0,
            "software_health": 2.0,
            "under_attack" : 1.5,

            # High priority systems
            "memory": 1.5,
            "communication_status": 1.5,
            "control": 1.5,

            # Medium priority systems
            "radiation_level": 1.0,
            "thermal_status": 1.0,
            "signal_strength": 1.0,
            "battery_health": 1.0,

            # Low priority systems
            "orbit_deviation": 0.5,
            "neighbor_state_trust": 0.5,
            "data_queue_size": 0.5,
            "redundancy_status": 0.5

        }

        system_health = 0.0
        total_weight = 0.0

        # Calculate system health
        for system, weight in system_weights.items():
            if system in  ["radiation_level", "thermal_status", "data_queue_size"]:
                health = 1.0 - (self.state[system] / self.observation_spaces[agent][system].high)
            elif system == "signal_strength":
                health = (self.state[system] - self.observation_spaces[agent][system].low) / \
                            (self.observation_spaces[agent][system].high - self.observation_spaces[agent][system].low)
            else:
                # higher value indicate better health
                health = self.state[system]

            system_health += health * weight
            total_weight += weight

        system_health /= total_weight

        # Base reward from system health
        reward = system_health * 2.0

        if self.state["under_attack"] < 0.2:
            reward += 0.5

        critical_health = (self.state["power"] + self.state["software_health"] + self.state["control"]) / 3.0
        reward += critical_health * 0.5

        # penalities and additional bonuses
        if system_health < 0.3:
            reward -= 1.0
        if self.state["redundancy_status"] > 0.7:
            reward += 0.2

        return np.clip(reward, -2.0, 2.0)
    
    def compute_reward(self, agent):
        if "attacker" in agent:
            return self.compute_attacker_reward(agent)
        elif "defender" in agent:
            return self.compute_defender_reward
        
        return 0.0
        
        
    def _apply_environment_decay(self):
        """Increase deterioration rate for satellites in extreme environments."""
        radiation_factor = self.state["radiation_level"] / 100.0  # Normalize radiation impact
        
        # Apply accelerated deterioration if radiation is high
        self.state["battery_health"] = max(0.7, self.state["battery_health"] - np.random.uniform(0.01, 0.05) * (1 + radiation_factor))
        self.state["control"] = max(0.8, self.state["control"] - np.random.uniform(0.02, 0.06) * (1 + radiation_factor))
        self.state["neighbor_state_trust"] = max(0.5, self.state["neighbor_state_trust"] - np.random.uniform(0.01, 0.05) * (1 + radiation_factor))


    def _update_done_status(self):
        """Checks if the episode should terminate due to extreme system failure."""
        for agent in self.agents:
            if self.state["software_health"] < 0.2 or self.state["communication_status"] < 0.2:
                self.dones[agent] = True  # System breakdown leads to episode termination

    def render(self, mode="human"):
        """Displays the current environment state for debugging and tracking."""
        if mode != "human":
            return  # Rendering only works in human-readable mode
        
        print("\n--- Environment Status ---")
        print(f"Satellite Power: {self.state['power']:.2f}")
        print(f"Memory Integrity: {self.state['memory']:.2f}")
        print(f"Control Stability: {self.state['control']:.2f}")
        print(f"Software Health: {self.state['software_health']:.2f}")
        print(f"Radiation Exposure: {self.state['radiation_level']:.2f}")
        print(f"Under Attack: {self.state['under_attack']}")
        print(f"Signal Strength: {self.state['signal_strength']:.2f} dBm")
        print(f"Communication Status: {self.state['communication_status']:.2f}")
        print(f"Battery Health: {self.state['battery_health']:.2f}")
        print(f"Thermal Status: {self.state['thermal_status']:.2f}")
        print(f"Orbit Deviation: {self.state['orbit_deviation']:.2f}")
        print(f"Neighbor Trust: {self.state['neighbor_state_trust']:.2f}")
        print(f"Data Queue Size: {self.state['data_queue_size']:.2f} MB")
        print(f"Redundancy Status: {self.state['redundancy_status']:.2f}")
        
        print("\nAgent Status:")
        for agent in self.agents:
            print(f"{agent} -> Reward: {self.rewards[agent]:.2f}, Done: {self.dones[agent]}")

        print("--- End of Render ---\n")

    def close(self):
        """Handles cleanup for environment shutdown."""
        print("Closing Satellite Defense Environment...")
        self.agents = []
        self.state = None
        self.rewards = {}
        self.dones = {}
        self.infos = {}
        print("Environment closed successfully.")