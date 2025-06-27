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
                
                "radiation_level": Box(low=0.0, high=100.0, shape=()), #cosmic radiation exposure
                
                "under_attack": Box(low=0.0, high=1.0, shape=()),    #0: no attack, 1:under attack

                "signal_strength": Box(low = -120.0, high = -50.0, shape=()), #satellite signal reception

                "communication_status": Box(low=0.0, high=1.0, shape=()), # 1: Fully operational, 0: no signal
                "battery_health": Box(low=0.0, high=1.0, shape=()), #long-term batter degradation; not just current charge
                "thermal_status": Box(low =-200.0, high= 200.0, shape =()), #subsystems operating within temperature ranges
                "orbit_deviation": Box(low=0.0, high=1.0, shape=()), #deviation from intended orbit path
                "neighbor_state_trust": Box(low=0.0, high=1.0, shape=()), #confidence/trust score on the shared data from neighbouring satellites
                "data_queue_size": Box(low = 0.0, high=100.0, shape=()), #reflects the varying traffic loads-congestion at 90 MB might force data offloading strategies
                "redundancy_status": Box(low = 0.0, high = 1.0, shape=()) #1: fully redundant, 0: no redundancy


            })
            for agent in self.agents
        }

        self.agent_selector = agent_selector(self.agents) #creates a turn based system for agents, acting will be in sequence
        self.agent_selection = self.agent_selector.reset() #Resets the agent selection to the first agent

        self.state = {
            "power": 1.0,
            "memory": 1.0,
            "control": 1.0,
            "software_health": 1.0,
            "radiation_level": 0.0,
            "under_attack": 0.0,
            "signal_strength": -80.0,
            "communication_status": 1.0,
            "battery_health": 1.0,
            "thermal_status": 0.0,
            "orbit_deviation": 0.0,
            "neighbor_state_trust": 1.0,
            "data_queue_size": 10.0,
            "redundancy_status": 1.0
        }
        self.rewards = {agent: 0.0 for agent in self.agents} #reward tracking system for each agent
        self.dones = {agent: False for agent in self.agents} #tracks if the agent finished acting in the episode``
        self.infos = {agent: {} for agent in self.agents} #holds miscellaneous information for each agent
        self.defender_risk = {agent: {} for agent in self.agents}
        self.num_steps = 0
        self.action_history = []
        self.episode_rewards = {agent: [] for agent in self.agents}

        # Add new tracking variables
        self.attack_history = {
            system: {
                "last_attack_step": None,
                "attack_frequency": 0,
                "successful_attacks": 0,
                "alert_level": 0.0  # 0-1 scale for system alertness
            }
            for system in ["power", "memory", "control", "software_health", 
                          "communication_status", "battery_health", "signal_strength",
                          "radiation_level", "thermal_status", "orbit_deviation",
                          "neighbor_state_trust", "data_queue_size", "redundancy_status"]
        }
        
        self.defender_memory = {
            agent: {
                "successful_defenses": [],
                "failed_defenses": [],
                "alert_systems": set()
            }
            for agent in self.agents if "defender" in agent
        }
        
        self.attacker_memory = {
            agent: {
                "successful_attacks": [],
                "failed_attacks": [],
                "identified_vulnerabilities": set()
            }
            for agent in self.agents if "attacker" in agent
        }

    def reset(self, seed=None, options= None):
        #restoring the agents
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        #applying the partial deterioration of the satellite system
        self.state = {
            "power": max(0.9, 1.0 - np.random.uniform(0.01, 0.03)),
            "memory": max(0.85, 1.0 - np.random.uniform(0.02, 0.04)),
            "control": max(0.85, 1.0 - np.random.uniform(0.02, 0.04)),
            "software_health": max(0.85, 1.0 - np.random.uniform(0.02, 0.04)),
            "radiation_level": min(100.0, np.random.uniform(0.5, 1.5)),
            "under_attack": 0.0,
            "signal_strength": max(-120, min(-50, -80 - np.random.uniform(1, 5))),
            "communication_status": max(0.95, 1.0 - np.random.uniform(0.01, 0.02)),
            "battery_health": max(0.9, 1.0 - np.random.uniform(0.01, 0.03)),
            "thermal_status": max(-200, min(200, np.random.uniform(-5, 5))),
            "orbit_deviation": min(5.0, np.random.uniform(0.02, 0.08)),
            "neighbor_state_trust": max(0.5, 1.0 - np.random.uniform(0.01, 0.05)),
            "data_queue_size": max(0, min(100, 10.0 + np.random.uniform(-5, 5))),
            "redundancy_status": max(0.6, 1.0 - np.random.uniform(0.01, 0.03))
        }

        
        self._apply_environment_decay()

        self.rewards = {agent:0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.defender_risk = {agent: {} for agent in self.agents}

        self.num_steps = 0

        return self.observe(self.agent_selection)


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

                # Add vulnerability analysis
                vulnerabilities = {
                    system: {
                        "current_value": self.state[system],
                        "success_rate": (self.attack_history[system]["successful_attacks"] / 
                                       max(1, self.attack_history[system]["attack_frequency"])),
                        "alert_level": self.attack_history[system]["alert_level"]
                    }
                    for system in self.attack_history
                }
                
                obs.update({
                    "vulnerability_analysis": vulnerabilities,
                    "previous_successes": self.attacker_memory[agent]["successful_attacks"][-5:],
                    "identified_weaknesses": list(self.attacker_memory[agent]["identified_vulnerabilities"])
                })

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

                # Add defense history and alerts
                alerts = {
                    system: {
                        "alert_level": self.attack_history[system]["alert_level"],
                        "last_attack": self.attack_history[system]["last_attack_step"],
                        "attack_frequency": self.attack_history[system]["attack_frequency"]
                    }
                    for system in self.attack_history
                }
                
                obs.update({
                    "system_alerts": alerts,
                    "defense_history": {
                        "recent_successes": self.defender_memory[agent]["successful_defenses"][-5:],
                        "high_risk_systems": list(self.defender_memory[agent]["alert_systems"])
                    }
                })

            return obs
        

    def step(self, action):
        print(f"Step action: {action}, type: {type(action)}")
        if not isinstance(action, np.ndarray):
            raise ValueError(f"Action must be numpy array, got {type(action)}")
        if self.dones[self.agent_selection]:
            return self.was_done_step(action)
            
        agent = self.agent_selection
        self.num_steps += 1
        previous_state = self.state.copy()

        # --- Advanced features integration ---
        self._validate_observation_spaces()
        self._validate_system_dependencies()
        self._enforce_state_bounds()
        self._update_performance_metrics()
        self._coordinate_defense_strategy()
        self.analyze_attack_patterns()
        # -------------------------------------

        # Apply action based on agent type
        action_success = False
        if "attacker" in agent:
            # Select target based on history and current vulnerabilities
            target_system = self._select_attack_target(agent)
            action_success = self._attacker_action(agent, action)
            self._update_attack_history(agent, target_system, action_success)
            self.apply_dependency_cascade(target_system)
        else:
            # Select defense based on alerts and history
            defense_target = self._select_defense_target(agent)
            action_success = self._defender_action(agent, action)
            self.apply_defense_reinforcement(defense_target)
        
        # Decay alert levels over time
        self._decay_alert_levels()
        
        self._apply_environment_decay()
        
        # Calculate rewards
        if "attacker" in agent:
            reward = self.compute_attacker_reward(agent)
        else:
            reward = self.compute_defender_reward(agent)

        # Log action
        self.log_action(agent, action, action_success)

        self.rewards[agent] = reward

        
        # Update info
        self.infos[agent].update({
            "action_success": action_success,
            "state_change": {
                k: round(self.state[k] - previous_state[k], 3)
                for k in self.state.keys()
            },
            "system_health": {
                k: round(v, 3) for k, v in self.state.items()
            }
        })
        
        self._update_done_status()
        
        # Handle episode termination
        if any(self.dones.values()):
            self._handle_episode_end()
            
        # Update agent selection
        self.agent_selection = self.agent_selector.next()
     
        
        # Accumulate rewards
        if not hasattr(self, 'cumulative_rewards'):
            self.cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulative_rewards[agent] += reward
        
        self._clear_rewards()

        if hasattr(self, "render_mode") and self.render_mode == 'human':
            self.render()
            
        return self._last()
    
    def _update_attack_history(self, agent, target_system, success):
        """Update attack history and alert levels"""
        self.attack_history[target_system]["last_attack_step"] = self.num_steps
        self.attack_history[target_system]["attack_frequency"] += 1
        
        if success:
            self.attack_history[target_system]["successful_attacks"] += 1
            self.attacker_memory[agent]["successful_attacks"].append({
                "step": self.num_steps,
                "system": target_system,
                "state": self.state[target_system]
            })
            self.attacker_memory[agent]["identified_vulnerabilities"].add(target_system)
            
            # Increase alert level for this and related systems
            self._raise_system_alerts(target_system)
        else:
            self.attacker_memory[agent]["failed_attacks"].append({
                "step": self.num_steps,
                "system": target_system
            })
    def _decay_alert_levels(self):
        """
        Gradually reduce alert levels for all systems each step.
        """
        decay_rate = 0.05  # Decay 5% per step
        for system in self.attack_history:
            self.attack_history[system]["alert_level"] = max(
                0.0, self.attack_history[system]["alert_level"] - decay_rate
            )
    def _validate_observation_spaces(self):
        """Validate and fix observation space definitions"""
        for agent in self.agents:
            obs_space = self.observation_spaces[agent]
            
            # Fix the typo: 'raditaion_level' -> 'radiation_level'
            if 'raditaion_level' in obs_space.spaces:
                obs_space.spaces['radiation_level'] = obs_space.spaces.pop('raditaion_level')
                
    def was_done_step(self, action):
        if self.dones[self.agent_selection]:
            return self._last()
        self.agent_selection = self.agent_selector.next()
        self._clear_rewards()
        return self._last()
    
    def _select_defense_target(self, agent):
        """Helper method to select defense target based on system state"""
        obs = self.observe(agent)
        priorities = [
            ("power", obs["power"]),
            ("software_health", obs["software_health"]), 
            ("control", obs["control"]),
            ("memory", obs["memory"]),
            ("communication_status", obs["communication_status"]),
            ("battery_health", obs["battery_health"])
        ]
        
        # Select system with lowest health as defense target
        return min(priorities, key=lambda x: x[1])[0]
    
    def _select_attack_target(self, agent):
        """
        Select the most vulnerable system for the attacker based on:
        - Highest vulnerability (lowest value or highest alert)
        - Past attack success
        """
        obs = self.observe(agent)
        vulnerabilities = {
            system: 1 - obs[system] if system in obs else 0
            for system in [
                "power", "memory", "control", "software_health", "communication_status",
                "battery_health", "signal_strength", "radiation_level", "thermal_status",
                "orbit_deviation", "neighbor_state_trust", "data_queue_size", "redundancy_status"
            ]
        }
        # Weight by alert level (more alert = more likely to be defended, so less attractive)
        for system in vulnerabilities:
            alert = self.attack_history[system]["alert_level"]
            vulnerabilities[system] *= (1 - 0.5 * alert)
        # Pick the system with the highest vulnerability score
        return max(vulnerabilities.items(), key=lambda x: x[1])[0]

    def _last(self):
        #return the last observation, reward, done, and infor
        agent = self.agent_selection
        observation = self.observe(agent)
        return (
            observation,
            self.rewards[agent],
            self.dones[agent],
            self.infos[agent]
        )
    
    def _clear_rewards(self):
        # clear rewards of all the agents except the current one
        for agent in self.agents:
            self.rewards[agent] = 0.0

    def _accumulate_rewards(self):
        # accumulate rewards for the current agent
        if not hasattr(self, 'cumulative_rewards'):
            self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.cumulative_rewards[self.agent_selection] += self.rewards[self.agent_selection]
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
        defense_strength = float(action[0])

        obs = self.observe(agent)

        roots = ["power", "battery_health", "radiation_level"]
        thermal_related = ["thermal_status"]
        software_related = ["software_health", "memory"]
        control_related = ["control"]
        communication_related = ["communication_status", "signal_strength"]
        redundancy_related = ["redundancy_status"]
        data_related = ["data_queue_size"]
        orbit_related = ["orbit_deviation"]
        trust_related = ["neighbor_trust"]

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
        self._apply_defense_effects(agent, action, defense_target)


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
            attack_success = np.random.rand() < params["success_prob"]

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


    def _apply_defense_effects(self, agent, action, defense_target):
        action_value = float(action[0])  # Now this will work
        
        defense_params = {
            "power": {
                "improvement": 0.2 * action_value,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "battery_health": {
                "improvement": 0.2 * action_value,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "radiation_level": {
                "improvement": -2.0,
                "cost": 0.15,
                "success_prob": 0.7
            },
            "memory": {
                "improvement": 0.15,
                "cost": 0.1,
                "success_prob": 0.9
            },
            "software_health": {
                "improvement": 0.2,
                "cost": 0.1,
                "success_prob": 0.85
            },
            "communication_status": {
                "improvement": 0.15,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "signal_strength": {
                "improvement": 5.0,
                "cost": 0.1,
                "success_prob": 0.75
            },
            "thermal_status": {
                "improvement": -10.0,
                "cost": 0.1,
                "success_prob": 0.9
            },
            "control": {
                "improvement": 0.2,
                "cost": 0.15,
                "success_prob": 0.8
            },
            "orbit_deviation": {
                "improvement": -0.1,
                "cost": 0.2,
                "success_prob": 0.7
            },
            "neighbor_state_trust": {
                "improvement": 0.15,
                "cost": 0.05,
                "success_prob": 0.9
            },
            "data_queue_size": {
                "improvement": -10.0,
                "cost": 0.1,
                "success_prob": 0.8
            },
            "redundancy_status": {
                "improvement": 0.2,
                "cost": 0.15,
                "success_prob": 0.85
            }
        }

        defense_target = self._select_defense_target(agent)
        
        if defense_target in defense_params:
            params = defense_params[defense_target]
            success = np.random.rand() < params["success_prob"]
            
            if success:
                improvement = params["improvement"]
                
                if defense_target in ["radiation_level", "thermal_status", "data_queue_size"]:
                    self.state[defense_target] = max(
                        self.observation_spaces[agent][defense_target].low,
                        min(self.state[defense_target] + improvement, 
                            self.observation_spaces[agent][defense_target].high)
                    )
                elif defense_target == "signal_strength":
                    self.state[defense_target] = min(-50.0, 
                        max(-120.0, self.state[defense_target] + improvement))
                else:
                    self.state[defense_target] = max(0.0, 
                        min(1.0, self.state[defense_target] + improvement))
                    
                self.rewards[agent] += 1.0
                self.state["under_attack"] = max(0.0, self.state["under_attack"] - 0.2)
                
                self.apply_defense_reinforcement(defense_target)
                
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

        threats = {priority: [] for priority in priority_groups.keys()}

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
        print(f"[DEBUG] Attacker {agent} reward: {reward}, state: {self.state}")
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
        print(f"[DEBUG] Defender {agent} reward: {reward}, state: {self.state}")
        return np.clip(reward, -2.0, 2.0)
    
    def compute_reward(self, agent):
        if "attacker" in agent:
            return self.compute_attacker_reward(agent)
        elif "defender" in agent:
            return self.compute_defender_reward
        
        return 0.0
        
        
    def _apply_environment_decay(self):
        decay_rates = {
            "power": np.random.uniform(0.001, 0.005),
            "memory": np.random.uniform(0.002, 0.004),
            "control": np.random.uniform(0.001, 0.003),
            "software_health": np.random.uniform(0.001, 0.004),
            "battery_health": np.random.uniform(0.001, 0.003),
            "communication_status": np.random.uniform(0.001, 0.002),
            "redundancy_status": np.random.uniform(0.001, 0.003)
        }

        env_effects = {
            "radiation_level": np.random.uniform(0.1, 0.3),
            "thermal_status": np.random.uniform(-2.0, 2.0),
            "signal_strength": np.random.uniform(-0.5, 0.5),
            "orbit_deviation": np.random.uniform(0.001, 0.005),
            "data_queue_size" : np.random.uniform(-1.0, 1.0)
        }

        for system, rate in decay_rates.items():
            self.state[system] = max(0.0, min(1.0, self.state[system] - rate))

        for system, effect in env_effects.items():
            if system == "radiation_level":
                self.state[system] = min(100.0, max(0.0, self.state[system] + effect))
            elif system == "thermal_status":
                self.state[system] = min(200.0, max(-200.0, self.state[system] + effect))
            elif system == "signal_strength":
                self.state[system] = min(-50.0, max(-120.0, self.state[system] + effect))
            elif system == "data_queue_size":
                self.state[system] = min(100.0, max(0.0, self.state[system] + effect))
            else:
                self.state[system] = min(1.0, max(0.0, self.state[system] + effect))

        self.infos["environment_decay"] = {
            "decay_rates": decay_rates,
            "environmental_effects": env_effects
        }


    def _update_done_status(self):
        """ Update episode termination conditions"""

        critical_thresholds = {"power": 0.1, "software_health": 0.1, "control": 0.1, "memory": 0.1}

        critical_failure = any(
            self.state[system] < threshold
            for system, threshold in critical_thresholds.items()

        )

        environmental_failure = (
            self.state["radiation_level"] > 95.0 or 
            abs(self.state["thermal_status"]) > 180.0 or
            self.state["orbit_deviation"] > 0.9
        )

        episode_done = critical_failure or environmental_failure

        self.dones = {agent: episode_done for agent in self.agents}

        if episode_done:
            termination_info = {
                "critical_failure": critical_failure,
                "environmental_failure": environmental_failure,
                "final_state": self.state.copy()
            }
            self.infos["termination"] = termination_info


    def render(self, mode="human"):
        """Displays the current environment state for debugging and tracking."""
        status_indicators = {
            "Optimal": "🟢",
            "Warning": "🟡",
            "Critical": "🟠"
        }
         
        print("\n 📝 Satellite Defense System Status 📝")

        #Critical Systems
        print("Critical Systems:")
        print(f"Power: {status_indicators['Critical'] if self.state['power'] < 0.3 else status_indicators['Warning'] if self.state['power'] < 0.7 else status_indicators['Optimal']} {self.state['power']:.2f}")
        print(f"Software: {status_indicators['Critical'] if self.state['software_health'] < 0.3 else status_indicators['Warning'] if self.state['software_health'] < 0.7 else status_indicators['Optimal']} {self.state['software_health']:.2f}")
        print(f"Control: {status_indicators['Critical'] if self.state['control'] < 0.3 else status_indicators['Warning'] if self.state['control'] < 0.7 else status_indicators['Optimal']} {self.state['control']:.2f}")
        

        #Environmental Conditions:
        print("\nEnvironmental Status:")
        print(f"Radiation: {status_indicators['Critical'] if self.state['radiation_level'] > 80 else status_indicators['Warning'] if self.state['radiation_level'] > 50 else status_indicators['Optimal']} {self.state['radiation_level']:.1f}")
        print(f"Thermal: {status_indicators['Critical'] if abs(self.state['thermal_status']) > 150 else status_indicators['Warning'] if abs(self.state['thermal_status']) > 100 else status_indicators['Optimal']} {self.state['thermal_status']:.1f}")
        
        #Attack Status
        print(f"\n Attack Status: {'⚠️ Under Attack' if self.state['under_attack'] > 0.5 else '✅ Secure'}")

        #Recent Action
        if hasattr(self, 'last_action'):
            print("\nRecent Actions:")
            print(f"Agent: {self.agent_selection}")
            print(f"Action: {self.last_action}")
            # Print the cumulative reward for the current agent
            if hasattr(self, 'cumulative_rewards'):
                print(f"Reward: {self.cumulative_rewards.get(self.agent_selection, 0.0):.2f}")
            else:
                print(f"Reward: {self.rewards[self.agent_selection]:.2f}")

            print("\n-------------------------------------------------------")

    def log_action(self, agent, action, success):
        if not hasattr(self, 'action_history'):
            self.action_history = []

        action_log = {
            'step': self.num_steps,
            'agent': agent,
            'action': action,
            'success': success,
            "reward": self.rewards[agent],
            'state_change': {
                k : round(v, 3) for k, v, in self.state.items()
            }
        }

        self.action_history.append(action_log)
        self.last_action = action

    def get_env_report(self):
        return {
            "episode_length": self.num_steps,
            "final_state": self.state.copy(),
            "cumulative_rewards": {
                agent: self.cumulative_rewards.get(agent, 0.0)
                for agent in self.agents
            },
            "action_history": self.action_history,
            "termination_info": self.infos.get("termination", None)
        }

    # Add PettingZoo API properties for compliance
    @property
    def observation_space(self):
        return self.observation_spaces[self.agents[0]]

    @property
    def action_space(self):
        return self.action_spaces[self.agents[0]]

    # Implement missing episode end handler
    def _handle_episode_end(self):
        """
        Finalize logs, print reports, and clean up at episode end.
        """
        print("\n--- Episode Ended ---")
        print(f"Total Steps: {self.num_steps}")
        print("Final State:")
        for k, v in self.state.items():
            print(f"  {k}: {v:.3f}")
        print("Cumulative Rewards:")
        for agent in self.agents:
            print(f"  {agent}: {self.cumulative_rewards.get(agent, 0):.2f}")
        if "termination" in self.infos:
            print("Termination Info:", self.infos["termination"])
        print("\n EPISODE END")
    # Fix reward reporting in close() to use cumulative_rewards
    def close(self):
        """Cleanup and Final Report"""
        print("\n ------ Final Report -----")

        print("Episode Statistics:")
        print(f"Total Steps: {self.num_steps}")
        print(f"Episode Length: {self.num_steps / len(self.agents):.1f} rounds")

        print("\nAgent Performance:")
        for agent in self.agents:
            print(f"{agent}: Total Reward = {self.cumulative_rewards.get(agent, 0.0):.2f}")

        print("\nFinal System Health:")
        for system, value in self.state.items():
            print(f"{system}: {value:.2f}")

        if "termination" in self.infos:
            print("\nTermination Reason:")
            for reason, occured in self.infos["termination"].items():
                if occured and reason != "final_state":
                    print(f"-{reason}")

        print("\n=== Report Done Successfully ===")

    def _validate_system_dependencies(self):
        """Validate system states don't violate physical constraints"""
        constraints_violated = []
        
        # Power constraints: if power is very low, other systems should be affected
        if self.state["power"] < 0.2:
            # Communication should be impacted
            if self.state["communication_status"] > 0.6:
                self.state["communication_status"] *= 0.8
                constraints_violated.append("power_communication_dependency")
            
            # Control systems should be impacted
            if self.state["control"] > 0.5:
                self.state["control"] *= 0.9
                constraints_violated.append("power_control_dependency")
        
        # Thermal constraints: extreme temperatures affect electronics
        if abs(self.state["thermal_status"]) > 150:
            degradation_factor = 0.98
            self.state["memory"] *= degradation_factor
            self.state["software_health"] *= degradation_factor
            constraints_violated.append("thermal_electronics_dependency")
        
        # Radiation constraints: high radiation affects memory and software
        if self.state["radiation_level"] > 80:
            radiation_factor = 0.99
            self.state["memory"] *= radiation_factor
            self.state["software_health"] *= radiation_factor
            constraints_violated.append("radiation_electronics_dependency")
        
        # Communication constraints: poor signal affects trust
        if self.state["signal_strength"] < -100:
            self.state["neighbor_state_trust"] *= 0.95
            constraints_violated.append("signal_trust_dependency")
        
        # Data queue constraints: full queue affects performance
        if self.state["data_queue_size"] > 90:
            self.state["software_health"] *= 0.98
            self.state["memory"] *= 0.99
            constraints_violated.append("queue_performance_dependency")
        
        return constraints_violated


    def _enforce_state_bounds(self):
        """Ensure all state values remain within valid bounds"""
        bounds = {
            "power": (0.0, 1.0),
            "memory": (0.0, 1.0),
            "control": (0.0, 1.0),
            "software_health": (0.0, 1.0),
            "radiation_level": (0.0, 100.0),
            "under_attack": (0.0, 1.0),
            "signal_strength": (-120.0, -50.0),
            "communication_status": (0.0, 1.0),
            "battery_health": (0.0, 1.0),
            "thermal_status": (-200.0, 200.0),
            "orbit_deviation": (0.0, 1.0),
            "neighbor_state_trust": (0.0, 1.0),
            "data_queue_size": (0.0, 100.0),
            "redundancy_status": (0.0, 1.0)
        }
        
        for key, (min_val, max_val) in bounds.items():
            if key in self.state:
                self.state[key] = np.clip(self.state[key], min_val, max_val)

    # 5. Performance metrics tracking
    def _update_performance_metrics(self):
        """Track performance metrics for analysis"""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                'system_health_history': [],
                'attack_frequency_history': [],
                'defense_effectiveness_history': [],
                'critical_events': []
            }
        
        # System health snapshot
        current_health = {
            'critical_avg': np.mean([
                self.state["power"], 
                self.state["software_health"], 
                self.state["control"]
            ]),
            'overall_avg': np.mean([
                v for k, v in self.state.items() 
                if k not in ["radiation_level", "thermal_status", "under_attack", 
                            "signal_strength", "orbit_deviation", "data_queue_size"]
            ]),
            'step': self.num_steps
        }
        self.performance_metrics['system_health_history'].append(current_health)
        
        # Track critical events
        if current_health['critical_avg'] < 0.3:
            self.performance_metrics['critical_events'].append({
                'type': 'critical_systems_low',
                'step': self.num_steps,
                'severity': 1.0 - current_health['critical_avg']
            })
        
        if self.state["radiation_level"] > 90:
            self.performance_metrics['critical_events'].append({
                'type': 'radiation_critical',
                'step': self.num_steps,
                'value': self.state["radiation_level"]
            })

    # 6. Agent coordination for defenders
    def _coordinate_defense_strategy(self):
        """Coordinate defense strategy among defender agents"""
        if not hasattr(self, 'defense_coordination'):
            self.defense_coordination = {
                'priority_assignments': {},
                'coordination_bonus': 0.0,
                'last_coordination_step': 0
            }
        
        # Identify systems needing defense
        vulnerable_systems = []
        for system, value in self.state.items():
            if system in ["power", "software_health", "control", "memory", 
                        "communication_status", "battery_health", "redundancy_status"]:
                if value < 0.6:
                    vulnerability = 1.0 - value
                    vulnerable_systems.append((system, vulnerability))
        
        # Sort by vulnerability
        vulnerable_systems.sort(key=lambda x: x[1], reverse=True)
        
        # Assign priorities to defenders
        defender_agents = [agent for agent in self.agents if "defender" in agent]
        
        for i, agent in enumerate(defender_agents):
            if i < len(vulnerable_systems):
                assigned_system = vulnerable_systems[i][0]
                self.defense_coordination['priority_assignments'][agent] = assigned_system
            else:
                # Assign to most critical system if no unique assignment
                if vulnerable_systems:
                    self.defense_coordination['priority_assignments'][agent] = vulnerable_systems[0][0]
        
        # Calculate coordination bonus
        if len(set(self.defense_coordination['priority_assignments'].values())) == len(vulnerable_systems):
            self.defense_coordination['coordination_bonus'] = 0.1  # Good coordination
        else:
            self.defense_coordination['coordination_bonus'] = 0.0
        
        self.defense_coordination['last_coordination_step'] = self.num_steps

    def analyze_attack_patterns(self):
        """Analyze attack patterns for strategic insights"""
        if not hasattr(self, 'attack_analysis'):
            self.attack_analysis = {
                'pattern_detection': {},
                'threat_prediction': {},
                'adaptation_counter': 0
            }
        
        # Analyze recent attack history
        recent_attacks = []
        for system, history in self.attack_history.items():
            if history['last_attack_step'] is not None:
                steps_ago = self.num_steps - history['last_attack_step']
                if steps_ago <= 10:  # Recent attacks
                    recent_attacks.append({
                        'system': system,
                        'steps_ago': steps_ago,
                        'frequency': history['attack_frequency'],
                        'success_rate': history['successful_attacks'] / max(1, history['attack_frequency'])
                    })
        
        # Detect patterns
        if len(recent_attacks) >= 3:
            # Sort by recency
            recent_attacks.sort(key=lambda x: x['steps_ago'])
            
            # Check for escalation pattern
            systems_attacked = [attack['system'] for attack in recent_attacks[-3:]]
            
            if len(set(systems_attacked)) == 1:
                # Focused attack detected
                target_system = systems_attacked[0]
                self.attack_analysis['pattern_detection']['focused_attack'] = {
                    'target': target_system,
                    'intensity': len([a for a in recent_attacks if a['system'] == target_system]),
                    'detected_step': self.num_steps
                }
            
            # Check for distributed attack pattern
            if len(set(systems_attacked)) == len(systems_attacked):
                self.attack_analysis['pattern_detection']['distributed_attack'] = {
                    'targets': systems_attacked,
                    'detected_step': self.num_steps
                }
        
        # Predict next likely targets
        threat_scores = {}
        for system, history in self.attack_history.items():
            # Base threat on success rate and alert level
            if history['attack_frequency'] > 0:
                success_rate = history['successful_attacks'] / history['attack_frequency']
                
                recency_factor = 1.0
                if history['last_attack_step'] is not None:
                    steps_since = self.num_steps - history['last_attack_step']
                    recency_factor = max(0.1, 1.0 - (steps_since / 20.0))
                
                threat_scores[system] = success_rate * recency_factor * (1.0 - history['alert_level'])
        
        # Store top 3 predicted targets
        if threat_scores:
            sorted_threats = sorted(threat_scores.items(), key=lambda x: x[1], reverse=True)
            self.attack_analysis['threat_prediction']['top_targets'] = sorted_threats[:3]
            self.attack_analysis['threat_prediction']['updated_step'] = self.num_steps
        
        # Analyze attack timing patterns
        attack_intervals = []
        for system, history in self.attack_history.items():
            if len(history.get('attack_timestamps', [])) >= 2:
                timestamps = history['attack_timestamps']
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                attack_intervals.extend(intervals)
        
        if attack_intervals:
            avg_interval = sum(attack_intervals) / len(attack_intervals)
            self.attack_analysis['pattern_detection']['timing_pattern'] = {
                'average_interval': avg_interval,
                'last_calculated': self.num_steps,
                'sample_size': len(attack_intervals)
            }
        
        # Detect adaptation behavior
        if hasattr(self, 'defense_changes') and self.defense_changes:
            recent_defense_changes = [
                change for change in self.defense_changes 
                if self.num_steps - change.get('step', 0) <= 5
            ]
            
            if recent_defense_changes:
                self.attack_analysis['adaptation_counter'] += 1
                self.attack_analysis['pattern_detection']['adaptation_detected'] = {
                    'counter': self.attack_analysis['adaptation_counter'],
                    'recent_changes': len(recent_defense_changes),
                    'detected_step': self.num_steps
                }
        
        # Calculate overall threat level
        if threat_scores:
            max_threat = max(threat_scores.values())
            avg_threat = sum(threat_scores.values()) / len(threat_scores)
            
            overall_threat_level = "LOW"
            if max_threat > 0.7:
                overall_threat_level = "CRITICAL"
            elif max_threat > 0.5:
                overall_threat_level = "HIGH"
            elif avg_threat > 0.3:
                overall_threat_level = "MEDIUM"
            
            self.attack_analysis['threat_prediction']['overall_level'] = overall_threat_level
            self.attack_analysis['threat_prediction']['max_score'] = max_threat
            self.attack_analysis['threat_prediction']['avg_score'] = avg_threat
        
        return self.attack_analysis