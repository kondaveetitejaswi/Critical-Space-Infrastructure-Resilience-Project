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
                    "orbit_deviation": self.state["orbit_deviation"],
                    "neighbor_state_trust": self.state["neighbor_state_trust"],
                    "data_queue_size": self.state["data_queue_size"],
                    "redundancy_status": self.state["redundancy_status"]   
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
            "memory_status": 1 - obs["memory"],
            "software_attack": 1 - obs["software_health"],
            "communication_attack": 1 - obs["communication_status"],
            "trust_attack": 1 - obs["neighbor_state_trust"],
            "data_overflow_attack": 1 - obs["data_queue_size"],
            "redundancy_attack": 1 - obs["redundancy_status"],
            "power_attack": 1 - obs["power"],
            "control_attack": 1 - obs["control"]
        }

        #choose attack type based on action value and vulnerabilities
        sorted_vulnerabilities = sorted(vulnerabilities.items(), key = lambda x: x[1], reverse = True)

        vulnerability_threshold = 0.6
        critical_threshold = 0.3

        if action_value < 0.15:
            # targeting systems that are critically vulnerable first
            critical_systems = [sys for sys, val in sorted_vulnerabilities if val < critical_threshold]
            if critical_systems:
                attack_type = critical_systems[0]
            else:
                attack_type = sorted_vulnerabilities[0][0]
        elif action_value < 0.25:
            attack_type = sorted_vulnerabilities[1][0] if len(sorted_vulnerabilities) > 1 else sorted_vulnerabilities[0][0]

        else:
            # Weighted distribution based on current system state and potential impact
            if obs["ememory"] < vulnerability_threshold and obs["software_health"] < vulnerability_threshold:
                attack_type = "memory_status" if action_value < 0.4 else "software_attack"
            elif obs["communication_status"] < vulnerability_threshold and obs["neighbor_state_trust"] < vulnerability_threshold:
                attack_type = "communication_attack" if action_value < 0.5 else "trust_attack"
            elif obs["power"] < vulnerability_threshold:
                attack_type = "power_attack" if action_value < 0.6 else "control_attack"
            elif obs["data_queue_size"] > 80:
                attack_type = "data_overflow_attack"
            elif obs["redundancy_status"] < vulnerability_threshold:
                attack_type = "redundancy_attack"
            else:
                attack_type = "control_attack"


    def _defender_action(self, agent, action):
        # New comprehensive defender action method
        defense_value = action[0] 

        obs = self.observe(agent)


    def _apply_attack_decay(self, agent, action):
        attack_success = False

        if action == "DoS_attack":
            success_prob = 0.4 if self.state["communication_status"] < 0.7 else 0.8
            attack_success = np.random.rand() < success_prob
            if attack_success:
                self.state["communication_status"] = max(0.0, self.state["communication_status"] - 0.2)
                self.state["under_attack"] = 1.0
                self.rewards[agent] += 1.0

        elif action == "Memory_corruption":
            success_prob = 0.5 if self.state["memory"] < 0.6 else 0.9
            attack_success = np.random.rand() < success_prob
            if attack_success:
                self.state["memory"] = max(0.0, self.state["memory"] - 0.2)
                self.state["under_attack"] = 1.0
                self.rewards[agent] += 1.0

        return attack_success


    def _update_attacker_strategy(self):
        """Attackers analyze past attack efficiency and adjust targeting logic."""
        attack_focus = {}  
        
        for attacker in self.agents:
            if "attacker" in attacker:
                attack_focus[attacker] = {
                    "communication_status": 1 - self.state["communication_status"],
                    "software_health": 1 - self.state["software_health"],
                    "memory": 1 - self.state["memory"]
                }

        # Attackers coordinate by prioritizing most vulnerable areas
        avg_weakness = {
            key: sum(values[key] for values in attack_focus.values()) / len(attack_focus)
            for key in attack_focus[list(attack_focus.keys())[0]]
        }

        self.attack_priority = max(avg_weakness, key=avg_weakness.get)  # Most exploitable target

    def _apply_defense_effects(self, agent, action):
        """Defenders reinforce weak spots to counter attack threats."""
        if action == "Increase_shielding":
            self.state["radiation_level"] = max(0.0, self.state["radiation_level"] - 2.0)

        elif action == "Boost_communication":
            self.state["communication_status"] = min(1.0, self.state["communication_status"] + 0.1)

        elif action == "Enhance_memory_integrity":
            self.state["memory"] = min(1.0, self.state["memory"] + 0.1)
        
    def _share_risk_alerts(self):
        """Defenders exchange vulnerability information to improve threat anticipation."""
        alert_levels = {}

        for defender in self.agents:
            if "defender" in defender:
                alert_levels[defender] = {
                    "under_attack": self.state["under_attack"],
                    "radiation_level": self.state["radiation_level"],
                    "neighbor_state_trust": self.state["neighbor_state_trust"]
                }

        # Broadcast average alert level to all defenders
        avg_alert = {
            key: sum(values[key] for values in alert_levels.values()) / len(alert_levels)
            for key in alert_levels[list(alert_levels.keys())[0]]
        }

        for defender in alert_levels:
            self.defender_risk[defender] = avg_alert


    def _adjust_defense_priorities(self):
        """Defenders allocate defensive measures dynamically based on attack patterns."""
        threat_levels = {
            "communication_status": self.state["communication_status"] < 0.6,
            "memory": self.state["memory"] < 0.6,
            "radiation_level": self.state["radiation_level"] > 80.0
        }

        # Prioritize protecting the most damaged aspect
        self.defense_target = max(threat_levels, key=threat_levels.get)


    def _compute_reward(self, agent):
        if "attacker" in agent:
            return (1 - self.state["software_health"]) * 5  + (1 - self.state["communication_status"]) * 3
        
        elif "defender" in agent:
            return (self.state["software_health"] * 4) + (self.state["communication_status"] * 2) + (1 - self.state["radiation_level"] / 100) * 1

        
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