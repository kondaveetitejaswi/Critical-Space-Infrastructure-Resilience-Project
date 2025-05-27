import numpy as np
from pettingzoo import AECEnv
from gymnasium.spaces import Box, Dict
from pettingzoo.utils import agent_selector

class SatelliteDefenseEnv(AECEnv):
    metadata = {"render_modes": ['human'], "name": "satellite_defense_v0", "is_multiagent": True}

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
                "power": Box(low=0.0, high=1.0, shape=()), #for power level
                "memory": Box(low=0.0, high=1.0, shape=()), #memory health
                "control": Box(low=0.0, high=1.0, shape=()), #control system stability
                "software_health": Box(low=0.0, high=1.0, shape=()), #software itnegrity
                "raditaion_level": Box(low=0.0, high=1.0, shape=()), #radiation exposure
                "under_attack": Box(low=0.0, high=1.0, shape=()),    #general attack status

                # attack states (each attack type gets a variable)

                "DoS_attack": Box(low=0.0, high=1.0, shape=()), #Denial of Service attack
                "Memory_corruption": Box(low=0.0, high=1.0, shape=()), # Memory corruption severity
                "Spoof_control": Box(low=0.0, high=1.0, shape=()), #control system spoofing
                "Inject_bug": Box(low=0.0, high=1.0, shape=()), #Bug injection effect
                "Radiation_surge": Box(low=0.0, high=1.0, shape=()), # Sudden radiation spikes

                # Defense system states (each defense action gets a variable)
                "Boost_power": Box(low=0.0, high=1.0, shape=()), # Power boost capability
                "Repair_memory": Box(low=0.0, high=1.0, shape=()), # Memory repair system
                "Stabilize_control": Box(low=0.0, high=1.0, shape=()), # Control stability mechanism
                "Reset_attack_flag": Box(low=0.0, high=1.0, shape=()) #Attack mitigation attack

            })
            for agent in self.agents
        }

        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        self.state = None
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        self.state = {
            "power": 1.0,
            "memory": 1.0,
            "control": 1.0,
            "software_health": 1.0,
            "radiation_level": 0.0,
            "under_attack": 0.0,

            "DoS_attack": 0.0,
            "Memory_corruption": 
            "Spoof_control":
            "Inject_bug":
            "Radiation_surge":

            "Boost_power":
            "Repair_memory":
            "Stabilize_control":
            "Reset_attack_flag":
        }

