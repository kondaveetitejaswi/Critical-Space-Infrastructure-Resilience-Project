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


    def reset(self, seed=None, options= None):
        #restoring the agents
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()

        #applying the partial deterioration of the satellite system
        self.state = {
            "power": max(0.9, self.state.get("power", 1.0) - np.random.uniform(0.01, 0.03)),
            "memory": max(0.85, self.state.get("memory", 1.0) - np.random.uniform(0.02, 0.04)),
            "control": max(0.85, self.state.get())
        }

    # def reset(self, seed=None, options=None):
    #     self.agents = self.possible_agents[:]
    #     self.agent_selector = agent_selector(self.agents)
    #     self.agent_selection = self.agent_selector.reset()

    #     self.state = {
    #         "power": 1.0,
    #         "memory": 1.0,
    #         "control": 1.0,
    #         "software_health": 1.0,
    #         "radiation_level": 0.0,
    #         "under_attack": 0.0,

    #         "DoS_attack": 0.0,
    #         "Memory_corruption": 0.0,
    #         "Spoof_control": 0.0,
    #         "Inject_bug": 0.0,
    #         "Radiation_surge": 0.0,

    #         "Boost_power": 1.0,
    #         "Repair_memory": 1.0,
    #         "Stabilize_control": 1.0,
    #         "Reset_attack_flag": 1.0
    #     }

    #     self.rewards = {agent: 0.0 for agent in self.agents}
    #     self.dones = {agent: False for agent in self.agents}
    #     self.infos = {agent: {} for agent in self.agents}



    def observe(self, agent):
            return {k: self.state[k] for k in self.observation_spaces[agent].spaces.keys()}
        
    def step(self, action):
        agent = self.agent_selection

        if self.dones[agent]:
            self._was_done_step(action)
            return
        
        value = action[0]

        if "attacker" in agent:
            self._attacker_action(action)
        else:
            self._defender_action(action)

        self._apply_radiation_decay()
        self._update_done_status()

        for agent in self.agents:
            self.rewards[agent] = self._get_reward(agent)

        self.agent_selection = self.agent_selector.next()

    def _attacker_action(self, agent, intensity):
        attack_map = {                
        }

        #simulate the attack strategies

    def _defender_action(self, agent, effort):
        defense_map = {
        }

    def _apply_radiation_decay(self):
        decay_rate = 0.01
        
    def _get_reward(self, agent):
        health = ()/#no.of variables


    def _update_done_status(self):


    def render(self):

        
    def close(self):
        pass


# have to work on the step function, attacket action, defender action,
# and the radiation decay function, getting reward, update status and the render function            

