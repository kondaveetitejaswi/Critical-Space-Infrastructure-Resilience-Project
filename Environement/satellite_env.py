import numpy as np
from pettingzoo import AECEnv
from gymnasium.spaces import Box, Dict
from pettingzoo.utils import agent_selector

class SatelliteDefenseEnv(AECEnv):
    metadata = {"render_modes": ['human'], "name": "satellite_defense_v0", "is_multiagent": True}

    