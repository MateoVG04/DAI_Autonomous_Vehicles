
"""
This module defines a SteeringAgent class that extends the BehaviourAgent from the CARLA simulator.
This module is used as a base for RL agents that require high-level navigation capabilities.
This module is adapted from the BehaviorAgent to allow for custom behavior parameters.
This module is used to expose the behaviour agents internal states for use in the RL environment.
"""



import random
import numpy as np
import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive, is_within_distance, compute_distance


class SteeringAgent(BehaviorAgent):
    """
    SteeringAgent class that extends BehaviorAgent to allow for custom behavior parameters.
    This class is used as a base for RL agents that require high-level navigation capabilities.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        super().__init__(vehicle, behavior=behavior, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

    @property
    def speed(self):
        return self._speed

    @property
    def speed_limit(self):
        return self._speed_limit

    @property
    def direction(self):
        return self._direction

    @property
    def behavior(self):
        return self._behavior

    def get_waypoints(self):
        plan = self.get_local_planner().get_plan()
        return [wp[0] for wp in plan]


    def run_step(self, debug=False):
        return super().run_step(debug=debug)