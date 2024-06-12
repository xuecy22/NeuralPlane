import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition

class Crash(BaseTerminationCondition):
    """
    Crash
    End up the simulation if the aircraft is crashed.
    """

    def __init__(self, config, device):
        super().__init__(config)
        self.device = torch.device(device)
        self.distance_limit = getattr(self.config, 'distance_limit', 200)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is crashed.

        Args:
            env: environment instance

        Returns:Q
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        bad_done = torch.zeros(env.n, dtype=torch.bool, device=self.device)
        done =  torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        ego_agents = torch.arange(env.num_envs, device=self.device) * env.num_agents
        enm_agents = ego_agents + 1
        ego_npos = env.s[ego_agents, 0]
        ego_epos = env.s[ego_agents, 1]
        ego_altitude = env.s[ego_agents, 2]
        enm_npos = env.s[enm_agents, 0]
        enm_epos = env.s[enm_agents, 1]
        enm_altitude = env.s[enm_agents, 2]
        distance = (ego_npos - enm_npos) ** 2 + (ego_epos - enm_epos) ** 2 + (ego_altitude - enm_altitude) ** 2
        bad_done[ego_agents] = distance <= self.distance_limit ** 2
        bad_done[enm_agents] = distance <= self.distance_limit ** 2
        if torch.any(bad_done):
            self.log(f'aircraft is crashed!')
            print(torch.sum(bad_done), 'aircraft is crashed!')
        return bad_done, done, exceed_time_limit, info
