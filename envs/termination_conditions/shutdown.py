import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class Shutdown(BaseTerminationCondition):
    """
    Shutdown
    End up the simulation if the aircraft is shutdown.
    """

    def __init__(self, config, device):
        super().__init__(config)
        self.device = torch.device(device)
        self.distance_limit = getattr(self.config, 'distance_limit', 200)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is shutdown.

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
        mask1 = env.blood[ego_agents] <= 0
        mask2 = env.blood[enm_agents] <= 0
        done[ego_agents] = mask2 & (~mask1)
        done[enm_agents] = mask2 & (~mask1)
        bad_done[ego_agents] = mask1
        bad_done[enm_agents] = mask1
        if torch.any(bad_done):
            self.log(f'aircraft is shutdown!')
            print(torch.sum(bad_done), 'aircraft is shutdown!')
        if torch.any(done):
            self.log(f'enemy is shutdown!')
            print(torch.sum(done), 'enemy aircraft is shutdown!')
        return bad_done, done, exceed_time_limit, info
