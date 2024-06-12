import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)
        self.min_alpha = getattr(config, 'min_alpha', -20)
        self.max_alpha = getattr(config, 'max_alpha', 45)
        self.min_beta = getattr(config, 'min_beta', -30)
        self.max_beta = getattr(config, 'max_beta', 30)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        alpha = env.model.get_AOA() * 180 / torch.pi
        beta = env.model.get_AOS() * 180 / torch.pi
        mask1 = (alpha < self.min_alpha) | (alpha > self.max_alpha)
        mask2 = (beta < self.min_beta) | (beta > self.max_beta)
        bad_done = mask1 | mask2
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'extreme state!')
            print(torch.sum(bad_done), 'extreme state!')
        return bad_done, done, exceed_time_limit, info
