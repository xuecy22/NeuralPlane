import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class LowSpeed(BaseTerminationCondition):
    """
    HighSpeed
    End up the simulation if speed are too low.
    """

    def __init__(self, config):
        super().__init__(config)
        self.min_velocity = getattr(config, 'min_velocity', 0.01)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if speed are too low.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        velocity = env.model.get_TAS() * 0.3048 / 340
        bad_done = (velocity - self.min_velocity) <= 0
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'speed is too low!')
            print(torch.sum(bad_done), 'speed is too low!')
        return bad_done, done, exceed_time_limit, info