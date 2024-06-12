import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class LowAltitude(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500.0)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if altitude are too low.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        npos, epos, altitude = env.model.get_position()
        bad_done = (altitude - self.altitude_limit) < 0
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'altitude is too low!')
            print(torch.sum(bad_done), 'altitude is too low!')
        return bad_done, done, exceed_time_limit, info
