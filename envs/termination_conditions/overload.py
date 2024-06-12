import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class Overload(BaseTerminationCondition):
    """
    Overload
    End up the simulation if acceleration are too high.
    """

    def __init__(self, config):
        super().__init__(config)
        self.acceleration_limit = getattr(config, 'acceleration_limit', 300.0)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if acceleration are too high.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        bad_done = self._judge_overload(env)
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'acceleration is too high!')
            print(torch.sum(bad_done), 'acceleration is too high!')
        return bad_done, done, exceed_time_limit, info

    def _judge_overload(self, env):
        ax, ay, az = env.model.get_acceleration()
        acceleration = ax ** 2 + ay ** 2 + az ** 2
        acceleration = torch.sqrt(acceleration)
        flag_overload = (acceleration - self.acceleration_limit) > 0
        return flag_overload
