import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, 'max_steps', 500)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        exceed_time_limit = (env.step_count - self.max_steps) >= 0
        bad_done = torch.zeros_like(exceed_time_limit)
        done = torch.zeros_like(exceed_time_limit)
        if torch.any(exceed_time_limit):
            self.log(f"step limits!")
            print(torch.sum(exceed_time_limit), "step limits!")
        return bad_done, done, exceed_time_limit, info
