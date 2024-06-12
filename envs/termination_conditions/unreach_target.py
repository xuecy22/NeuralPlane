import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class UnreachTarget(BaseTerminationCondition):
    """
    UnreachHeading
    End up the simulation if the aircraft didn't reach the target position in limited time.
    """

    def __init__(self, config, device):
        super().__init__(config)
        self.device = torch.device(device)
        self.max_check_interval = getattr(config, 'max_check_interval', 2500)
        self.min_check_interval = getattr(config, 'min_check_interval', 300)
    
    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target in limited time.

        Args:
            env: environment instance

        Returns:Q
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        npos, epos, altitude = env.model.get_position()
        vt = env.model.get_vt()
        check_time = env.step_count
        # 判断时间
        mask1 = check_time >= self.max_check_interval
        # mask2 = check_time >= self.min_check_interval
        # 判断是否到达target_npos
        mask3 = torch.abs(npos - task.target_npos) >= 100
        # 判断是否到达target_epos
        mask4  = torch.abs(epos - task.target_epos) >= 100
        # 判断是否到达target_altitude
        mask5 = torch.abs(altitude - task.target_altitude) >= 100
        # 当超过时间且未达到目标时，判断为True
        bad_done = mask1 & ((mask3 | mask4) | mask5)
        # 当达到目标且时间符合要求时，重新设置目标
        # done =  ((~((mask3 | mask4) | mask5)) & (~mask1)) & mask2
        done =  ((~((mask3 | mask4) | mask5)) & (~mask1))
        exceed_time_limit = torch.zeros_like(done)
        if torch.any(bad_done):
            self.log(f'unreach target!')
            print(torch.sum(bad_done), 'unreach target!')
        if torch.any(done):
            self.log(f'reset target!')
            print(torch.sum(done), 'reset target!')
        return bad_done, done, exceed_time_limit, info
