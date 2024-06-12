import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from termination_condition_base import BaseTerminationCondition
from utils.utils import wrap_PI


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config, device):
        super().__init__(config)
        self.device = torch.device(device)
        self.max_check_interval = getattr(config, 'max_check_interval', 1500)
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
        roll, picth, heading = env.model.get_posture()
        npos, epos, altitude = env.model.get_position()
        vt = env.model.get_vt()
        check_time = env.step_count
        # 判断时间
        mask1 = check_time >= self.max_check_interval
        mask2 = check_time >= self.min_check_interval
        # 判断是否到达target_heading
        mask3 = torch.abs(wrap_PI(heading - task.target_heading)) >= torch.pi / 36
        # 判断是否到达target_altitude
        mask4 = torch.abs(altitude - task.target_altitude) >= 100
        # 判断是否到达target_vt
        mask5  = torch.abs(vt - task.target_vt) >= 20
        # 判断roll是否满足要求
        # mask6 = torch.abs(wrap_PI(roll)) >= torch.pi / 36
        # 当超过时间且未达到目标时，判断为True
        # bad_done = mask1 & ((mask3 | mask4) | (mask5 | mask6))
        bad_done = mask1 & ((mask3 | mask4) | mask5)
        # 当达到目标且时间符合要求时，重新设置目标
        # done =  ((~((mask3 | mask4) | (mask5 | mask6))) & (~mask1)) & mask2
        done =  ((~((mask3 | mask4) | mask5)) & (~mask1)) & mask2
        exceed_time_limit = torch.zeros_like(done)
        if torch.any(bad_done):
            self.log(f'unreach heading!')
            print(torch.sum(bad_done), 'unreach heading!')
        if torch.any(done):
            self.log(f'reset target!')
            print(torch.sum(done), 'reset target!')
        return bad_done, done, exceed_time_limit, info
