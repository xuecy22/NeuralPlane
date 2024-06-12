import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class PositionReward(BaseRewardFunction):
    """
    Measure the difference between the current position and the target position
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        npos, epos, altitude = env.model.get_position()
        delta_npos = (npos - task.target_npos) * 0.3048 / 1000
        delta_epos = (epos - task.target_epos) * 0.3048 / 1000
        delta_altitude = (altitude - task.target_altitude) * 0.3048 / 1000
        reward_npos = -delta_npos ** 2
        reward_epos = -delta_epos ** 2
        reward_altitude = -delta_altitude ** 2
        reward_target = reward_npos + reward_epos + reward_altitude
        return 0.1 * reward_target
