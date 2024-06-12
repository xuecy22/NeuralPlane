import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class PostureReward(BaseRewardFunction):
    """
    Measure the difference between the current posture and the target posture
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
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        delta_pitch = wrap_PI(pitch - task.target_pitch) / torch.pi
        delta_heading = wrap_PI(heading - task.target_heading) / torch.pi
        delta_vt = (vt - task.target_vt) * 0.3048 / 340
        reward_pitch = -delta_pitch ** 2
        reward_heading = -delta_heading ** 2
        reward_vt = -delta_vt ** 2
        reward_target = reward_pitch + reward_heading + reward_vt
        return reward_target
