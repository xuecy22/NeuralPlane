import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Done: +50
    - Bad_done: -50
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        reward = -200 * env.bad_done + 200 * env.is_done
        return reward
