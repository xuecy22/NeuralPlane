from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_reward(self, task, env):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        raise NotImplementedError
