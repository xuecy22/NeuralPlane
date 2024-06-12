import torch
import random
import numpy as np
import gym
from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    def __init__(self, config, n, device, random_seed):
        self.config = config
        self.n = n
        self.device = device
        self.reward_functions = []
        self.termination_conditions = []
        self.num_observation = getattr(self.config, 'num_observation', 12)
        self.num_actions = getattr(self.config, 'num_actions', 5)

        self.load_observation_space()
        self.load_action_space()
        
        if random_seed is not None:
            self.seed(random_seed)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.num_observation, ))

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = gym.spaces.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(self.num_actions, ))
    
    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    @abstractmethod
    def reset(self, env):
        """Task-specific reset

        Args:
            env: environment instance
        """
        raise NotImplementedError
    
    def get_reward(self, env):
        """
        Aggregate reward functions

        Args:
            env: environment instance

        Returns:
            reward(float): total reward of the current timestep
        """
        reward = torch.zeros(self.n, device=self.device)
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env)
        return reward

    def get_termination(self, env, info={}):
        """
        Aggregate termination conditions

        Args:
            env: environment instance

        Returns:
            (tuple):
                is_dones(bool): whether the episode has terminated properly
                bad_dones(bool): whether the episode has terminated improperly
                exceed_time_limits(bool): whether the episode has exceeded time limit
        """
        dones = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        bad_dones = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        exceed_time_limits = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for condition in self.termination_conditions:
            bad_done, done, exceed_time_limit, info = condition.get_termination(self, env, info)
            dones = dones + done
            bad_dones = bad_dones + bad_done
            exceed_time_limits = exceed_time_limits + exceed_time_limit
        return dones, bad_dones, exceed_time_limits, info

    @abstractmethod
    def get_obs(self, env):
        """
        Extract useful informations from environment.
        """
        raise NotImplementedError
