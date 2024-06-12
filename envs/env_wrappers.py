"""
A simplified version from OpenAI Baselines code to work with gym.env parallelization.
"""
import torch
from abc import ABC, abstractmethod
from utils.utils import _t2n


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()


class GPUVecEnv(VecEnv):

    def __init__(self, env_fns):
        assert len(env_fns) == 1, "Number of create env funcitions must be 1!"
        self.gpu_vec_env = env_fns[0]()
        assert hasattr(self.gpu_vec_env, "num_envs"), "Parameter of env must contain num_envs!"
        super().__init__(self.gpu_vec_env.num_envs, self.gpu_vec_env.observation_space, self.gpu_vec_env.action_space)
        self.agents = self.gpu_vec_env.num_agents

    def step(self, actions):
        actions = torch.tensor(actions, device=self.gpu_vec_env.device, dtype=torch.float32)
        actions = torch.reshape(actions, (self.num_envs * self.agents, self.gpu_vec_env.num_actions))
        obs, rews, dones, bad_dones, exceed_time_limits, infos = self.gpu_vec_env.step(actions)
        obs = torch.reshape(obs, (self.num_envs, self.agents, self.gpu_vec_env.num_observation))
        rews = torch.reshape(rews, (self.num_envs, self.agents, 1))
        dones = torch.reshape(dones, (self.num_envs, self.agents, 1))
        bad_dones = torch.reshape(bad_dones, (self.num_envs, self.agents, 1))
        exceed_time_limits = torch.reshape(exceed_time_limits, (self.num_envs, self.agents, 1))
        obs, rews, dones, bad_dones, exceed_time_limits = _t2n(obs), _t2n(rews), _t2n(dones), _t2n(bad_dones), _t2n(exceed_time_limits)
        return obs, rews, dones, bad_dones, exceed_time_limits, infos
    
    def reset(self):
        obs = self.gpu_vec_env.reset()
        obs = torch.reshape(obs, (self.num_envs, self.agents, self.gpu_vec_env.num_observation))
        obs = _t2n(obs)
        return obs

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True
