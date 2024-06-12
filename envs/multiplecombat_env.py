import sys
import os
import gym
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from envs.env_base import BaseEnv
from termination_conditions.low_altitude import LowAltitude
from termination_conditions.overload import Overload
from termination_conditions.high_speed import HighSpeed
from termination_conditions.low_speed import LowSpeed
from termination_conditions.extreme_state import ExtremeState
from termination_conditions.crash import Crash
from termination_conditions.timeout import Timeout
# from termination_conditions.unreach_target import UnreachTarget
from termination_conditions.shutdown import Shutdown
from utils.utils import wrap_PI, get2d_AO_TA_R, get_AO_TA_R, orientation_reward, range_reward, orientation_fn, distance_fn
from algorithms.pid.controller import Controller


device = "cuda:0"


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is a fly-combat env for multiple agents to do combating task.
    """
    def __init__(self, num_envs=1, config='selfplay', random_seed=None, device=device):
        super().__init__(num_envs, config, random_seed, device)
        self.ego_agents = getattr(self.config, 'ego_agents', 2)
        self.enm_agents = getattr(self.config, 'enm_agents', 2)
        if self.num_agents != self.ego_agents + self.enm_agents:
            raise NotImplementedError(f"Number of agents must be equal to ego plus enm!")
        self.recent_s = [None, None]
        self.init_T = getattr(self.config, 'init_T', 2000)
        self.target_dist = getattr(self.config, 'target_dist', 3)
        self.max_altitude = getattr(self.config, 'max_altitude', 20000)
        self.min_altitude = getattr(self.config, 'min_altitude', 19000)
        self.max_vt = getattr(self.config, 'max_vt', 1200)
        self.min_vt = getattr(self.config, 'min_vt', 1000)
        self.max_heading = getattr(self.config, 'max_heading', 0.5)
        self.min_heading = getattr(self.config, 'min_heading', -0.5)
        self.max_npos = getattr(self.config, 'max_npos', 10000)
        self.min_npos = getattr(self.config, 'min_npos', -10000)
        self.max_epos = getattr(self.config, 'max_epos', 10000)
        self.min_epos = getattr(self.config, 'min_epos', -10000)
        # 血量
        self.blood = 100 * torch.ones(self.n, device=self.device)
        self.controller = Controller(dt=self.dt, n=self.n, device=device)
        self.termination_conditions = [
            Overload(self.config),
            LowAltitude(self.config),
            HighSpeed(self.config),
            LowSpeed(self.config),
            ExtremeState(self.config),
            Crash(self.config, device),
            Timeout(self.config),
            Shutdown(self.config, device)
            # UnreachTarget(self.config, device)
        ]

    def update_recent_s(self, s):
        self.recent_s[1] = self.recent_s[0]
        self.recent_s[0] = s
    
    def obs(self):
        # todo: 仅适用于1v1
        """
        Convert simulation states into the format of observation_space
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        """
        # ego_info
        norm_altitude = self.s[:, 2].reshape(-1, 1) * 0.3048 / 5000
        roll_sin = torch.sin(self.s[:, 3].reshape(-1, 1))
        roll_cos = torch.cos(self.s[:, 3].reshape(-1, 1))
        pitch_sin = torch.sin(self.s[:, 4].reshape(-1, 1))
        pitch_cos = torch.cos(self.s[:, 4].reshape(-1, 1))
        norm_vx = self.velocity[:, 0].reshape(-1, 1) * 0.3048 / 340
        norm_vy = self.velocity[:, 1].reshape(-1, 1) * 0.3048 / 340
        norm_vz = self.velocity[:, 2].reshape(-1, 1) * 0.3048 / 340
        norm_vt = self.s[:, 6].reshape(-1, 1) * 0.3048 / 340

        # relative enm info
        ego_agents = torch.arange(self.num_envs, device=self.device) * self.num_agents
        enm_agents = ego_agents + 1
        ego_vx = self.velocity[ego_agents, 0].reshape(-1, 1)
        enm_vx = self.velocity[enm_agents, 0].reshape(-1, 1)
        delta_vx = torch.hstack((enm_vx - ego_vx, ego_vx - enm_vx))
        delta_vx = delta_vx.reshape(-1, 1) * 0.3048 / 340

        ego_altitude = self.s[ego_agents, 2].reshape(-1, 1)
        enm_altitude = self.s[enm_agents, 2].reshape(-1, 1)
        delta_altitude = torch.hstack((enm_altitude - ego_altitude, ego_altitude - enm_altitude))
        delta_altitude = delta_altitude.reshape(-1, 1) * 0.3048 / 1000

        ego_pos = self.s[ego_agents, :3]
        enm_pos = self.s[enm_agents, :3]
        ego_vel = self.es[ego_agents, :3]
        enm_vel = self.es[enm_agents, :3]
        ego_AO, ego_TA, distance, side_flag = get2d_AO_TA_R(ego_pos, enm_pos, ego_vel, enm_vel, return_side=True)
        norm_ego_AO = torch.hstack((ego_AO.reshape(-1, 1), torch.pi - ego_TA.reshape(-1, 1)))
        norm_ego_AO = norm_ego_AO.reshape(-1, 1)
        norm_ego_TA = torch.hstack((ego_TA.reshape(-1, 1), torch.pi - ego_AO.reshape(-1, 1)))
        norm_ego_TA = norm_ego_TA.reshape(-1, 1)
        norm_R = torch.hstack((distance.reshape(-1, 1), distance.reshape(-1, 1)))
        norm_R = norm_R.reshape(-1, 1) * 0.3048 / 10000
        norm_side_flag = torch.hstack((side_flag.reshape(-1, 1), -side_flag.reshape(-1, 1)))
        norm_side_flag = norm_side_flag.reshape(-1, 1)

        obs = torch.hstack((norm_altitude, roll_sin))
        obs = torch.hstack((obs, roll_cos))
        obs = torch.hstack((obs, pitch_sin))
        obs = torch.hstack((obs, pitch_cos))
        obs = torch.hstack((obs, norm_vx))
        obs = torch.hstack((obs, norm_vy))
        obs = torch.hstack((obs, norm_vz))
        obs = torch.hstack((obs, norm_vt))
        obs = torch.hstack((obs, delta_vx))
        obs = torch.hstack((obs, delta_altitude))
        obs = torch.hstack((obs, norm_ego_AO))
        obs = torch.hstack((obs, norm_ego_TA))
        obs = torch.hstack((obs, norm_R))
        obs = torch.hstack((obs, norm_side_flag))
        return obs
    
    def reward(self):
        # s = self.recent_s[0]
        # last_s = self.recent_s[1]

        # # smooth_reward
        # roll = s[:, 3]
        # pitch = s[:, 4]
        # yaw = s[:, 5]
        # last_roll = last_s[:, 3]
        # last_pitch = last_s[:, 4]
        # last_yaw = last_s[:, 5]
        # smooth_error_scale = torch.pi
        # delta_roll = wrap_PI(roll - last_roll)
        # delta_pitch = wrap_PI(pitch - last_pitch)
        # delta_yaw = wrap_PI(yaw - last_yaw)
        # reward_roll = -(delta_roll / smooth_error_scale) ** 2
        # reward_pitch = -(delta_pitch / smooth_error_scale) ** 2
        # reward_yaw = -(delta_yaw / smooth_error_scale) ** 2
        # reward = reward_roll + reward_pitch + reward_yaw
        # # reward = 2000 * reward_roll + 1000 * reward_pitch + 1000 * reward_yaw
        # reward = 1000 * reward

        # posture reward
        ego_agents = torch.arange(self.num_envs, device=self.device) * self.num_agents
        enm_agents = ego_agents + 1
        ego_pos = self.s[ego_agents, :3]
        enm_pos = self.s[enm_agents, :3]
        ego_vel = self.es[ego_agents, :3]
        enm_vel = self.es[enm_agents, :3]
        ego_AO, ego_TA, distance = get_AO_TA_R(ego_pos, enm_pos, ego_vel, enm_vel)
        ego_orientation_reward = orientation_reward(ego_AO, ego_TA)
        enm_orientation_reward = orientation_reward(torch.pi - ego_TA, torch.pi - ego_AO)
        ego_range_reward = range_reward(self.target_dist, distance * 0.3048 / 1000)
        enm_range_reward = ego_range_reward
        ego_reward = ego_orientation_reward * ego_range_reward
        enm_reward = enm_orientation_reward * enm_range_reward
        reward = torch.hstack((ego_reward.reshape(-1, 1), enm_reward.reshape(-1, 1)))
        reward = reward.reshape(-1)
        return reward
    
    def reset(self):
        self.s = torch.zeros((self.n, self.num_states), device=self.device)  # state
        self.u = torch.zeros((self.n, self.num_controls), device=self.device)
        self.s[:, 0] = torch.rand_like(self.s[:, 0]) * (self.max_npos - self.min_npos) + self.min_npos
        self.s[:, 1] = torch.rand_like(self.s[:, 1]) * (self.max_epos - self.min_epos) + self.min_epos
        self.s[:, 2] = torch.rand_like(self.s[:, 2]) * (self.max_altitude - self.min_altitude) + self.min_altitude
        self.s[:, 5] = torch.rand_like(self.s[:, 5]) * (self.max_heading - self.min_heading) + self.min_heading
        self.s[:, 6] = torch.rand_like(self.s[:, 6]) * (self.max_vt - self.min_vt) + self.min_vt
        self.u[:, 0] = self.init_T
        self.dynamics.update(self.u)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.eas2tas = self.dynamics.EAS2TAS(self.s)
        self.velocity = self.dynamics.velocity(self.s)
        self.acceleration = self.dynamics.acceleration(self.s, self.es)
        self.blood = 100 * torch.ones(self.n, device=self.device)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.is_done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.bad_done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.exceed_time_limit = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        obs = self.obs()
        self.recent_s[1] = self.s
        self.recent_s[0] = self.s
        return obs
    
    def reset_done_envs(self):
        """Only reset envs that are already done."""
        done = self.is_done.bool()
        bad_done = self.bad_done.bool()
        exceed_time_limit = self.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        reset = torch.reshape(reset, (self.num_envs, self.num_agents, 1))
        reset = torch.any(reset.squeeze(axis=-1), axis=-1)
        reset_env = torch.nonzero(reset).squeeze(axis=-1)
        agents = torch.arange(self.n, device=self.device)
        agents = agents.reshape(self.num_envs, self.num_agents)
        reset_agents = agents[reset_env, :]
        reset_agents = reset_agents.reshape(-1)
        self.s[reset_agents, :] = torch.zeros_like(self.s[reset_agents, :])  # state
        self.u[reset_agents, :] = torch.zeros_like(self.u[reset_agents, :])
        self.s[reset_agents, 0] = torch.rand_like(self.s[reset_agents, 0]) * (self.max_npos - self.min_npos) + self.min_npos
        self.s[reset_agents, 1] = torch.rand_like(self.s[reset_agents, 1]) * (self.max_epos - self.min_epos) + self.min_epos
        self.s[reset_agents, 2] = torch.rand_like(self.s[reset_agents, 2]) * (self.max_altitude - self.min_altitude) + self.min_altitude
        self.s[reset_agents, 5] = torch.rand_like(self.s[reset_agents, 5]) * (self.max_heading - self.min_heading) + self.min_heading
        self.s[reset_agents, 6] = torch.rand_like(self.s[reset_agents, 6]) * (self.max_vt - self.min_vt) + self.min_vt
        self.u[reset_agents, 0] = self.init_T
        self.dynamics.update(self.u)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.eas2tas = self.dynamics.EAS2TAS(self.s)
        self.velocity = self.dynamics.velocity(self.s)
        self.acceleration = self.dynamics.acceleration(self.s, self.es)
        self.blood[reset_agents] = 100
        self.step_count[reset_agents] = 0
        self.is_done[:] = 0
        self.bad_done[:] = 0
        self.exceed_time_limit[:] = 0
        self.recent_s[0] = self.s
    
    def step(self, action):
        self.reset_done_envs()

        action = torch.clamp(action, -1, 1)
        self.controller.roll_dem = 0.9 * self.controller.roll_dem + 0.1 * action[:, 1].reshape(-1, 1) * 4 * torch.pi / 9
        self.controller.pitch_dem = 0.9 * self.controller.pitch_dem + 0.1 * action[:, 2].reshape(-1, 1) * torch.pi / 12
        # self.controller.roll_dem = wrap_PI(self.s[:, 3].reshape(-1, 1) + action[:, 1].reshape(-1, 1) * torch.pi / 18)
        # self.controller.pitch_dem = wrap_PI(self.s[:, 4].reshape(-1, 1) + action[:, 2].reshape(-1, 1) * torch.pi / 18)
        self.controller.yaw_dem = wrap_PI(self.s[:, 5].reshape(-1, 1) + action[:, 3].reshape(-1, 1) * torch.pi / 60)
        self.controller.stabilize(self.s, self.es, self.eas2tas)
        T = 0.9 * self.u[:, 0].reshape(-1, 1) + 0.1 * action[:, 0].reshape(-1, 1) * 0.225 * 76300 / 0.3048
        el = -self.controller.el
        ail = -self.controller.ail
        rud = -self.controller.rud
        lef = torch.zeros((self.n, 1), device=self.device)
        action = torch.hstack((T, el))
        action = torch.hstack((action, ail))
        action = torch.hstack((action, rud))
        action = torch.hstack((action, lef))
        obs, reward, done, bad_done, exceed_time_limit, info = super().step(action)

        ego_agents = torch.arange(self.num_envs, device=self.device) * self.num_agents
        enm_agents = ego_agents + 1
        ego_pos = self.s[ego_agents, :3]
        enm_pos = self.s[enm_agents, :3]
        ego_vel = self.es[ego_agents, :3]
        enm_vel = self.es[enm_agents, :3]
        AO, TA, R = get_AO_TA_R(ego_pos, enm_pos, ego_vel, enm_vel)
        self.blood[enm_agents] -= orientation_fn(AO) * distance_fn(R * 0.3048 / 1000)
        self.blood[ego_agents] -= orientation_fn(torch.pi - TA) * distance_fn(R * 0.3048 / 1000)

        self.update_recent_s(self.s)
        return obs, reward, done, bad_done, exceed_time_limit, info
