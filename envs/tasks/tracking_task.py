import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.position_reward import PositionReward
from reward_functions.event_driven_reward import EventDrivenReward
from termination_conditions.low_altitude import LowAltitude
from termination_conditions.overload import Overload
from termination_conditions.high_speed import HighSpeed
from termination_conditions.low_speed import LowSpeed
from termination_conditions.extreme_state import ExtremeState
from termination_conditions.timeout import Timeout
from termination_conditions.unreach_target import UnreachTarget
from utils.utils import wrap_PI


class TrackingTask(BaseTask):
    '''
    Control target angle with control surface
    '''
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)

        self.target_npos = torch.zeros(self.n, device=self.device)
        self.target_epos = torch.zeros(self.n, device=self.device)
        self.target_altitude = torch.zeros(self.n, device=self.device)
        self.max_distance = getattr(self.config, 'max_distance', 2000)
        self.min_distance = getattr(self.config, 'min_distance', 2000)
        self.noise_scale = getattr(self.config, 'noise_scale', 0.01)

        self.reward_functions = [
            PositionReward(self.config),
            EventDrivenReward(self.config),
        ]
        
        self.termination_conditions = [
            Overload(self.config),
            LowAltitude(self.config),
            HighSpeed(self.config),
            LowSpeed(self.config),
            ExtremeState(self.config),
            # Timeout(self.config),
            UnreachTarget(self.config, device)
        ]

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = torch.sum(reset)

        npos, epos, altitude = env.model.get_position()

        distance = torch.rand(size, device=self.device) * (self.max_distance - self.min_distance) + self.min_distance
        theta1 = torch.rand(size, device=self.device) * torch.pi / 3 - torch.pi / 6
        # theta1 = torch.ones(size, device=self.device) * torch.pi / 2
        theta2 = torch.rand(size, device=self.device) * torch.pi / 3 - torch.pi / 6
        # theta2 = torch.zeros(size, device=self.device)
        delta_npos = distance * torch.cos(theta1) * torch.cos(theta2)
        delta_epos = distance * torch.cos(theta1) * torch.sin(theta2)
        delta_altitude = distance * torch.sin(theta1)
        # delta_npos = 0
        # delta_epos = 0
        # delta_altitude = distance

        self.target_npos[reset] = npos[reset] + delta_npos
        self.target_epos[reset] = epos[reset] + delta_epos
        self.target_altitude[reset] = altitude[reset] + delta_altitude
    
    def get_obs(self, env):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 22):
            0. ego_delta_npos      (unit: km)
            1. ego_delta_epos       (unit km)
            2. ego_delta_altitude            (unit: km)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
            16. ego_T                  (unit: %)
            17. ego_el                 (unit: %)
            18. ego_ail                (unit: %)
            19. ego_rud                (unit: %)
            20. ego_lef                (unit: %)
            21. EAS2TAS
        """
        npos, epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        EAS = env.model.get_EAS()
        alpha = env.model.get_AOA()
        beta = env.model.get_AOS()
        P, Q, R = env.model.get_angular_velocity()
        T = env.model.get_thrust()
        el, ail, rud, lef = env.model.get_control_surface()
        eas2tas = env.model.get_EAS2TAS()

        norm_delta_npos = (npos - self.target_npos).reshape(-1, 1) * 0.3048 / 1000
        norm_delta_epos = (epos - self.target_epos).reshape(-1, 1) * 0.3048 / 1000
        norm_delta_altitude = (altitude - self.target_altitude).reshape(-1, 1) * 0.3048 / 1000
        norm_altitude = altitude.reshape(-1, 1) * 0.3048 / 5000
        roll_sin = torch.sin(roll.reshape(-1, 1))
        roll_cos = torch.cos(roll.reshape(-1, 1))
        pitch_sin = torch.sin(pitch.reshape(-1, 1))
        pitch_cos = torch.cos(pitch.reshape(-1, 1))
        # norm_vt = vt.reshape(-1, 1) * 0.3048 / 340
        norm_EAS = EAS.reshape(-1, 1) * 0.3048 / 340
        alpha_sin = torch.sin(alpha.reshape(-1, 1))
        alpha_cos = torch.cos(alpha.reshape(-1, 1))
        beta_sin = torch.sin(beta.reshape(-1, 1))
        beta_cos = torch.cos(beta.reshape(-1, 1))
        norm_P = P.reshape(-1, 1)
        norm_Q = Q.reshape(-1, 1)
        norm_R = R.reshape(-1, 1)
        norm_T = T.reshape(-1, 1) / 0.225 / 76300 * 0.3048
        norm_el = el.reshape(-1, 1) / 45
        norm_ail = ail.reshape(-1, 1) / 45
        norm_rud = rud.reshape(-1, 1) / 45
        norm_lef = lef.reshape(-1, 1) / 45
        obs = torch.hstack((norm_delta_npos, norm_delta_epos))
        obs = torch.hstack((obs, norm_delta_altitude))
        obs = torch.hstack((obs, norm_altitude))
        obs = torch.hstack((obs, roll_sin))
        obs = torch.hstack((obs, roll_cos))
        obs = torch.hstack((obs, pitch_sin))
        obs = torch.hstack((obs, pitch_cos))
        obs = torch.hstack((obs, norm_EAS))
        obs = torch.hstack((obs, alpha_sin))
        obs = torch.hstack((obs, alpha_cos))
        obs = torch.hstack((obs, beta_sin))
        obs = torch.hstack((obs, beta_cos))
        obs = torch.hstack((obs, norm_P))
        obs = torch.hstack((obs, norm_Q))
        obs = torch.hstack((obs, norm_R))
        obs = torch.hstack((obs, norm_T))
        obs = torch.hstack((obs, norm_el))
        obs = torch.hstack((obs, norm_ail))
        obs = torch.hstack((obs, norm_rud))
        obs = torch.hstack((obs, norm_lef))
        obs = torch.hstack((obs, eas2tas.reshape(-1, 1)))
        return obs + torch.randn_like(obs) * self.noise_scale
