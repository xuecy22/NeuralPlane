import torch
import pid
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import algorithms.pid.pid as pid
from algorithms.utils.utils import parse_config
device = "cuda:0"


class PitchController:
    def __init__(self, config='pitchcontroller', dt=0.01, n=1, device=device):
        self.config = parse_config(config)
        Kp = getattr(self.config, 'Kp')
        Ki = getattr(self.config, 'Ki')
        Kd = getattr(self.config, 'Kd')
        Kff = getattr(self.config, 'Kff')
        Kimax = getattr(self.config, 'Kimax')
        self.rate_pid = pid.PID(Kp=Kp, Ki=Ki, Kd=Kd, Kff=Kff, Kimax=Kimax, dt=dt, n=n, device=device)
        self.dt = dt
        self.n = n
        self.device = torch.device(device)
        self.tau = getattr(self.config, 'tau')
        self.rmax_pos = getattr(self.config, 'rmax_pos')
        self.rmax_neg = getattr(self.config, 'rmax_neg')
        self.roll_ff = getattr(self.config, 'roll_ff')
        self.gravity = getattr(self.config, 'gravity')
        self.last_out = torch.zeros((self.n, 1), device=self.device)
    
    def get_rate_out(self, desired_rate, scaler, env):
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        roll_rate, pitch_rate, yaw_rate = env.model.get_euler_angular_velocity()
        pitch_rate = pitch_rate.reshape(-1, 1)
        limit_I = torch.abs(self.last_out) > 45
        self.rate_pid.update_all(desired_rate * scaler * scaler, pitch_rate * scaler * scaler, limit_I)
        ff_out = self.rate_pid.get_ff() / (scaler * eas2tas + 1e-8)
        # ff_out = self.rate_pid.get_ff() / scaler
        p_out = self.rate_pid.get_p()
        i_out = self.rate_pid.get_i()
        d_out = self.rate_pid.get_d()
        out = ff_out + p_out + i_out + d_out
        out = 180 * out / torch.pi
        self.last_out = out
        out = torch.clamp(out, -45, 45)
        return out
    
    def get_coordination_rate_offset(self, env):
        TAS = env.model.get_TAS()
        roll, pitch, yaw = env.model.get_posture()
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        vt = TAS.reshape(-1, 1)
        roll = roll.reshape(-1, 1)
        pitch = pitch.reshape(-1, 1)
        mask1 = torch.abs(roll) < (torch.pi / 2)
        mask2 = roll >= (torch.pi / 2)
        mask3 = roll <= (-torch.pi / 2)
        roll1 = torch.clamp(roll, -4 * torch.pi / 9, 4 * torch.pi / 9)
        roll2 = torch.clamp(roll, 5 * torch.pi / 9, torch.pi)
        roll3 = torch.clamp(roll, -torch.pi, -5 * torch.pi / 9)
        inverted = ~mask1
        roll = mask1 * roll1 + mask2 * roll2 + mask3 * roll3
        mask1 = torch.abs(pitch) <= (7 * torch.pi / 18)
        # w = self.gravity * torch.tan(roll) / vt * eas2tas
        # Q = w * torch.sin(pitch)
        # R = w * torch.cos(roll) * torch.cos(pitch)
        # rate_offset = mask1 * (Q * torch.cos(roll) - R * torch.sin(roll)) * self.roll_ff
        rate_offset = mask1 * torch.cos(pitch) * torch.abs(self.gravity / vt * torch.tan(roll) * torch.sin(roll) * eas2tas) * self.roll_ff
        rate_offset = rate_offset * ~inverted - rate_offset * inverted
        return inverted, rate_offset

    def get_servo_out(self, angle_err, scaler, env):
        if self.tau < 0.05:
            self.tau = 0.05
        desired_rate = angle_err / self.tau
        inverted, rate_offset = self.get_coordination_rate_offset(env)
        desired_rate1 = desired_rate + rate_offset
        if self.rmax_pos:
            desired_rate1 = torch.clamp(desired_rate1, max=self.rmax_pos)
        if self.rmax_neg:
            desired_rate1 = torch.clamp(desired_rate1, min=-self.rmax_neg)
        desired_rate = ~inverted * desired_rate1 + inverted * (rate_offset - desired_rate)

        roll, pitch, yaw = env.model.get_posture()
        roll = roll.reshape(-1, 1)
        roll_wrapped = torch.abs(roll)
        pitch = pitch.reshape(-1, 1)
        pitch_wrapped = torch.abs(pitch)
        mask = roll_wrapped > (torch.pi / 2)
        roll_wrapped = mask * (torch.pi - roll_wrapped) + (~mask) * roll_wrapped
        mask = (roll_wrapped > (5 * torch.pi / 18)) & (pitch_wrapped < (7 * torch.pi / 18))
        roll_prop = (roll_wrapped - 5 * torch.pi / 18) / (4 * torch.pi / 18)
        roll_prop = roll_prop * mask
        desired_rate = desired_rate * (1 - roll_prop)
        return self.get_rate_out(desired_rate, scaler, env)
