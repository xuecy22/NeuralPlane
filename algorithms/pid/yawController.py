import torch
import pid
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import algorithms.pid.pid as pid
from algorithms.utils.utils import parse_config
device = "cuda:0"


# fix: dt 修改为控制器实际调用周期
# fix: vt 和 acceleration单位
class YawController:
    def __init__(self, config='yawcontroller', dt=0.01, n=1, device=device):
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
        self.KA = getattr(self.config, 'KA')
        self.KI = getattr(self.config, 'KI')
        self.KD = getattr(self.config, 'KD')
        self.KFF = getattr(self.config, 'KFF')
        self.imax = getattr(self.config, 'imax')
        self.rate_enable = getattr(self.config, 'rate_enable')
        self.tau = getattr(self.config, 'tau')
        self.rmax_pos = getattr(self.config, 'rmax_pos')
        self.gravity = getattr(self.config, 'gravity')
        self.last_out = torch.zeros((self.n, 1), device=self.device)
        self.last_rate_hp_out = torch.zeros((self.n, 1), device=self.device)
        self.last_rate_hp_in = torch.zeros((self.n, 1), device=self.device)
        self.integrator = torch.zeros((self.n, 1), device=self.device)

    def get_servo_out(self, scaler, state, acceleration, eas2tas):
        roll = state[:, 3].reshape(-1, 1)
        vt = state[:, 6].reshape(-1, 1)
        rate_z = state[:, 11].reshape(-1, 1)
        mask = torch.abs(roll) < (torch.pi / 2)
        roll = mask * torch.clamp(roll, -4 * torch.pi / 9, 4 * torch.pi / 9) + (~mask) * roll
        rate_offset = self.gravity * torch.sin(roll) * self.KFF * eas2tas / vt
        ay = acceleration[:, 1]
        rate_hp_in = (rate_z - rate_offset) * 180 / torch.pi
        rate_hp_out = 0.996008 * self.last_rate_hp_out + rate_hp_in - self.last_rate_hp_in
        self.last_rate_hp_out = rate_hp_out
        self.last_rate_hp_in = rate_hp_in
        integ_in = -self.KI * (self.KA * ay + rate_hp_out)
        if self.KD > 0:
            mask1 = self.last_out < -45
            mask2 = self.last_out > 45
            mask3 = ~(mask1 | mask2)
            self.integrator += torch.max(integ_in * self.dt, torch.zeros((self.n, 1), device=self.device)) * mask1
            self.integrator += torch.min(integ_in * self.dt, torch.zeros((self.n, 1), device=self.device)) * mask2
            self.integrator = self.integrator + integ_in * self.dt * mask3
        else:
            self.integrator = torch.zeros((self.n, 1), device=self.device)
        if self.KD < 0.0001:
            return torch.zeros((self.n, 1), device=self.device)
        intLimScaled = self.imax * 0.01 / (self.KD * scaler * scaler)
        self.integrator = torch.clamp(self.integrator, -intLimScaled, intLimScaled)
        self.last_out =  self.KD * self.integrator * scaler * scaler + self.KD * (-rate_hp_out) * scaler * scaler
        return torch.clamp(self.last_out, -45, 45)

    
    def get_rate_out(self, desired_rate, scaler, env):
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        roll_rate, pitch_rate, yaw_rate = env.model.get_euler_angular_velocity()
        yaw_rate = yaw_rate.reshape(-1, 1)
        limit_I = torch.abs(self.last_out) >= 45
        self.rate_pid.update_all(desired_rate * scaler * scaler, yaw_rate * scaler * scaler, limit_I)
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
    
    # def get_servo_out(self, angle_err, scaler, estate, eas2tas):
    #     if self.tau < 0.05:
    #         self.tau = 0.05
    #     desired_rate = angle_err / self.tau
    #     if self.rmax_pos:
    #         desired_rate = torch.clamp(desired_rate, -self.rmax_pos, self.rmax_pos)
    #     return self.get_rate_out(desired_rate, scaler, estate, eas2tas)
