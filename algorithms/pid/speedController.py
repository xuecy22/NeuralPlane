import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import algorithms.pid.pid as pid
from algorithms.utils.utils import parse_config
device = "cuda:0"


class SpeedController:
    def __init__(self, config='speedcontroller', dt=0.01, n=1, device=device):
        self.config = parse_config(config)
        Kp = getattr(self.config, 'Kp')
        Ki = getattr(self.config, 'Ki')
        Kd = getattr(self.config, 'Kd')
        Kff = getattr(self.config, 'Kff')
        Kimax = getattr(self.config, 'Kimax')
        self.pid = pid.PID(Kp=Kp, Ki=Ki, Kd=Kd, Kff=Kff, Kimax=Kimax, dt=dt, n=n, device=device)
        self.dt = dt
        self.n = n
        self.device = torch.device(device)
        self.last_out = torch.zeros((self.n, 1), device=self.device)
    
    def get_rate_out(self, desired_rate, scaler, estate, eas2tas):
        rate_x = estate[:, 3].reshape(-1, 1)
        limit_I = torch.abs(self.last_out) >= 100
        self.rate_pid.update_all(desired_rate, rate_x, limit_I)
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

    def get_servo_out(self, angle_err, scaler, estate, eas2tas):
        if self.tau < 0.05:
            self.tau = 0.05
        desired_rate = angle_err / self.tau
        if self.rmax_pos:
            desired_rate = torch.clamp(desired_rate, -self.rmax_pos, self.rmax_pos)
        return self.get_rate_out(desired_rate, scaler, estate, eas2tas)
