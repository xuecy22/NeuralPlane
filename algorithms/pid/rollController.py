import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import algorithms.pid.pid as pid
from algorithms.utils.utils import parse_config
device = "cuda:0"


class RollController:
    def __init__(self, config='rollcontroller', dt=0.01, n=1, device=device):
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
        self.last_out = torch.zeros((self.n, 1), device=self.device)
    
    def get_rate_out(self, desired_rate, scaler, env):
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        roll_rate, pitch_rate, yaw_rate = env.model.get_euler_angular_velocity()
        roll_rate = roll_rate.reshape(-1, 1)
        limit_I = torch.abs(self.last_out) >= 45
        self.rate_pid.update_all(desired_rate * scaler * scaler, roll_rate * scaler * scaler, limit_I)
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

    def get_servo_out(self, angle_err, scaler, env):
        if self.tau < 0.05:
            self.tau = 0.05
        desired_rate = angle_err / self.tau
        if self.rmax_pos:
            desired_rate = torch.clamp(desired_rate, -self.rmax_pos, self.rmax_pos)
        return self.get_rate_out(desired_rate, scaler, env)
