import torch
device = "cuda:0"


class PID:
    def __init__(self, Kp=0, Ki=0, Kd=0, Kff=0, Kimax=0, dt=0.01, n=1, device=device):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kff = Kff
        self.Kimax = Kimax
        self.dt = dt
        self.n = n
        self.reset = True
        self.device = torch.device(device)
    
    def update_all(self, target, measurement, limit):
        if torch.isnan(target).any() or torch.isinf(target).any():
            return torch.zeros((self.n, 1), device=self.device)
        if torch.isnan(measurement).any() or torch.isinf(measurement).any():
            return torch.zeros((self.n, 1), device=self.device)
        if self.reset:
            self.reset = False
            self.target = target
            self.error = target - measurement
            self.derivative = torch.zeros((self.n, 1), device=self.device)
            self.integrator = torch.zeros((self.n, 1), device=self.device)
        else:
            last_error = self.error
            self.target = target
            self.error = target - measurement
            self.derivative = (self.error - last_error) / self.dt
        self.update_i(limit)
        return self.error * self.Kp + self.derivative * self.Kd + self.integrator

    def update_i(self, limit):
        if self.Ki != 0 and self.dt > 0:
            self.integrator = self.integrator + self.error * self.Ki * self.dt * (~limit | (self.error * self.dt < 0))
            self.integrator = torch.clamp(self.integrator, -self.Kimax, self.Kimax)
        else:
            self.integrator = torch.zeros((self.n, 1), device=self.device)
    
    def get_p(self):
        return self.error * self.Kp
    
    def get_i(self):
        return self.integrator
    
    def get_d(self):
        return self.derivative * self.Kd
    
    def get_ff(self):
        return self.target * self.Kff
    
    def reset_I(self):
        self.integrator = torch.zeros((self.n, 1), device=self.device)
