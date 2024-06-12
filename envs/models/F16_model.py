import os
import sys
import torch
from torchdiffeq import odeint_adjoint as odeint
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model_base import BaseModel
from F16.F16_dynamics import F16Dynamics


class F16Model(BaseModel):
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.num_states = getattr(self.config, 'num_states', 12)
        self.num_controls = getattr(self.config, 'num_controls', 5)
        self.dt = getattr(self.config, 'dt', 0.02)
        self.solver = getattr(self.config, 'solver', 'euler')
        self.airspeed = getattr(self.config, 'airspeed', 0)

        self.s = torch.zeros((self.n, self.num_states), device=self.device)  # state
        self.recent_s = torch.zeros((self.n, self.num_states), device=self.device)  # recent state
        self.u = torch.zeros((self.n, self.num_controls), device=self.device) # control
        self.recent_u = torch.zeros((self.n, self.num_controls), device=self.device)  # recent control

        # init parameters
        self.max_altitude = getattr(self.config, 'max_altitude', 20000)
        self.min_altitude = getattr(self.config, 'min_altitude', 19000)
        self.max_vt = getattr(self.config, 'max_vt', 1200)
        self.min_vt = getattr(self.config, 'min_vt', 1000)
        self.init_state = self.config.init_state

        self.dynamics = F16Dynamics(device)

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = torch.sum(reset)
        self.s[reset, :] = torch.zeros((size, self.num_states), device=self.device)  # state
        self.u[reset, :] = torch.zeros((size, self.num_controls), device=self.device)
        self.s[reset, 2] = torch.rand_like(self.s[reset, 2]) * (self.max_altitude - self.min_altitude) + self.min_altitude
        self.s[reset, 6] = torch.rand_like(self.s[reset, 6]) * (self.max_vt - self.min_vt) + self.min_vt
        self.u[reset, 0] = self.init_state['init_T']
        self.recent_s[reset] = self.s[reset]
        self.recent_u[reset] = self.u[reset]

    def get_extended_state(self):
        x = torch.hstack((self.s, self.u))
        return self.dynamics.nlplant(x)
    
    def update(self, action):
        action = torch.clamp(action, -1, 1)
        T = 0.9 * self.u[:, 0].reshape(-1, 1) + 0.1 * action[:, 0].reshape(-1, 1) * 0.225 * 76300 / 0.3048
        el = 0.9 * self.u[:, 1].reshape(-1, 1) + 0.1 * action[:, 1].reshape(-1, 1) * 45
        ail = 0.9 * self.u[:, 2].reshape(-1, 1) + 0.1 * action[:, 2].reshape(-1, 1) * 45
        rud = 0.9 * self.u[:, 3].reshape(-1, 1) + 0.1 * action[:, 3].reshape(-1, 1) * 45
        lef = torch.zeros((self.n, 1), device=self.device)
        self.recent_u = self.u
        self.u = torch.hstack((T, el))
        self.u = torch.hstack((self.u, ail))
        self.u = torch.hstack((self.u, rud))
        self.u = torch.hstack((self.u, lef))
        self.recent_s = self.s
        self.s = odeint(self.dynamics,
                        torch.hstack((self.s, self.u)),
                        torch.tensor([0., self.dt], device=self.device),
                        method=self.solver)[1, :, :self.num_states]
    
    def get_state(self):
        return self.s
    
    def get_control(self):
        return self.u
    
    def get_position(self):
        return self.s[:, 0], self.s[:, 1], self.s[:, 2]
    
    def get_ground_speed(self):
        es = self.get_extended_state()
        return es[:, 0], es[:, 1]
    
    def get_climb_rate(self):
        es = self.get_extended_state()
        return es[:, 2]
    
    def get_posture(self):
        return self.s[:, 3], self.s[:, 4], self.s[:, 5]
    
    def get_euler_angular_velocity(self):
        es = self.get_extended_state()
        return es[:, 3], es[:, 4], es[:, 5]
    
    def get_vt(self):
        return self.s[:, 6]
    
    def get_TAS(self):
        return self.s[:, 6] + self.airspeed * torch.ones_like(self.s[:, 6])
    
    def get_EAS(self):
        TAS = self.get_TAS()
        EAS2TAS = self.get_EAS2TAS()
        EAS = TAS / EAS2TAS
        return EAS
    
    def get_AOA(self):
        return self.s[:, 7]
    
    def get_AOS(self):
        return self.s[:, 8]
    
    def get_angular_velocity(self):
        return self.s[:, 9], self.s[:, 10], self.s[:, 11]
    
    def get_thrust(self):
        return self.u[:, 0]
    
    def get_control_surface(self):
        return self.u[:, 1], self.u[:, 2], self.u[:, 3], self.u[:, 4]

    
    def get_velocity(self):
        # 根据飞行状态计算三轴速度
        sina = torch.sin(self.s[:, 7])
        cosa = torch.cos(self.s[:, 7])
        sinb = torch.sin(self.s[:, 8])
        cosb = torch.cos(self.s[:, 8])
        vel_u = self.s[:, 6] * cosb * cosa # x轴速度
        vel_v = self.s[:, 6] * sinb # y轴速度
        vel_w = self.s[:, 6] * cosb * sina # z轴速度
        return vel_u, vel_v, vel_w
    
    def get_acceleration(self):
        # 根据飞行状态计算三轴加速度
        xdot = self.get_extended_state()
        sina = torch.sin(self.s[:, 7])
        cosa = torch.cos(self.s[:, 7])
        sinb = torch.sin(self.s[:, 8])
        cosb = torch.cos(self.s[:, 8])
        vel_u = self.s[:, 6] * cosb * cosa # x轴速度
        vel_v = self.s[:, 6] * sinb # y轴速度
        vel_w = self.s[:, 6] * cosb * sina # z轴速度
        u_dot = cosb * cosa * xdot[:, 6] - self.s[:, 6] * sinb * cosa * xdot[:, 8] - self.s[:, 6] * cosb * sina * xdot[:, 7]
        v_dot = sinb * xdot[:, 6] + self.s[:, 6] * cosb * xdot[:, 8]
        w_dot = cosb * sina * xdot[:, 6] - self.s[:, 6] * sinb * sina * xdot[:, 8] + self.s[:, 6] * cosb * cosa * xdot[:, 7]
        ax = u_dot + self.s[:, 10] * vel_w - self.s[:, 11] * vel_v
        ay = v_dot + self.s[:, 11] * vel_u - self.s[:, 9] * vel_w
        az = w_dot + self.s[:, 9] * vel_v - self.s[:, 10] * vel_u
        return ax, ay, az
    
    def get_G(self):
        # 根据飞行状态计算过载
        nx_cg, ny_cg, nz_cg = self.get_accels()
        G = torch.sqrt(nx_cg ** 2 + ny_cg ** 2 + nz_cg ** 2)
        return G
    
    def get_EAS2TAS(self):
        # 根据高度计算EAS2TAS
        alt = self.s[:, 2]
        tfac = 1 - .703e-5 * (alt)
        eas2tas = 1 / torch.pow(tfac, 4.14)
        eas2tas = torch.sqrt(eas2tas)
        return eas2tas
    
    def get_accels(self):
        # 根据飞行状态计算三轴过载
        grav = 32.174
        xdot = self.get_extended_state()
        sina = torch.sin(self.s[:, 7])
        cosa = torch.cos(self.s[:, 7])
        sinb = torch.sin(self.s[:, 8])
        cosb = torch.cos(self.s[:, 8])
        vel_u = self.s[:, 6] * cosb * cosa
        vel_v = self.s[:, 6] * sinb
        vel_w = self.s[:, 6] * cosb * sina
        u_dot = cosb * cosa * xdot[:, 6] - self.s[:, 6] * sinb * cosa * xdot[:, 8] - self.s[:, 6] * cosb * sina * xdot[:, 7]
        v_dot = sinb * xdot[:, 6] + self.s[:, 6] * cosb * xdot[:, 8]
        w_dot = cosb * sina * xdot[:, 6] - self.s[:, 6] * sinb * sina * xdot[:, 8] + self.s[:, 6] * cosb * cosa * xdot[:, 7]
        nx_cg = 1.0 / grav * (u_dot + self.s[:, 10] * vel_w - self.s[:, 11] * vel_v) + torch.sin(self.s[:, 4])
        ny_cg = 1.0 / grav * (v_dot + self.s[:, 11] * vel_u - self.s[:, 9] * vel_w) - torch.cos(self.s[:, 4]) * torch.sin(self.s[:, 3])
        nz_cg = -1.0 / grav * (w_dot + self.s[:, 9] * vel_v - self.s[:, 10] * vel_u) + torch.cos(self.s[:, 4]) * torch.cos(self.s[:, 3])
        return nx_cg, ny_cg, nz_cg

    def get_atmos(self):
        # 根据高度和速度计算动压、马赫数、静压
        alt = self.s[:, 2]
        vt = self.s[:, 6]
        rho0 = 2.377e-3
        tfac = 1 - .703e-5 * (alt)
        temp = 519.0 * tfac
        temp = (alt >= 35000.0) * 390 + (alt < 35000.0) * temp
        rho = rho0 * pow(tfac, 4.14)
        mach = (vt) / torch.sqrt(1.4 * 1716.3 * temp)
        qbar = .5 * rho * pow(vt, 2)
        ps = 1715.0 * rho * temp

        ps = (ps == 0) * 1715 + (ps != 0) * ps

        return (mach, qbar, ps)
