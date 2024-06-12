import os
import sys
import torch
from torchdiffeq import odeint_adjoint as odeint
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model_base import BaseModel
from UAV.UAV_dynamics import UAVDynamics


class UAVModel(BaseModel):
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

        # init parameters
        self.max_altitude = getattr(self.config, 'max_altitude', 20000)
        self.min_altitude = getattr(self.config, 'min_altitude', 19000)
        self.max_vt = getattr(self.config, 'max_vt', 1200)
        self.min_vt = getattr(self.config, 'min_vt', 1000)
        self.init_state = self.config.init_state

        self.dynamics = UAVDynamics()

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = torch.sum(reset)
        self.s[reset, :] = torch.zeros((size, self.num_states), device=self.device)  # state
        self.u[reset, :] = torch.zeros((size, self.num_controls), device=self.device)
        self.s[reset, 2] = (torch.rand_like(self.s[reset, 2]) * (self.max_altitude - self.min_altitude) + self.min_altitude) * 0.3048
        self.s[reset, 6] = (torch.rand_like(self.s[reset, 6]) * (self.max_vt - self.min_vt) + self.min_vt) * 0.3048
        self.u[reset, 0] = self.init_state['init_T']
        self.recent_s[reset] = self.s[reset]

    def get_extended_state(self):
        x = torch.hstack((self.s, self.u))
        return self.dynamics.nlplant(x)
    
    def update(self, action):
        action = torch.clamp(action, -1, 1)
        Fx = 0.9 * self.u[:, 0].reshape(-1, 1) + 0.1 * action[:, 0].reshape(-1, 1) * 27000
        Fy = 0.9 * self.u[:, 1].reshape(-1, 1) + 0.1 * action[:, 1].reshape(-1, 1) * 27000
        Fz = 0.9 * self.u[:, 2].reshape(-1, 1) + 0.1 * action[:, 2].reshape(-1, 1) * 27000
        self.u = torch.hstack((Fx, Fy))
        self.u = torch.hstack((self.u, Fz))
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
        return self.s[:, 0] / 0.3048, self.s[:, 1] / 0.3048, self.s[:, 2] / 0.3048
    
    def get_ground_speed(self):
        es = self.get_extended_state()
        return es[:, 0] / 0.3048, es[:, 1] / 0.3048
    
    def get_climb_rate(self):
        es = self.get_extended_state()
        return es[:, 2] / 0.3048
    
    def get_posture(self):
        return self.s[:, 3], self.s[:, 4], self.s[:, 5]
    
    def get_euler_angular_velocity(self):
        es = self.get_extended_state()
        return es[:, 3], es[:, 4], es[:, 5]
    
    def get_vt(self):
        U = self.s[:, 6]
        V = self.s[:, 7]
        W = self.s[:, 8]
        vt = torch.sqrt(U ** 2 + V ** 2 + W ** 2)
        return vt / 0.3048
    
    def get_TAS(self):
        vt = self.get_vt()
        return vt + self.airspeed * torch.ones_like(vt)
    
    def get_EAS(self):
        TAS = self.get_TAS()
        EAS2TAS = self.get_EAS2TAS()
        EAS = TAS / EAS2TAS
        return EAS
    
    def get_AOA(self):
        return torch.zeros_like(self.s[:, 0])
    
    def get_AOS(self):
        return torch.zeros_like(self.s[:, 0])
    
    def get_angular_velocity(self):
        return self.s[:, 9], self.s[:, 10], self.s[:, 11]
    
    def get_thrust(self):
        return torch.zeros_like(self.u[:, 0])
    
    def get_control_surface(self):
        return torch.zeros_like(self.u[:, 0]), torch.zeros_like(self.u[:, 0]), torch.zeros_like(self.u[:, 0]), torch.zeros_like(self.u[:, 0])
    
    def get_velocity(self):
        return self.s[:, 6] / 0.3048, self.s[:, 7] / 0.3048, self.s[:, 8] / 0.3048
    
    def get_acceleration(self):
        # 根据飞行状态计算三轴加速度
        xdot = self.get_extended_state()
        vel_u, vel_v, vel_w = self.get_velocity()
        u_dot = xdot[:, 6] / 0.3048
        v_dot = xdot[:, 7] / 0.3048
        w_dot = xdot[:, 8] / 0.3048
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
        alt = self.s[:, 2] / 0.3048
        tfac = 1 - .703e-5 * (alt)
        eas2tas = 1 / torch.pow(tfac, 4.14)
        eas2tas = torch.sqrt(eas2tas)
        return eas2tas
    
    def get_accels(self):
        # 根据飞行状态计算三轴过载
        grav = 32.174
        xdot = self.get_extended_state()
        vel_u, vel_v, vel_w = self.get_velocity()
        u_dot = xdot[:, 6] / 0.3048
        v_dot = xdot[:, 7] / 0.3048
        w_dot = xdot[:, 8] / 0.3048
        nx_cg = 1.0 / grav * (u_dot + self.s[:, 10] * vel_w - self.s[:, 11] * vel_v) + torch.sin(self.s[:, 4])
        ny_cg = 1.0 / grav * (v_dot + self.s[:, 11] * vel_u - self.s[:, 9] * vel_w) - torch.cos(self.s[:, 4]) * torch.sin(self.s[:, 3])
        nz_cg = -1.0 / grav * (w_dot + self.s[:, 9] * vel_v - self.s[:, 10] * vel_u) + torch.cos(self.s[:, 4]) * torch.cos(self.s[:, 3])
        return nx_cg, ny_cg, nz_cg

    def get_atmos(self):
        # 根据高度和速度计算动压、马赫数、静压
        alt = self.s[:, 2] / 0.3048
        vt = self.get_vt()
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

# if __name__ == "__main__":
#     uav = UAVDynamics()
#     state = torch.zeros(1, 12)
#     state[:, 0] = 107
#     state[:, 1] = 28
#     state[:, 2] = 600
#     state[:, 8] = 500
#     control = torch.zeros(1, 3)
#     control[:, 2] = 10000
#     for i in range(10):
#         state = odeint(uav, torch.hstack((state, control)),
#                         torch.tensor([0., 0.02], device=torch.device('cuda:0')),
#                         method='euler')[1, :, :12]
#         estate = uav.compute_extended_state(torch.hstack((state, control)))
#         print("第{:}次的坐标为({:},{:},{:})，姿态为({:},{:},{:})".format(i, state[:, 0], state[:, 1], state[:, 2],
#                                                                         state[:, 3], state[:, 4], state[:, 5]))