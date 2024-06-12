import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.pid.rollController import RollController
from algorithms.pid.pitchController import PitchController
from algorithms.pid.yawController import YawController
from algorithms.pid.TECS import TECS
from algorithms.pid.L1Controller import L1Controller
from algorithms.utils.utils import wrap_2PI, wrap_PI
device = "cuda:0"

# fix reset controller
class Controller:
    def __init__(self, airspeed_min=100, airspeed_max=2300, dt=0.02, n=1, device=device):
        self.roll_controller = RollController(dt=dt, n=n, device=device)
        self.pitch_controller = PitchController(dt=dt, n=n, device=device)
        self.yaw_controller = YawController(dt=dt, n=n, device=device)
        self.tecs_controller = TECS(dt=5*dt, n=n, device=device)
        self.l1_controller = L1Controller(dt=5*dt, n=n, device=device)
        self.airspeed_min = airspeed_min
        self.airspeed_max = airspeed_max
        self.dt = dt
        self.n = n
        self.gravity = 32.174
        self.kff_rudder_mix = 0
        self.roll_limit = torch.pi / 4
        self.device = torch.device(device)
        self.speed_scaler = torch.ones((self.n, 1), device=self.device)
        self.ail = torch.zeros((self.n, 1), device=self.device) # 控制roll
        self.el = torch.zeros((self.n, 1), device=self.device) # 控制pitch
        self.rud = torch.zeros((self.n, 1), device=self.device) # 控制yaw
        self.pitch_dem = torch.zeros((self.n, 1), device=self.device) # tecs输出期望俯仰角
        self.roll_dem = torch.zeros((self.n, 1), device=device) # l1controller输出期望滚转角
        self.yaw_dem = torch.zeros((self.n, 1), device=device) # 期望偏航角
        self.yaw_rate_dem = torch.zeros((self.n, 1), device=device) # l1controller输出期望偏航角速度
        self.throttle_dem = torch.zeros((self.n, 1), device=self.device) # tecs输出油门
        self.STEdot_dem = torch.zeros((self.n, 1), device=self.device) # 期望总能量变化率
        self.STEdot_est = torch.zeros((self.n, 1), device=self.device) # 实际总能量变化率
        self.SEBdot_dem = torch.zeros((self.n, 1), device=self.device) # 期望L变化率
        self.SEBdot_est = torch.zeros((self.n, 1), device=self.device) # 实际L变化率

    def calc_speed_scaler(self, TAS):
        scale_min = min(0.5, 1000 / (2 * self.airspeed_max))
        scale_max = max(2.0, 1000 / (0.7 * self.airspeed_min))
        vt = TAS.reshape(-1, 1)
        self.speed_scaler = 1000 / (vt + 1e-8)
        self.speed_scaler = torch.clamp(self.speed_scaler, scale_min, scale_max)
    
    def stabilize_roll(self, env):
        roll, pitch, yaw = env.model.get_posture()
        roll = roll.reshape(-1, 1)
        angle_err = wrap_PI(self.roll_dem - roll)
        self.ail = self.roll_controller.get_servo_out(angle_err, self.speed_scaler, env)

    def stabilize_pitch(self, env):
        roll, pitch, yaw = env.model.get_posture()
        pitch = pitch.reshape(-1, 1)
        angle_err = wrap_PI(self.pitch_dem - pitch)
        self.el = self.pitch_controller.get_servo_out(angle_err, self.speed_scaler, env)
    
    def stabilize_yaw(self, env):
        # yaw = state[:, 5].reshape(-1, 1)
        # angle_err = wrap_PI(self.yaw_dem - yaw)
        self.rud = self.yaw_controller.get_rate_out(self.yaw_rate_dem, self.speed_scaler, env)
        # self.rud = self.yaw_controller.get_servo_out(angle_err, self.speed_scaler, estate, eas2tas)
        # self.rud = self.yaw_controller.get_servo_out(self.speed_scaler, state, acceleration, eas2tas)
    
    def stabilize(self, env):
        TAS = env.model.get_TAS()
        self.calc_speed_scaler(TAS)
        self.stabilize_roll(env)
        self.stabilize_pitch(env)
        self.stabilize_yaw(env)
        # see if we should zero the attitude controller integrators. 
    
    def cal_pitch_throttle(self, hgt_dem, TAS_dem, env):
        self.tecs_controller.update_pitch_throttle(hgt_dem, TAS_dem, env)
        self.pitch_dem = self.tecs_controller.pitch_dem
        self.throttle_dem = self.tecs_controller.throttle_dem
        self.STEdot_dem = self.tecs_controller.STEdot_dem
        self.STEdot_est = self.tecs_controller.STEdot_est
        self.SEBdot_dem = self.tecs_controller.SEBdot_dem
        self.SEBdot_est = self.tecs_controller.SEBdot_est
    
    def update_waypoint(self, prev_WP, next_WP, dist_min, state, estate, eas2tas):
        vt = state[:, 6].reshape(-1, 1)
        # pitch = state[:, 4].reshape(-1, 1)
        self.l1_controller.update_waypoint(prev_WP, next_WP, dist_min, state, estate)
        self.roll_dem = self.l1_controller.nav_roll(state)
        self.roll_dem = torch.clamp(self.roll_dem, -self.roll_limit, self.roll_limit)
        # w = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = w * torch.cos(self.roll_dem) / torch.cos(pitch)
        self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
    
    def update_loiter(self, center_WP, radius, loiter_direction, env):
        TAS = env.model.get_TAS()
        roll, pitch, yaw = env.model.get_posture()
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        vt = TAS.reshape(-1, 1)
        # pitch = state[:, 4].reshape(-1, 1)
        TAS_dem_adj = self.tecs_controller.TAS_dem_adj
        self.l1_controller.update_loiter(center_WP, radius, loiter_direction, env, TAS_dem_adj)
        self.roll_dem = self.l1_controller.nav_roll(pitch)
        self.roll_dem = torch.clamp(self.roll_dem, -self.roll_limit, self.roll_limit)
        # w = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # Q = w * torch.sin(pitch)
        # R = w * torch.cos(self.roll_dem) * torch.cos(pitch)
        # self.yaw_rate_dem = (Q * torch.sin(self.roll_dem) + R * torch.cos(self.roll_dem)) / torch.cos(pitch)
        self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt
    
    def update_heading_hold(self, navigation_heading, env):
        TAS = env.model.get_TAS()
        vt = TAS.reshape(-1, 1)
        roll, pitch, yaw = env.model.get_posture()
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        self.l1_controller.update_heading_hold(navigation_heading, env)
        self.roll_dem = self.l1_controller.nav_roll(pitch)
        self.roll_dem = torch.clamp(self.roll_dem, -self.roll_limit, self.roll_limit)
        # w = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = w * torch.cos(self.roll_dem) / torch.cos(pitch)
        self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt
    
    def update_level_flight(self, env):
        TAS = env.model.get_TAS()
        vt = TAS.reshape(-1, 1)
        roll, pitch, yaw = env.model.get_posture()
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        self.l1_controller.update_level_flight(yaw)
        self.roll_dem = self.l1_controller.nav_roll(pitch)
        self.roll_dem = torch.clamp(self.roll_dem, -self.roll_limit, self.roll_limit)
        # w = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = w * torch.cos(self.roll_dem) / torch.cos(pitch)
        self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt * eas2tas
        # self.yaw_rate_dem = self.gravity * torch.tan(self.roll_dem) / vt

    def get_action(self):
        T = self.throttle_dem
        el = -self.el / 45
        ail = -self.ail / 45
        rud = -self.rud / 45
        action = torch.hstack((T, el))
        action = torch.hstack((action, ail))
        action = torch.hstack((action, rud))
        return action
