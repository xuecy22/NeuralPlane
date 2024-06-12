import torch
import time
import os
import sys
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.utils import parse_config
device = "cuda:0"


# fix: self.dt 修改为控制器实际调用周期
# fix: 油门状态是否需要记录？
# fix: L rate初始变化方向？
# fix: TAS滤波去掉？
# fix: SEBdot_dem 是否需要加入期望的势能变化率？
# fix: integKE 是否需要加入动能误差的积分？
# fix: acc滤波去掉？
# fix: 调整积分参数
class TECS:
    def __init__(self, config='tecs', airspeed_min=100, airspeed_max=2300, dt=0.1, n=1, device=device):
        """
        TAS_state: 速度滤波器输出
        acc_x_lpf: 加速度低通滤波
        """
        self.config = parse_config(config)
        self.airspeed_min = airspeed_min
        self.airspeed_max = airspeed_max
        self.dt = dt
        self.n = n
        self.device = torch.device(device)
        self.reset = True
        self.last_time = time.time()
        self.maxClimbRate = getattr(self.config, 'maxClimbRate') / 0.3048
        self.minSinkRate = getattr(self.config, 'minSinkRate') / 0.3048
        self.maxSinkRate = getattr(self.config, 'maxSinkRate') / 0.3048
        self.timeConst = getattr(self.config, 'timeConst') # 时间常数
        self.thrDamp = getattr(self.config, 'thrDamp')
        self.integGain = getattr(self.config, 'integGain')
        self.vertAccLim = getattr(self.config, 'vertAccLim') / 0.3048
        self.hgtCompFiltOmega = getattr(self.config, 'hgtCompFiltOmega') # height filter
        self.spdCompFiltOmega = getattr(self.config, 'spdCompFiltOmega') # speed filter
        self.rollComp = getattr(self.config, 'rollComp')
        self.spdWeight = getattr(self.config, 'spdWeight')
        self.pitchDamp = getattr(self.config, 'pitchDamp')
        self.pitch_max = getattr(self.config, 'maxPitch') * torch.pi / 180
        self.pitch_min = getattr(self.config, 'minPitch') * torch.pi / 180
        self.throttle_cruise = getattr(self.config, 'throttle_cruise')
        self.THR_max = getattr(self.config, 'throttle_max') * 0.01
        self.THR_min = getattr(self.config, 'throttle_min') * 0.01
        self.gravity = getattr(self.config, 'gravity')
        self.hgt_dem_tconst = 5
        self.SKE_weighting = 1
        # 基本状态量
        self.height = torch.zeros((self.n, 1), device=self.device)
        self.climb_rate = torch.zeros((self.n, 1), device=self.device) # 爬升速率
        self.acc_x = torch.zeros((self.n, 1), device=self.device)
        self.acc_x_lpf = torch.zeros((self.n, 1), device=self.device)
        # 真实空速相关
        # self.EAS_dem = torch.zeros((self.n, 1), device=self.device)
        self.TAS_dem = torch.zeros((self.n, 1), device=self.device)
        self.TAS_dem_adj = torch.zeros((self.n, 1), device=self.device)
        self.TAS_rate_dem = torch.zeros((self.n, 1), device=self.device)
        self.TAS_rate_dem_lpf = torch.zeros((self.n, 1), device=self.device)
        self.TAS_max = torch.zeros((self.n, 1), device=self.device)
        self.TAS_min = torch.zeros((self.n, 1), device=self.device)
        self.integDTAS_state = torch.zeros((self.n, 1), device=self.device)
        self.TAS_state = torch.zeros((self.n, 1), device=self.device)
        # 高度相关
        self.hgt_dem = torch.zeros((self.n, 1), device=self.device)
        self.hgt_dem_in = torch.zeros((self.n, 1), device=self.device)
        self.hgt_dem_in_prev = torch.zeros((self.n, 1), device=self.device)
        self.hgt_dem_rate_ltd = torch.zeros((self.n, 1), device=self.device)
        self.hgt_rate_dem = torch.zeros((self.n, 1), device=self.device)
        self.hgt_dem_lpf = torch.zeros((self.n, 1), device=self.device)
        self.post_TO_hgt_offset = torch.zeros((self.n, 1), device=self.device)
        # 俯仰角相关
        self.pitch_dem_unc = torch.zeros((self.n, 1), device=self.device)
        self.climb_rate_limit = torch.zeros((self.n, 1), device=self.device) # 爬升速率限制
        self.sink_rate_limit = torch.zeros((self.n, 1), device=self.device) # 下降速率限制
        self.max_climb_scaler = torch.ones((self.n, 1), device=self.device)
        self.max_sink_scaler = torch.ones((self.n, 1), device=self.device)
        self.integSEBdot = torch.zeros((self.n, 1), device=self.device)
        self.integKE = torch.zeros((self.n, 1), device=self.device)
        self.last_pitch_dem = torch.zeros((self.n, 1), device=self.device)
        # 动能势能相关
        self.STEdot_max = torch.zeros((self.n, 1), device=self.device)
        self.STEdot_min = torch.zeros((self.n, 1), device=self.device)
        self.STEdot_dem = torch.zeros((self.n, 1), device=self.device)
        self.STEdot_est = torch.zeros((self.n, 1), device=self.device)
        self.SPE_dem = torch.zeros((self.n, 1), device=self.device)
        self.SKE_dem = torch.zeros((self.n, 1), device=self.device) # 动能
        self.SPEdot_dem = torch.zeros((self.n, 1), device=self.device)
        self.SKEdot_dem = torch.zeros((self.n, 1), device=self.device)
        self.SPE_est = torch.zeros((self.n, 1), device=self.device) # 势能
        self.SKE_est = torch.zeros((self.n, 1), device=self.device)
        self.SPEdot = torch.zeros((self.n, 1), device=self.device)
        self.SKEdot = torch.zeros((self.n, 1), device=self.device)
        self.STE_error = torch.zeros((self.n, 1), device=self.device)
        self.STEdotErrLast = torch.zeros((self.n, 1), device=self.device)
        self.SEB_dem = torch.zeros((self.n, 1), device=self.device)
        self.SEB_est = torch.zeros((self.n, 1), device=self.device)
        self.SEBdot_dem = torch.zeros((self.n, 1), device=self.device)
        self.SEBdot_est = torch.zeros((self.n, 1), device=self.device)
        # 油门输出
        self.throttle_dem = torch.zeros((self.n, 1), device=self.device)
        self.integTHR_state = torch.zeros((self.n, 1), device=self.device)

    def update(self, env):
        npos, epos, altitude = env.model.get_position()
        climb_rate = env.model.get_climb_rate()
        roll, pitch, yaw = env.model.get_posture()
        if self.reset:
            self.climb_rate_limit = self.maxClimbRate * self.max_climb_scaler
            self.sink_rate_limit = self.maxSinkRate * self.max_sink_scaler
            self.last_pitch_dem = pitch.reshape(-1, 1)
            self.hgt_dem = altitude.reshape(-1, 1)
            self.hgt_dem_in_prev = altitude.reshape(-1, 1)
            self.hgt_dem_lpf = altitude.reshape(-1, 1)
            self.hgt_dem_rate_ltd = altitude.reshape(-1, 1)
            self.hgt_dem_prev = altitude.reshape(-1, 1)
        self.height = altitude.reshape(-1, 1)
        self.climb_rate = climb_rate.reshape(-1, 1)
        self.STEdot_max = self.climb_rate_limit * self.gravity
        self.STEdot_min = -self.sink_rate_limit * self.gravity
        self.update_speed(env)

    def update_speed(self, env):
        # caculate TAS_dem TAS_max TAS_min acc_x acc_x_lpf
        TAS = env.model.get_TAS()
        ax, ay, az = env.model.get_acceleration()
        vt = TAS.reshape(-1, 1)
        eas2tas = env.model.get_EAS2TAS().reshape(-1, 1)
        # pitch = state[:, 4].reshape(-1, 1)
        self.acc_x = ax.reshape(-1, 1)
        # self.acc_x = acceleration[:, 0].reshape(-1, 1) + self.gravity * torch.sin(pitch)
        # self.TAS_dem = self.EAS_dem * eas2tas
        if self.reset:
            self.acc_x_lpf = self.acc_x
            self.TAS_max = self.airspeed_max * eas2tas
        else:
            alpha = self.dt / (self.dt + self.timeConst)
            self.acc_x_lpf = self.acc_x_lpf * (1 - alpha) + self.acc_x * alpha
            # velRateMax = 0.5 * self.STEdot_max / torch.max(self.TAS_state, self.airspeed_min * eas2tas)
            # self.TAS_max = self.TAS_max + self.dt * velRateMax
        self.TAS_max = torch.min(self.TAS_max, self.airspeed_max * eas2tas)
        self.TAS_min = self.airspeed_min * eas2tas
        self.TAS_max = torch.max(self.TAS_max, self.TAS_min)
        # caculate TAS_state
        self.TAS_state = vt
        # if self.reset:
        #     self.TAS_state = vt
        #     self.integDTAS_state = torch.zeros((self.n, 1), device=self.device)
        # else:
        #     aspdErr = vt - self.TAS_state
        #     integDTAS_input = aspdErr * self.spdCompFiltOmega * self.spdCompFiltOmega
        #     self.integDTAS_state = self.integDTAS_state + integDTAS_input * self.dt
        #     TAS_input = self.integDTAS_state + self.acc_x + aspdErr * self.spdCompFiltOmega * 1.4142
        #     self.TAS_state = self.TAS_state + TAS_input * self.dt
    
    def update_speed_demand(self):
        if self.reset:
            self.TAS_dem_adj = self.TAS_state
        self.TAS_dem = torch.clamp(self.TAS_dem, self.TAS_min, self.TAS_max)
        # velRateMax = 0.5 * self.STEdot_max / self.TAS_state
        # velRateMin = 0.5 * self.STEdot_min / self.TAS_state
        velRateMax = self.STEdot_max / self.TAS_state
        velRateMin = self.STEdot_min / self.TAS_state
        TAS_dem_previous = self.TAS_dem_adj
        mask1 = (self.TAS_dem - TAS_dem_previous) > (velRateMax * self.dt)
        mask2 = (self.TAS_dem - TAS_dem_previous) < (velRateMin * self.dt)
        mask3 = ~(mask1 | mask2)
        self.TAS_dem_adj = (TAS_dem_previous + velRateMax * self.dt) * mask1
        self.TAS_dem_adj += (TAS_dem_previous + velRateMin * self.dt) * mask2
        self.TAS_dem_adj += self.TAS_dem * mask3
        self.TAS_rate_dem = velRateMax * mask1 + velRateMin * mask2
        self.TAS_rate_dem += (self.TAS_dem - TAS_dem_previous) / self.dt * mask3
        # calculate a low pass filtered _TAS_rate_dem
        if self.reset:
            self.TAS_rate_dem_lpf = self.TAS_rate_dem
            self.reset = False
        else:
            alpha = self.dt / (self.dt + self.timeConst)
            self.TAS_rate_dem_lpf = self.TAS_rate_dem_lpf * (1 - alpha) + self.TAS_rate_dem * alpha
        self.TAS_dem_adj = torch.clamp(self.TAS_dem_adj, self.TAS_min, self.TAS_max)
    
    def update_height_demand(self):
        self.climb_rate_limit = self.maxClimbRate * self.max_climb_scaler
        self.sink_rate_limit = self.maxSinkRate * self.max_sink_scaler
        hgt_dem = 0.5 * (self.hgt_dem_in + self.hgt_dem_in_prev)
        self.hgt_dem_in_prev = self.hgt_dem_in
        mask1 = (hgt_dem - self.hgt_dem_rate_ltd) > (self.climb_rate_limit * self.dt)
        mask2 = (hgt_dem - self.hgt_dem_rate_ltd) < (-self.sink_rate_limit * self.dt)
        mask3 = ~(mask1 | mask2)
        self.hgt_dem_rate_ltd += self.climb_rate_limit * self.dt * mask1
        self.hgt_dem_rate_ltd += -self.sink_rate_limit * self.dt * mask2
        self.hgt_dem_rate_ltd = self.hgt_dem_rate_ltd * ~mask3 + self.hgt_dem * mask3
        coef = min(self.dt / (self.dt + max(self.hgt_dem_tconst, self.dt)), 1)
        self.hgt_rate_dem = (self.hgt_dem_rate_ltd - self.hgt_dem_lpf) / self.hgt_dem_tconst
        self.hgt_dem_lpf = self.hgt_dem_rate_ltd * coef + (1 - coef) * self.hgt_dem_lpf
        self.post_TO_hgt_offset *= 1 - coef
        self.hgt_dem = self.hgt_dem_lpf + self.post_TO_hgt_offset
        max_climb_condition = self.pitch_dem_unc > self.pitch_max
        max_descent_condition = self.pitch_dem_unc < self.pitch_min
        hgt_dem_alpha = self.dt / max(self.dt + self.hgt_dem_tconst, self.dt)
        mask1 = max_climb_condition & (self.hgt_dem > self.hgt_dem_prev)
        mask2 = max_descent_condition & (self.hgt_dem < self.hgt_dem_prev)
        mask3 = ~(mask1 | mask2)
        self.max_climb_scaler = self.max_climb_scaler * ~mask1 + self.max_climb_scaler * (1 - hgt_dem_alpha) * mask1
        self.max_climb_scaler = (self.max_climb_scaler * (1 - hgt_dem_alpha) + hgt_dem_alpha) * mask3 + self.max_climb_scaler * ~mask3
        self.max_sink_scaler = self.max_sink_scaler * ~mask2 + self.max_sink_scaler * (1 - hgt_dem_alpha) * mask2
        self.max_sink_scaler = (self.max_sink_scaler * (1 - hgt_dem_alpha) + hgt_dem_alpha) * mask3 + self.max_sink_scaler * ~mask3
        self.hgt_dem_prev = self.hgt_dem

    def update_energies(self):
        self.SPE_dem = self.hgt_dem * self.gravity
        self.SKE_dem = 0.5 * self.TAS_dem_adj * self.TAS_dem_adj
        self.SKEdot_dem = self.TAS_state * (self.TAS_rate_dem - self.TAS_rate_dem_lpf)
        # self.SKEdot_dem = self.TAS_state * self.TAS_rate_dem
        self.SPE_est = self.height * self.gravity
        self.SKE_est = 0.5 * self.TAS_state * self.TAS_state
        self.SPEdot = self.climb_rate * self.gravity
        self.SKEdot = self.TAS_state * (self.acc_x - self.acc_x_lpf)
        # self.SKEdot = self.TAS_state * self.acc_x
        self.STEdot_est = self.SPEdot + self.SKEdot
    
    def update_throttle_with_airspeed(self, env):
        # SPE_err_max = torch.max(self.SKE_est - 0.5 * self.TAS_min * self.TAS_min, torch.zeros((self.n, 1), device=self.device))
        # SPE_err_min = torch.min(self.SKE_est - 0.5 * self.TAS_max * self.TAS_max, torch.zeros((self.n, 1), device=self.device))
        SPE_err_max = torch.max(0.5 * self.TAS_max * self.TAS_max - self.SKE_dem, torch.zeros((self.n, 1), device=self.device))
        SPE_err_min = torch.min(0.5 * self.TAS_min * self.TAS_min - self.SKE_dem, torch.zeros((self.n, 1), device=self.device))
        # rate of change of potential energy is proportional to height error
        self.SPEdot_dem = (self.SPE_dem - self.SPE_est) / self.timeConst
        # Calculate total energy error
        self.STE_error = torch.clamp(self.SPE_dem - self.SPE_est, SPE_err_min, SPE_err_max) + self.SKE_dem - self.SKE_est
        # self.STEdot_dem = self.SPEdot_dem + self.SKEdot_dem
        self.STEdot_dem = torch.clamp(self.SPEdot_dem + self.SKEdot_dem, self.STEdot_min, self.STEdot_max)
        STEdot_error = self.STEdot_dem - self.SPEdot - self.SKEdot
        # Apply 0.5 second first order filter to STEdot_error
        # This is required to remove accelerometer noise from the  measurement
        filt_coef = 2 * self.dt
        STEdot_error = filt_coef * STEdot_error + (1 - filt_coef) * self.STEdotErrLast
        self.STEdotErrLast = STEdot_error
        # Calculate throttle demand
        # Calculate gain scaler from specific energy error to throttle
        K_STE2Thr = (self.THR_max - self.THR_min) / (self.timeConst * (self.STEdot_max - self.STEdot_min))
        # Calculate feed-forward throttle
        nomThr = self.throttle_cruise * 0.01
        # Use the demanded rate of change of total energy as the feed-forward demand, but add
        # additional component which scales with (1/cos(bank angle) - 1) to compensate for induced
        # drag increase during turns.
        roll, pitch, yaw = env.model.get_posture()
        roll = roll.reshape(-1, 1)
        pitch = pitch.reshape(-1, 1)
        yaw = yaw.reshape(-1, 1)
        a = torch.cos(yaw) * torch.sin(roll) * torch.sin(pitch) - torch.cos(roll) * torch.sin(yaw)
        b = torch.cos(yaw) * torch.cos(roll) + torch.sin(yaw) * torch.sin(roll) * torch.sin(pitch)
        cosPhi = torch.sqrt(a * a + b * b)
        self.STEdot_dem = self.STEdot_dem + self.rollComp * (1 / torch.clamp(cosPhi * cosPhi, 0.1, 1) - 1)
        ff_throttle = nomThr + self.STEdot_dem / (self.STEdot_max - self.STEdot_min) * (self.THR_max - self.THR_min)
        # Calculate PD + FF throttle
        throttle_damp = self.thrDamp
        self.throttle_dem = (self.STE_error + STEdot_error * throttle_damp) * K_STE2Thr + ff_throttle
        THRmin_clipped_to_zero = min(max(self.THR_min, 0), self.THR_max)
        # Calculate integrator state upper and lower limits
        # Set to a value that will allow 0.1 (10%) throttle saturation to allow for noise on the demand
        # Additionally constrain the integrator state amplitude so that the integrator comes off limits faster.
        maxAmp = 0.5 * (self.THR_max - THRmin_clipped_to_zero)
        integ_max = torch.clamp(self.THR_max - self.throttle_dem + 0.1, -maxAmp, maxAmp)
        integ_min = torch.clamp(self.THR_min - self.throttle_dem - 0.1, -maxAmp, maxAmp)
        # Calculate integrator state, constraining state
        # Set integrator to a max throttle value during climbout
        self.integTHR_state = self.integTHR_state + self.STE_error * self.integGain * self.dt * K_STE2Thr
        # self.integTHR_state = self.integTHR_state + STEdot_error * self.integGain * self.dt * K_STE2Thr
        self.integTHR_state = torch.clamp(self.integTHR_state, integ_min, integ_max)
        # Sum the components.
        self.throttle_dem = 0.5 * self.throttle_dem + self.integTHR_state
        # Constrain throttle demand and record clipping
        self.throttle_dem = torch.clamp(self.throttle_dem, self.THR_min, self.THR_max)
        # pdb.set_trace()
    
    # def update_pitch(self):
    #     # Calculate Speed/Height Control Weighting
    #     # This is used to determine how the pitch control prioritises speed and height control
    #     # A weighting of 1 provides equal priority (this is the normal mode of operation)
    #     # A SKE_weighting of 0 provides 100% priority to height control. This is used when no airspeed measurement is available
    #     # A SKE_weighting of 2 provides 100% priority to speed control. This is used when an underspeed condition is detected. In this instance, if airspeed
    #     # rises above the demanded value, the pitch angle will be increased by the TECS controller.
    #     self.SKE_weighting = min(max(self.spdWeight, 0), 2)
    #     SPE_weighting = 2 - self.SKE_weighting
    #     # either weight can fade to 0, but don't go above 1 to prevent instability if tuned at a speed weight of 1 and wieghting is varied to end points in flight.
    #     SPE_weighting = min(SPE_weighting, 1)
    #     self.SKE_weighting = min(self.SKE_weighting, 1)
    #     # Calculate demanded specific energy balance and error
    #     self.SEB_dem = self.SPE_dem * SPE_weighting - self.SKE_dem * self.SKE_weighting
    #     self.SEB_est = self.SPE_est * SPE_weighting - self.SKE_est * self.SKE_weighting
    #     SEB_error = self.SEB_dem - self.SEB_est
    #     # track demanded height using the specified time constant
    #     # self.SEBdot_dem = self.hgt_rate_dem * self.gravity * SPE_weighting + SEB_error / self.timeConst
    #     self.SEBdot_dem = SEB_error / self.timeConst
    #     # self.SEBdot_dem = self.SPEdot_dem * SPE_weighting - self.SKEdot_dem * self.SKE_weighting
    #     SEBdot_dem_min = -self.maxSinkRate * self.gravity
    #     SEBdot_dem_max = self.maxClimbRate * self.gravity
    #     self.SEBdot_dem = torch.clamp(self.SEBdot_dem, SEBdot_dem_min, SEBdot_dem_max)
    #     # calculate specific energy balance rate error
    #     self.SEBdot_est = self.SPEdot * SPE_weighting - self.SKEdot * self.SKE_weighting
    #     SEBdot_error = self.SEBdot_dem - self.SEBdot_est
    #     # sum predicted plus damping correction
    #     # integral correction is added later
    #     # During flare a different damping gain is used
    #     pitch_damp = self.pitchDamp
    #     SEBdot_dem_total = self.SEBdot_dem + SEBdot_error * pitch_damp
    #     # inverse of gain from SEB to pitch angle
    #     gainInv = self.TAS_state * self.gravity
    #     # don't allow the integrator to rise by more than 20% of its full
    #     # Calculate max and min values for integrator state that will allow for no more than
    #     # 5deg of saturation. This allows for some pitch variation due to gusts before the
    #     # integrator is clipped. Otherwise the effectiveness of the integrator will be reduced in turbulence
    #     integSEBdot_min = (gainInv * (self.pitch_min - torch.pi / 36)) - SEBdot_dem_total
    #     integSEBdot_max = (gainInv * (self.pitch_max + torch.pi / 36)) - SEBdot_dem_total
    #     # Calculate integrator state, constraining input if pitch limits are exceeded
    #     # don't allow the integrator to rise by more than 10% of its full
    #     # range in one step. This prevents single value glitches from
    #     # causing massive integrator changes. See Issue#4066
    #     integSEB_range = integSEBdot_max - integSEBdot_min
    #     integSEB_delta = torch.clamp(SEBdot_error * self.integGain * self.dt, -integSEB_range * 0.1, integSEB_range * 0.1)
    #     # predict what pitch will be with uncontrained integration
    #     # self.pitch_dem_unc = (SEBdot_dem_total + self.integSEBdot + integSEB_delta + self.integKE) / gainInv
    #     # integrate SEB rate error and apply integrator state limits
    #     inhibit_integrator = ((self.pitch_dem_unc > self.pitch_max) & (integSEB_delta > 0)) | \
    #         ((self.pitch_dem_unc < self.pitch_min) & (integSEB_delta < 0))
    #     coef = 1 - self.dt / (self.dt + self.timeConst)
    #     self.integSEBdot += ~inhibit_integrator * integSEB_delta
    #     self.integSEBdot = self.integSEBdot * ~inhibit_integrator + self.integSEBdot * coef * inhibit_integrator
    #     self.integKE += ((self.SKE_est - self.SKE_dem) * self.SKE_weighting * self.dt / self.timeConst) * ~inhibit_integrator
    #     self.integKE = self.integKE * ~inhibit_integrator + self.integKE * coef * inhibit_integrator
    #     self.integSEBdot = torch.clamp(self.integSEBdot, integSEBdot_min, integSEBdot_max)
    #     KE_integ_limit = 0.25 * (self.pitch_max - self.pitch_min) * gainInv # allow speed trim integrator to access 505 of pitch range
    #     self.integKE = torch.clamp(self.integKE, -KE_integ_limit, KE_integ_limit)
    #     # Calculate pitch demand from specific energy balance signals
    #     # self.pitch_dem_unc = (0.8 * SEBdot_dem_total + self.integSEBdot + 0.5 * self.integKE) / gainInv
    #     self.pitch_dem_unc = (0.8 * SEBdot_dem_total + self.integSEBdot) / gainInv
    #     # Constrain pitch demand
    #     self.pitch_dem = torch.clamp(self.pitch_dem_unc, self.pitch_min, self.pitch_max)
    #     # Rate limit the pitch demand to comply with specified vertical
    #     # acceleration limit
    #     ptchRateIncr = self.dt * self.vertAccLim / self.TAS_state
    #     mask1 = (self.pitch_dem - self.last_pitch_dem) > ptchRateIncr
    #     mask2 = (self.pitch_dem - self.last_pitch_dem) < -ptchRateIncr
    #     mask3 = ~(mask1 | mask2)
    #     self.pitch_dem = (self.last_pitch_dem + ptchRateIncr) * mask1 + (self.last_pitch_dem - ptchRateIncr) * mask2 + self.pitch_dem * mask3
    #     self.last_pitch_dem = self.pitch_dem
    #     # pdb.set_trace()
    
    def update_pitch(self):
        self.SKE_weighting = min(max(self.spdWeight, 0), 2)
        SPE_weighting = 2 - self.SKE_weighting
        # either weight can fade to 0, but don't go above 1 to prevent instability if tuned at a speed weight of 1 and wieghting is varied to end points in flight.
        SPE_weighting = min(SPE_weighting, 1)
        self.SKE_weighting = min(self.SKE_weighting, 1)
        # Calculate demanded specific energy balance and error
        self.SEB_dem = self.SPE_dem * SPE_weighting - self.SKE_dem * self.SKE_weighting
        self.SEB_est = self.SPE_est * SPE_weighting - self.SKE_est * self.SKE_weighting
        SEB_error = self.SEB_dem - self.SEB_est
        # track demanded height using the specified time constant
        self.SEBdot_dem = self.SPEdot_dem * SPE_weighting - self.SKEdot_dem * self.SKE_weighting
        SEBdot_dem_min = -self.maxSinkRate * self.gravity
        SEBdot_dem_max = self.maxClimbRate * self.gravity
        self.SEBdot_dem = torch.clamp(self.SEBdot_dem, SEBdot_dem_min, SEBdot_dem_max)
        # calculate specific energy balance rate error
        self.SEBdot_est = self.SPEdot * SPE_weighting - self.SKEdot * self.SKE_weighting
        SEBdot_error = self.SEBdot_dem - self.SEBdot_est
        # sum predicted plus damping correction
        # integral correction is added later
        # During flare a different damping gain is used
        pitch_damp = self.pitchDamp
        SEBdot_dem_total = 0.5 * self.SEBdot_dem * self.timeConst + SEBdot_error * pitch_damp + 0.8 * SEB_error
        # inverse of gain from SEB to pitch angle
        gainInv = self.TAS_state * self.gravity * self.timeConst
        mask1 = self.pitch_dem_unc > self.pitch_max
        mask2 = self.pitch_dem_unc < self.pitch_min
        mask3 = ~(mask1 | mask2)
        integSEB_delta = torch.min(SEB_error * self.integGain, self.pitch_max - self.pitch_dem_unc) * mask1
        integSEB_delta += torch.min(SEB_error * self.integGain, self.pitch_min - self.pitch_dem_unc) * mask2
        integSEB_delta += SEB_error * self.integGain * mask3
        # predict what pitch will be with uncontrained integration
        # self.pitch_dem_unc = (SEBdot_dem_total + self.integSEBdot + integSEB_delta + self.integKE) / gainInv
        # integrate SEB rate error and apply integrator state limits
        inhibit_integrator = ((self.pitch_dem_unc > self.pitch_max) & (integSEB_delta > 0)) | \
            ((self.pitch_dem_unc < self.pitch_min) & (integSEB_delta < 0))
        coef = 1 - self.dt / (self.dt + self.timeConst)
        self.integSEBdot += ~inhibit_integrator * integSEB_delta * self.dt
        self.integSEBdot = self.integSEBdot * ~inhibit_integrator + self.integSEBdot * coef * inhibit_integrator
        self.integKE += ((self.SKE_est - self.SKE_dem) * self.SKE_weighting * self.dt / self.timeConst) * ~inhibit_integrator
        self.integKE = self.integKE * ~inhibit_integrator + self.integKE * coef * inhibit_integrator
        KE_integ_limit = 0.25 * (self.pitch_max - self.pitch_min) * gainInv # allow speed trim integrator to access 505 of pitch range
        self.integKE = torch.clamp(self.integKE, -KE_integ_limit, KE_integ_limit)
        # Calculate pitch demand from specific energy balance signals
        # self.pitch_dem_unc = (0.8 * SEBdot_dem_total + self.integSEBdot + 0.5 * self.integKE) / gainInv
        self.pitch_dem_unc = (SEBdot_dem_total + self.integSEBdot) / gainInv
        # Constrain pitch demand
        self.pitch_dem = torch.clamp(self.pitch_dem_unc, self.pitch_min, self.pitch_max)
        # Rate limit the pitch demand to comply with specified vertical
        # acceleration limit
        ptchRateIncr = self.dt * self.vertAccLim / self.TAS_state
        mask1 = (self.pitch_dem - self.last_pitch_dem) > ptchRateIncr
        mask2 = (self.pitch_dem - self.last_pitch_dem) < -ptchRateIncr
        mask3 = ~(mask1 | mask2)
        self.pitch_dem = (self.last_pitch_dem + ptchRateIncr) * mask1 + (self.last_pitch_dem - ptchRateIncr) * mask2 + self.pitch_dem * mask3
        self.last_pitch_dem = self.pitch_dem
        # pdb.set_trace()
    
    def update_pitch_throttle(self, hgt_dem, TAS_dem, env):
        # Convert inputs
        hgt_dem_in_raw = hgt_dem
        self.TAS_dem = TAS_dem
        # Don't allow height deamnd to continue changing in a direction that saturates vehicle manoeuvre limits
        # if vehicle is unable to follow the demanded climb or descent.
        max_climb_condition = self.pitch_dem_unc > self.pitch_max
        max_descent_condition = self.pitch_dem_unc < self.pitch_min
        mask1 = max_climb_condition & (hgt_dem_in_raw > self.hgt_dem_in_prev)
        mask2 = max_descent_condition & (hgt_dem_in_raw < self.hgt_dem_in_prev)
        mask3 = ~(mask1 | mask2)
        self.hgt_dem_in = hgt_dem_in_raw * mask3 + self.hgt_dem_in_prev * ~mask3
        self.THR_max = max(self.THR_max, self.THR_min + 0.01)
        # don't allow max pitch to go below min pitch
        self.pitch_max = max(self.pitch_max, self.pitch_min)
        # initialise selected states and variables if DT > 1 second or in climbout
        self.update(env)
        # Calculate the speed demand
        self.update_speed_demand()
        # Calculate the height demand
        self.update_height_demand()
        # Calculate specific energy quantitiues
        self.update_energies()
        # Calculate pitch demand
        self.update_pitch()
        # Calculate throttle demand - use simple pitch to throttle if no
        # airspeed sensor.
        # Note that caller can demand the use of
        # synthetic airspeed for one loop if needed. This is required
        # during QuadPlane transition when pitch is constrained
        self.update_throttle_with_airspeed(env)
        # now = time.time()
        # sim_time = now - self.last_time
        # self.reset = sim_time > 1
        # self.last_time = now
