import torch
import math
import time
import os
import sys
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.utils import parse_config, get_diff_angle, get_length, get_vector_dot, get_cross_error, wrap_PI
device = "cuda:0"


class L1Controller:
    def __init__(self, config='l1controller', dt=0.1, n=1, device=device):
        self.config = parse_config(config)
        self.dt = dt
        self.n = n
        self.device = torch.device(device)
        self.last_time = time.time()
        self.L1_period = getattr(self.config, 'L1_period')
        self.L1_damping = getattr(self.config, 'L1_damping')
        self.L1_xtrack_i_gain = getattr(self.config, 'L1_xtrack_i_gain')
        self.loiter_bank_limit = getattr(self.config, 'loiter_bank_limit')
        self.gravity = getattr(self.config, 'gravity')
        self.current_loc = torch.zeros((self.n, 2), device=self.device)
        self.ground_speed = torch.zeros((self.n, 2), device=self.device)
        self.L1_xtrack_i = torch.zeros((self.n, 1), device=self.device)
        self.L1_dist = torch.zeros((self.n, 1), device=self.device)
        self.target_bearing = torch.zeros((self.n, 1), device=self.device) # 当前位置到目标点航向角/rad
        self.crosstrack_error = torch.zeros((self.n, 1), device=self.device) # 当前位置到航线的垂线长度
        self.nav_bearing = torch.zeros((self.n, 1), device=self.device)
        self.Nu = torch.zeros((self.n, 1), device=self.device)
        self.last_Nu = torch.zeros((self.n, 1), device=self.device)
        self.latAccDem = torch.zeros((self.n, 1), device=self.device) # 横向加速度/ft/s^2
        self.bearing_error = torch.zeros((self.n, 1), device=self.device)
        self.WPcircle = torch.zeros((self.n, 1), dtype=torch.bool, device=self.device)
    
    # prevent indecision in our turning by using our previous turn
    # decision if we are in a narrow angle band pointing away from the
    # target and the turn angle has changed sign
    def prevent_indecision(self, yaw):
        Nu_limit = 0.9 * torch.pi
        mask1 = (torch.abs(self.Nu) > Nu_limit) & (torch.abs(self.last_Nu) > Nu_limit) &\
            (torch.abs(wrap_PI(self.target_bearing - yaw)) > (2 * torch.pi / 3)) & ((self.Nu * self.last_Nu) < 0)
        # we are moving away from the target waypoint and pointing
        # away from the waypoint (not flying backwards). The sign
        # of Nu has also changed, which means we are
        # oscillating in our decision about which way to go
        self.Nu = self.last_Nu * mask1 + self.Nu * ~mask1
    
    def loiter_radius(self, radius, eas2tas, TAS_dem):
        # prevent an insane loiter bank limit
        sanitized_bank_limit = min(max(self.loiter_bank_limit, 0), 89)
        lateral_accel_sea_level = math.tan(sanitized_bank_limit * torch.pi / 180) * self.gravity
        nominal_velocity_sea_level =  TAS_dem / eas2tas
        eas2tas_sq = eas2tas * eas2tas
        # Missing a sane input for calculating the limit, or the user has
        # requested a straight scaling with altitude. This will always vary
        # with the current altitude, but will at least protect the airframe
        mask1 = (abs(sanitized_bank_limit) < 1e-6) | (torch.abs(nominal_velocity_sea_level) < 1e-6) |\
              (abs(lateral_accel_sea_level) < 1e-6)
        result = radius * eas2tas_sq * mask1
        sea_level_radius = nominal_velocity_sea_level * nominal_velocity_sea_level / (lateral_accel_sea_level + 1e-6)
        mask2 = (~mask1) & (sea_level_radius > radius)
        result += radius * eas2tas_sq * mask2
        mask3 = ~(mask1 | mask2)
        result += torch.max(sea_level_radius * eas2tas_sq, radius) * mask3
        return result

    # update L1 control for waypoint navigation
    def update_waypoint(self, prev_WP, next_WP, dist_min, state, estate):
        now = time.time()
        sim_time = now - self.last_time
        if sim_time > 1:
            self.L1_xtrack_i = torch.zeros((self.n, 1), device=self.device)
        self.last_time = now
        # Calculate L1 gain required for specified damping
        K_L1 = 4.0 * self.L1_damping * self.L1_damping
        # Get current position and velocity
        self.current_loc = state[:, :2]
        self.ground_speed = estate[:, :2]
        # update target_bearing
        self.target_bearing = get_diff_angle(self.current_loc, next_WP)
        # Calculate groundspeed
        groundSpeed = get_length(self.ground_speed)
        # Calculate time varying control parameters
        # Calculate the L1 length required for specified period
        # 0.3183099 = 1 / pi
        self.L1_dist = torch.max(self.L1_damping * self.L1_period * groundSpeed / torch.pi, dist_min * torch.ones((self.n, 1), device=self.device))
        # Calculate the NE position of WP B relative to WP A
        AB = next_WP - prev_WP
        AB_length = get_length(AB)
        # Check for AB zero length and track directly to the destination if too small
        mask1 = AB_length < 1e-6
        AB = AB * ~mask1 + (next_WP - self.current_loc) * mask1
        AB_length = get_length(AB)
        mask1 = AB_length < 1e-6
        yaw = state[:, 5].reshape(-1, 1)
        AB = AB * ~mask1 + torch.hstack((torch.cos(yaw), torch.sin(yaw))) * mask1
        AB_length = get_length(AB)
        AB = AB / AB_length
        # Calculate the NE position of the aircraft relative to WP A
        A_air = self.current_loc - prev_WP
        # calculate distance to target track, for reporting
        self.crosstrack_error = get_cross_error(A_air, AB)
        # Determine if the aircraft is behind a +-135 degree degree arc centred on WP A
        # and further than L1 distance from WP A. Then use WP A as the L1 reference point
        # Otherwise do normal L1 guidance
        WP_A_dist = get_length(A_air)
        alongTrackDist = get_vector_dot(A_air, AB)
        mask1 = (WP_A_dist > self.L1_dist) & ((alongTrackDist / torch.max(WP_A_dist, torch.ones(self.n, 1))) < -0.7071)
        # Calc Nu to fly To WP A
        A_air_unit = A_air / get_length(A_air) # Unit vector from WP A to aircraft
        xtrackVel = get_cross_error(self.ground_speed, -A_air_unit) # Velocity across line
        ltrackVel = get_vector_dot(self.ground_speed, -A_air_unit) # Velocity along line
        self.Nu = torch.atan2(xtrackVel, ltrackVel) * mask1
        # bearing (radians) from AC to L1 point
        self.nav_bearing = torch.atan2(-A_air_unit[:, 1] , -A_air_unit[:, 0]).reshape(-1, 1) * mask1
        mask2 = (~mask1) & (alongTrackDist > (AB_length + groundSpeed * 3))
        # we have passed point B by 3 seconds. Head towards B
        # Calc Nu to fly To WP B
        B_air = self.current_loc - next_WP
        B_air_unit = B_air / get_length(B_air) # Unit vector from WP B to aircraft
        xtrackVel = get_cross_error(self.ground_speed, -B_air_unit) # Velocity across line
        ltrackVel = get_vector_dot(self.ground_speed, -B_air_unit) # Velocity along line
        self.Nu += torch.atan2(xtrackVel, ltrackVel) * mask2
        # bearing (radians) from AC to L1 point
        self.nav_bearing += torch.atan2(-B_air_unit[:, 1], -B_air_unit[:, 0]).reshape(-1, 1) * mask2
        mask3 = ~(mask1 | mask2)
        # Calc Nu to fly along AB line
        # Calculate Nu2 angle (angle of velocity vector relative to line connecting waypoints)
        xtrackVel = get_cross_error(self.ground_speed, AB) # Velocity cross track
        ltrackVel = get_vector_dot(self.ground_speed, AB) # Velocity along track
        Nu2 = torch.atan2(xtrackVel,ltrackVel)
        # Calculate Nu1 angle (Angle to L1 reference point)
        sine_Nu1 = self.crosstrack_error / torch.max(self.L1_dist, 0.1 * torch.ones((self.n, 1), device=self.device))
        # Limit sine of Nu1 to provide a controlled track capture angle of 45 deg
        sine_Nu1 = torch.clamp(sine_Nu1, -0.7071, 0.7071)
        Nu1 = torch.asin(sine_Nu1)
        # compute integral error component to converge to a crosstrack of zero when traveling
        # straight but reset it when disabled or if it changes. That allows for much easier
        # tuning by having it re-converge each time it changes.
        mask1 = torch.abs(Nu1) < (5 * torch.pi / 180)
        self.L1_xtrack_i += Nu1 * self.L1_xtrack_i_gain * self.dt * mask1
        self.L1_xtrack_i = torch.clamp(self.L1_xtrack_i, -0.1, 0.1)
        # to converge to zero we must push Nu1 harder
        Nu1 += self.L1_xtrack_i
        self.Nu += (Nu1 + Nu2) * mask3
        # bearing (radians) from AC to L1 point
        self.nav_bearing += wrap_PI(torch.atan2(AB[:, 1], AB[:, 0]).reshape(-1, 1) + Nu1) * mask3
        self.prevent_indecision(state)
        self.last_Nu = self.Nu
        # Limit Nu to +-(pi/2)
        self.Nu = torch.clamp(self.Nu, -torch.pi / 2, torch.pi / 2)
        self.latAccDem = K_L1 * groundSpeed * groundSpeed / self.L1_dist * torch.sin(self.Nu)
        # Waypoint capture status is always false during waypoint following
        self.WPcircle = torch.zeros((self.n, 1), dtype=torch.bool, device=self.device)
        self.bearing_error = self.Nu # bearing error angle (radians), +ve to left of track
    
    # update L1 control for loitering
    def update_loiter(self, center_WP, radius, loiter_direction, env, TAS_dem):
        # scale loiter radius with square of EAS2TAS to allow us to stay stable at high altitude
        # radius = self.loiter_radius(radius, eas2tas, TAS_dem)
        # Calculate guidance gains used by PD loop (used during circle tracking)
        npos, epos, altitude = env.model.get_position()
        vx, vy = env.model.get_ground_speed()
        roll, pitch, yaw = env.model.get_posture()
        omega = (2 * torch.pi / self.L1_period)
        Kx = omega * omega
        Kv = 2 * self.L1_damping * omega
        # Calculate L1 gain required for specified damping (used during waypoint capture)
        K_L1 = 4 * self.L1_damping * self.L1_damping
        # Get current position and velocity
        self.current_loc = torch.hstack((npos.reshape(-1, 1), epos.reshape(-1, 1)))
        self.ground_speed = torch.hstack((vx.reshape(-1, 1), vy.reshape(-1, 1)))
        groundSpeed = get_length(self.ground_speed)
        # update target_bearing
        self.target_bearing = get_diff_angle(self.current_loc, center_WP)
        # Calculate time varying control parameters
        # Calculate the L1 length required for specified period
        # 0.3183099 = 1/pi
        self.L1_dist = self.L1_damping * self.L1_period * groundSpeed / torch.pi
        # Calculate the NE position of the aircraft relative to WP A
        A_air = self.current_loc - center_WP
        # Calculate the unit vector from WP A to aircraft protect against being on the waypoint and having zero velocity
        # if too close to the waypoint, use the velocity vector if the velocity vector is too small, use the heading vector
        mask1 = get_length(A_air) > 0.1
        A_air_unit = A_air / get_length(A_air) * mask1
        yaw = yaw.reshape(-1, 1)
        mask2 = (~mask1) & (groundSpeed < 0.1)
        A_air_unit += torch.hstack((torch.cos(yaw), torch.sin(yaw))) * mask2
        mask3 = ~(mask1 | mask2)
        A_air_unit += self.ground_speed / groundSpeed * mask3
        # Calculate Nu to capture center_WP
        xtrackVelCap = get_cross_error(A_air_unit, self.ground_speed) # Velocity across line - perpendicular to radial inbound to WP
        ltrackVelCap = -get_vector_dot(self.ground_speed, A_air_unit) # Velocity along line - radial inbound to WP
        self.Nu = torch.atan2(xtrackVelCap, ltrackVelCap)
        self.prevent_indecision(yaw)
        self.last_Nu = self.Nu
        self.Nu = torch.clamp(self.Nu, -torch.pi / 2, torch.pi / 2) # Limit Nu to +- Pi/2
        # Calculate lat accln demand to capture center_WP (use L1 guidance law)
        latAccDemCap = K_L1 * groundSpeed * groundSpeed / self.L1_dist * torch.sin(self.Nu)
        # Calculate radial position and velocity errors
        xtrackVelCirc = -ltrackVelCap # Radial outbound velocity - reuse previous radial inbound velocity
        xtrackErrCirc = get_length(A_air) - radius # Radial distance from the loiter circle
        # keep crosstrack error for reporting
        self.crosstrack_error = xtrackErrCirc
        # Calculate PD control correction to circle waypoint_ahrs.roll
        latAccDemCircPD = (xtrackErrCirc * Kx + xtrackVelCirc * Kv)
        # Calculate tangential velocity
        velTangent = xtrackVelCap * loiter_direction
        # Prevent PD demand from turning the wrong way by limiting the command when flying the wrong way
        mask1 = (ltrackVelCap < 0) & (velTangent < 0)
        latAccDemCircPD =  torch.max(latAccDemCircPD, torch.zeros((self.n, 1), device=self.device)) * mask1 + latAccDemCircPD * ~mask1
        # Calculate centripetal acceleration demand
        latAccDemCircCtr = velTangent * velTangent / torch.max((0.5 * radius), (radius + xtrackErrCirc))
        # Sum PD control and centripetal acceleration to calculate lateral manoeuvre demand
        latAccDemCirc = loiter_direction * (latAccDemCircPD + latAccDemCircCtr)
        # Perform switchover between 'capture' and 'circle' modes at the
        # point where the commands cross over to achieve a seamless transfer
        # Only fly 'capture' mode if outside the circle
        mask1 = (xtrackErrCirc > 0) & ((loiter_direction * latAccDemCap) < (loiter_direction * latAccDemCirc))
        self.latAccDem = latAccDemCap * mask1
        self.WPcircle = ~mask1
        self.bearing_error = self.Nu * mask1 # angle between demanded and achieved velocity vector, +ve to left of track
        # bearing (radians) from AC to L1 point
        self.nav_bearing = torch.atan2(-A_air_unit[:, 1], -A_air_unit[:, 0]).reshape(-1, 1)
        self.latAccDem += latAccDemCirc * ~mask1

    # update L1 control for heading hold navigation
    def update_heading_hold(self, navigation_heading, env):
        # Calculate normalised frequency for tracking loop
        vx, vy = env.model.get_ground_speed()
        roll, pitch, yaw = env.model.get_posture()
        omegaA = 4.4428 / self.L1_period # sqrt(2)*pi/period
        # copy to _target_bearing_cd and _nav_bearing
        self.target_bearing = wrap_PI(navigation_heading)
        self.nav_bearing = navigation_heading
        yaw = yaw.reshape(-1, 1)
        self.Nu = wrap_PI(self.target_bearing - wrap_PI(yaw))
        self.ground_speed = torch.hstack((vx.reshape(-1, 1), vy.reshape(-1, 1)))
        # Calculate groundspeed
        groundSpeed = get_length(self.ground_speed)
        # Calculate time varying control parameters
        self.L1_dist = groundSpeed / omegaA # L1 distance is adjusted to maintain a constant tracking loop frequency
        VomegaA = groundSpeed * omegaA
        # Waypoint capture status is always false during heading hold
        self.WPcircle = torch.zeros((self.n, 1), dtype=torch.bool, device=self.device)
        self.crosstrack_error = torch.zeros((self.n, 1), device=self.device)
        self.bearing_error = self.Nu # bearing error angle (radians), +ve to left of track
        # Limit Nu to +-pi
        self.Nu = torch.clamp(self.Nu, -torch.pi / 2, torch.pi / 2)
        self.latAccDem = 2 * torch.sin(self.Nu) * VomegaA
    
    # update L1 control for level flight on current heading
    def update_level_flight(self, yaw):
        # copy to target_bearing and nav_bearing
        yaw = yaw.reshape(-1, 1)
        self.target_bearing = yaw
        self.nav_bearing = yaw
        self.bearing_error = torch.zeros((self.n, 1), device=self.device)
        self.crosstrack_error = torch.zeros((self.n, 1), device=self.device)
        # Waypoint capture status is always false during heading hold
        self.WPcircle = torch.zeros((self.n, 1), dtype=torch.bool, device=self.device)
        self.latAccDem = torch.zeros((self.n, 1), device=self.device)
    
    # return the bank angle needed to achieve tracking from the last update_*() operation
    def nav_roll(self, pitch):
        pitch = pitch.reshape(-1, 1)
        result = torch.cos(pitch) * torch.atan(self.latAccDem / self.gravity)
        result = torch.clamp(result, -torch.pi / 2, torch.pi / 2)
        return result
