import os
import sys
import numpy as np
import torch
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv
from envs.env_wrappers import GPUVecEnv
from algorithms.pid.controller import Controller
import logging
logging.basicConfig(level=logging.DEBUG)


def _t2n(x):
    return x.detach().cpu().numpy()

episode_rewards = 0
device = "cuda:0"

env = ControlEnv(num_envs=1, config="heading", model='F16', random_seed=0, device=device)

controller = Controller(dt=env.model.dt, n=env.n, device=device)

print("Start render")
ego_obs = env.reset()
# 状态量
npos, epos, altitude = env.model.get_position()
npos_buf = np.mean(_t2n(npos))
epos_buf = np.mean(_t2n(epos))
altitude_buf = np.mean(_t2n(altitude))

roll, pitch, yaw = env.model.get_posture()
roll_buf = np.mean(_t2n(roll))
pitch_buf = np.mean(_t2n(pitch))
yaw_buf = np.mean(_t2n(yaw))

yaw_rate = env.model.get_extended_state()[:, 5]
yaw_rate_buf = np.mean(_t2n(yaw_rate))

roll_dem_buf = np.array(0)
pitch_dem_buf = np.array(0)
yaw_rate_dem_buf = np.array(0)

vt = env.model.get_vt()
vt_buf = np.mean(_t2n(vt))

alpha = env.model.get_AOA()
alpha_buf = np.mean(_t2n(alpha))

beta = env.model.get_AOS()
beta_buf = np.mean(_t2n(beta))

G = env.model.get_G()
G_buf = np.mean(_t2n(G))
# 控制量
T = env.model.get_thrust()
T_buf = np.mean(_t2n(T))
throttle_buf = np.mean(_t2n(T * 0.3048 / 82339 / 0.225))

el, ail, rud, lef = env.model.get_control_surface()
el_buf = np.mean(_t2n(el))
ail_buf = np.mean(_t2n(ail))
rud_buf = np.mean(_t2n(rud))
# 目标
target_altitude_buf = np.mean(_t2n(env.task.target_altitude))
target_heading_buf = np.mean(_t2n(env.task.target_heading))
target_vt_buf = np.mean(_t2n(env.task.target_vt))

counts = 0
start = time.time()

while True:
    state = env.model.get_state()
    estate = env.model.get_extended_state()
    ax, ay, az = env.model.get_acceleration()
    acceleration = torch.hstack((ax.reshape(-1, 1), ay.reshape(-1, 1)))
    acceleration = torch.hstack((acceleration, az.reshape(-1, 1)))
    EAS2TAS = env.model.get_EAS2TAS()
    hgt_dem = env.task.target_altitude.reshape(-1, 1)
    TAS_dem = env.task.target_vt.reshape(-1, 1)
    navigation_heading = env.task.target_heading.reshape(-1, 1)
    if counts % 5 == 0:
        controller.cal_pitch_throttle(hgt_dem, TAS_dem, env)
        if counts < 2000:
            controller.update_level_flight(env)
        else:
            controller.update_heading_hold(navigation_heading, env)
    controller.stabilize(env)
    # Obser reward and next obs
    ego_actions = controller.get_action()
    ego_obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(ego_actions, render=True, count=counts)

    npos, epos, altitude = env.model.get_position()
    npos_buf = np.hstack((npos_buf, np.mean(_t2n(npos))))
    epos_buf = np.hstack((epos_buf, np.mean(_t2n(epos))))
    altitude_buf = np.hstack((altitude_buf, np.mean(_t2n(altitude))))

    roll, pitch, yaw = env.model.get_posture()
    roll_buf = np.hstack((roll_buf, np.mean(_t2n(roll))))
    pitch_buf = np.hstack((pitch_buf, np.mean(_t2n(pitch))))
    yaw_buf = np.hstack((yaw_buf, np.mean(_t2n(yaw))))

    yaw_rate = env.model.get_extended_state()[:, 5]
    yaw_rate_buf = np.hstack((yaw_rate_buf, np.mean(_t2n(yaw_rate))))

    roll_dem_buf = np.hstack((roll_dem_buf, np.mean(_t2n(controller.roll_dem))))
    pitch_dem_buf = np.hstack((pitch_dem_buf, np.mean(_t2n(controller.pitch_dem))))
    yaw_rate_dem_buf = np.hstack((yaw_rate_dem_buf, np.mean(_t2n(controller.yaw_rate_dem))))

    vt = env.model.get_vt()
    vt_buf = np.hstack((vt_buf, np.mean(_t2n(vt))))

    alpha = env.model.get_AOA()
    alpha_buf = np.hstack((alpha_buf, np.mean(_t2n(alpha))))

    beta = env.model.get_AOS()
    beta_buf = np.hstack((beta_buf, np.mean(_t2n(beta))))

    G = env.model.get_G()
    G_buf = np.hstack((G_buf, np.mean(_t2n(G))))

    T = env.model.get_thrust()
    T_buf = np.hstack((T_buf, np.mean(_t2n(T))))
    throttle_buf = np.hstack((throttle_buf, np.mean(_t2n(T * 0.3048 / 82339 / 0.225))))

    el, ail, rud, lef = env.model.get_control_surface()
    el_buf = np.hstack((el_buf, np.mean(_t2n(el))))
    ail_buf = np.hstack((ail_buf, np.mean(_t2n(ail))))
    rud_buf = np.hstack((rud_buf, np.mean(_t2n(rud))))

    target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))
    target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
    target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))

    counts += 1
    print(counts, _t2n(rewards))
    episode_rewards += _t2n(rewards)
    if torch.any(dones):
        break
# save result
np.save('./result/npos.npy', npos_buf)
np.save('./result/epos.npy', epos_buf)
np.save('./result/altitude.npy', altitude_buf)
np.save('./result/roll.npy', roll_buf)
np.save('./result/pitch.npy', pitch_buf)
np.save('./result/yaw.npy', yaw_buf)
np.save('./result/yaw_rate.npy', yaw_rate_buf)
np.save('./result/roll_dem.npy', roll_dem_buf)
np.save('./result/pitch_dem.npy', pitch_dem_buf)
np.save('./result/yaw_rate_dem.npy', yaw_rate_dem_buf)
np.save('./result/vt.npy', vt_buf)
np.save('./result/alpha.npy', alpha_buf)
np.save('./result/beta.npy', beta_buf)
np.save('./result/G.npy', G_buf)

np.save('./result/T.npy', T_buf)
np.save('./result/throttle.npy', throttle_buf)
np.save('./result/ail.npy', ail_buf)
np.save('./result/el.npy', el_buf)
np.save('./result/rud.npy', rud_buf)

np.save('./result/target_altitude.npy', target_altitude_buf)
np.save('./result/target_heading.npy', target_heading_buf)
np.save('./result/target_vt.npy', target_vt_buf)
end = time.time()
print('total time:', end - start)
print('episode reward:', episode_rewards)
