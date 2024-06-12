import os
import sys
import numpy as np
import torch
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv
from envs.planning_env import PlanningEnv
from envs.env_wrappers import GPUVecEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

CURRENT_WORK_PATH = os.getcwd()

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = False
    
def _t2n(x):
    return x.detach().cpu().numpy()

episode_rewards = 0
ego_run_dir = CURRENT_WORK_PATH + "/../scripts/runs/2024-05-15_16-09-20_Control_heading_ppo_v1/episode_149"
device = "cuda:0"
config = "heading"

env = ControlEnv(num_envs=1, config=config, model='F16', random_seed=5, device=device)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device(device))
ego_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_latest.pt"))

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
if config == 'heading':
    target_altitude_buf = np.mean(_t2n(env.task.target_altitude))
    target_heading_buf = np.mean(_t2n(env.task.target_heading))
    target_vt_buf = np.mean(_t2n(env.task.target_vt))
elif config == 'control':
    target_pitch_buf = np.mean(_t2n(env.task.target_pitch))
    target_heading_buf = np.mean(_t2n(env.task.target_heading))
    target_vt_buf = np.mean(_t2n(env.task.target_vt))
elif config == 'tracking':
    target_npos_buf = np.mean(_t2n(env.task.target_npos))
    target_epos_buf = np.mean(_t2n(env.task.target_epos))
    target_altitude_buf = np.mean(_t2n(env.task.target_altitude))

counts = 0
env.render(count=counts)
ego_rnn_states = torch.zeros((1, 1, 128), device=torch.device(device))
masks = torch.ones((1, 1), device=torch.device(device))
start = time.time()
unreach_target = 0
reset_target = 0
while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    # print(ego_actions)
    # Obser reward and next obs
    ego_obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(ego_actions, render=True, count=counts)
    unreach_target += int(_t2n(bad_dones))
    reset_target += int(_t2n(dones))

    npos, epos, altitude = env.model.get_position()
    npos_buf = np.hstack((npos_buf, np.mean(_t2n(npos))))
    epos_buf = np.hstack((epos_buf, np.mean(_t2n(epos))))
    altitude_buf = np.hstack((altitude_buf, np.mean(_t2n(altitude))))

    roll, pitch, yaw = env.model.get_posture()
    roll_buf = np.hstack((roll_buf, np.mean(_t2n(roll))))
    pitch_buf = np.hstack((pitch_buf, np.mean(_t2n(pitch))))
    yaw_buf = np.hstack((yaw_buf, np.mean(_t2n(yaw))))

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

    if config == 'heading':
        target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))
        target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
        target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))
    elif config == 'control':
        target_pitch_buf = np.hstack((target_pitch_buf, np.mean(_t2n(env.task.target_pitch))))
        target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
        target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))
    elif config == 'tracking':
        target_npos_buf = np.hstack((target_npos_buf, np.mean(_t2n(env.task.target_npos))))
        target_epos_buf = np.hstack((target_epos_buf, np.mean(_t2n(env.task.target_epos))))
        target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))

    counts += 1
    print(counts, _t2n(rewards))
    episode_rewards += _t2n(rewards)
    if counts >= 10000:
        break
# save result
np.save('./result/npos.npy', npos_buf)
np.save('./result/epos.npy', epos_buf)
np.save('./result/altitude.npy', altitude_buf)
np.save('./result/roll.npy', roll_buf)
np.save('./result/pitch.npy', pitch_buf)
np.save('./result/yaw.npy', yaw_buf)
np.save('./result/vt.npy', vt_buf)
np.save('./result/alpha.npy', alpha_buf)
np.save('./result/beta.npy', beta_buf)
np.save('./result/G.npy', G_buf)

np.save('./result/T.npy', T_buf)
np.save('./result/throttle.npy', throttle_buf)
np.save('./result/ail.npy', ail_buf)
np.save('./result/el.npy', el_buf)
np.save('./result/rud.npy', rud_buf)

if config == 'heading':
    np.save('./result/target_altitude.npy', target_altitude_buf)
    np.save('./result/target_heading.npy', target_heading_buf)
    np.save('./result/target_vt.npy', target_vt_buf)
elif config == 'control':
    np.save('./result/target_pitch.npy', target_pitch_buf)
    np.save('./result/target_heading.npy', target_heading_buf)
    np.save('./result/target_vt.npy', target_vt_buf)
elif config == 'tracking':
    np.save('./result/target_npos.npy', target_npos_buf)
    np.save('./result/target_epos.npy', target_epos_buf)
    np.save('./result/target_altitude.npy', target_altitude_buf)
end = time.time()
print('total time:', end - start)
print('episode reward:', episode_rewards)
print('average episode reward:', episode_rewards / (unreach_target + reset_target))
print('unreach target:', unreach_target)
print('reset target:', reset_target)
print('success rate:', reset_target / (reset_target + unreach_target))