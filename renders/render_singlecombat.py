import os
import sys
import numpy as np
import torch
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.singlecombat_env import SingleCombatEnv
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
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = True
ego_policy_index = 199
enm_policy_index = 74
episode_rewards = 0
ego_run_dir = CURRENT_WORK_PATH + "/scripts/runs/2024-02-20_19-24-11_SingleCombat_selfplay_ppo_v1"
enm_run_dir = CURRENT_WORK_PATH + "/scripts/runs/2024-02-20_19-24-11_SingleCombat_selfplay_ppo_v1"
experiment_name = ego_run_dir.split('/')[-4]
device = "cuda:0"

env = SingleCombatEnv(num_envs=1, config="selfplay", random_seed=0, device=device)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device(device))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device(device))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))


print("Start render")
obs = env.reset()
counts = 0
env.render(count=counts, filepath=f'{experiment_name}.txt.acmi')
ego_rnn_states = torch.zeros((1, 1, 128), device=torch.device(device))
masks = torch.ones((num_agents // 2, 1), device=torch.device(device))
enm_obs =  obs[num_agents // 2:, :]
ego_obs =  obs[:num_agents // 2, :]
enm_rnn_states = torch.zeros_like(ego_rnn_states)
start = time.time()
fail = 0
success = 0
while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    actions = torch.vstack((ego_actions, enm_actions))
    # Obser reward and next obs
    ego_obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(ego_actions)
    rewards = rewards[:num_agents // 2, ...]
    fail += int(_t2n(torch.any(bad_dones + exceed_time_limits)))
    success += int(_t2n(torch.any(dones)))
    counts += 1
    print(_t2n(env.step_count[0]), _t2n(rewards))
    print(_t2n(env.step_count[0]), 'ego_blood:', _t2n(env.blood[0]))
    print(_t2n(env.step_count[0]), 'enm_blood:', _t2n(env.blood[1]))
    env.render(count=counts)
    if counts >= 2000:
        break
    episode_rewards += _t2n(rewards)
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

end = time.time()
print('total time:', end - start)
print('episode reward:', episode_rewards)
print('average episode reward:', episode_rewards / (fail + success))
print('fail:', fail)
print('success:', success)
