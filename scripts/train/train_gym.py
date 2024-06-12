#!/usr/bin/env python
import sys
import os
import gym
import datetime
import torch
import random
import numpy as np
from pathlib import Path
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from envs.env_wrappers import GPUVecEnv
from runner.F16sim_runner import F16SimRunner as Runner
from envs.control_env import ControlEnv
import torch.utils.tensorboard as tb


class GymEnv:
    def __init__(self, env):
        self.env = env
        self.action_shape = self.env.action_space.shape
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        observation = self.env.reset()
        return np.array(observation).reshape((1, -1))

    def step(self, action):
        action = np.array(action).reshape(self.action_shape)
        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation).reshape((1, -1))
        done = np.array(done).reshape((1,-1))
        reward = np.array(reward).reshape((1, -1))
        return observation, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    

class GymHybridEnv(GymEnv):
    def __init__(self, env) -> None:
        self.env = env
        self.action_space = self.env.action_space
        self.discrete_dims = self.action_space[0].shape[0]
        self.continuous_dims = self.action_space[1].shape[0]
        self.action_shape = (self.discrete_dims+self.continuous_dims,)
        self.observation_space = self.env.observation_space
    
    def reset(self):
        observation = self.env.reset()
        return np.array(observation).reshape((1, -1))

    def step(self, action):
        action = np.array(action).reshape(self.action_shape)
        discrete_a, continuous_a = action[:self.discrete_dims].astype(np.int32), action[self.discrete_dims:]
        action = (discrete_a, continuous_a)
        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation).reshape((1, -1))
        done = np.array(done).reshape((1,-1))
        reward = np.array(reward).reshape((1, -1))
        return observation, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        pass


def make_train_env(all_args):
    def get_env_fn():
        def init_env():
            if all_args.env_name == "Control":
                env = ControlEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, random_seed=all_args.seed)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    return GPUVecEnv([get_env_fn()])

def make_eval_env(all_args):
    def get_env_fn():
        def init_env():
            if all_args.env_name == "Control":
                env = ControlEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, random_seed=all_args.seed * 50000)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    return GPUVecEnv([get_env_fn()])


def parse_args(args, parser):
    group = parser.add_argument_group("Gym Env parameters")
    group.add_argument('--scenario-name', type=str, default='CartPole-v1',
                       help="the name of gym env")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="the max length of an episode")
    group.add_argument('--num-agents', type=int, default=1,
                       help="number of agents controlled by RL policy")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/runs/{}_{}_{}_{}_{}'.
                   format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), all_args.env_name, all_args.scenario_name, all_args.algorithm_name, all_args.experiment_name))
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # tensorboard
    writer = tb.SummaryWriter(run_dir)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = Runner(config, writer)
    runner.run()

    # post process
    envs.close()
    writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
