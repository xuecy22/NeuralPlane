#!/usr/bin/env python
import sys
import os
import traceback
import datetime
import torch
import random
import logging
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from runner.F16sim_runner import F16SimRunner
from runner.selfplay_F16sim_runner import SelfplayJSBSimRunner
from envs.control_env import ControlEnv
from envs.planning_env import PlanningEnv
from envs.singlecombat_env import SingleCombatEnv
from envs.env_wrappers import GPUVecEnv
import torch.utils.tensorboard as tb

CURRENT_WORK_PATH = os.getcwd()

def make_train_env(all_args):
    def get_env_fn():
        def init_env():
            if all_args.env_name == "Control":
                env = ControlEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, model= all_args.model_name, random_seed=all_args.seed, device=all_args.device)
            elif all_args.env_name == "Planning":
                env = PlanningEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, model= all_args.model_name, random_seed=all_args.seed, device=all_args.device)
            elif all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, random_seed=all_args.seed, device=all_args.device)
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
                env = ControlEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, model= all_args.model_name, random_seed=all_args.seed, device=all_args.device)
            elif all_args.env_name == "Planning":
                env = PlanningEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, model= all_args.model_name, random_seed=all_args.seed, device=all_args.device)
            elif all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(num_envs=all_args.n_eval_rollout_threads, config=all_args.scenario_name, random_seed=all_args.seed * 50000, device=all_args.device)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    return GPUVecEnv([get_env_fn()])


def parse_args(args, parser):
    group = parser.add_argument_group("F16Sim Env parameters")
    group.add_argument("--env-name", type=str, default='Control',
                       help="specify the name of environment")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--model-name', type=str, default='F16',
                       help="Which model to run on")
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
        device = torch.device(all_args.device)  # use cude mask to control using which GPU
        # torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        # torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/runs/{}_{}_{}_{}_{}_{}'.
                   format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), all_args.env_name, all_args.scenario_name, all_args.model_name, all_args.algorithm_name, all_args.experiment_name))
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
    if all_args.use_selfplay:
        runner = SelfplayJSBSimRunner(config, writer)
    else:
        runner = F16SimRunner(config, writer)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()
        writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
