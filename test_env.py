import numpy as np
from envs.control_env import ControlEnv
from envs.env_wrappers import GPUVecEnv
import logging
import time
import os

logging.basicConfig(level=logging.DEBUG)
CURRENT_WORK_PATH = os.getcwd()

def test_env():
    parallel_num = 1
    envs = GPUVecEnv([lambda: ControlEnv() for _ in range(parallel_num)])

    envs.reset()
    # DataType test

    episode_reward = 0
    step = 0
    while True:
        actions = np.array([[envs.action_space.sample() for _ in range(envs.agents)] for _ in range(parallel_num)])
        _, rewards, dones, _, _, _ = envs.step(actions)
        episode_reward += rewards[:,0,:]
        step += 1
        print(f"step:{step}, avg_reward:{episode_reward / step}")
        # terminate if any of the parallel envs has been done
        if np.any(dones):
            print(episode_reward)
            break
    envs.close()

if __name__ == "__main__":
    test_env()
