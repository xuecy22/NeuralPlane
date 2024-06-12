import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from env_base import BaseEnv
from models.F16_model import F16Model
from models.UAV_model import UAVModel
from tasks.heading_task import HeadingTask
from tasks.control_task import ControlTask
from tasks.tracking_task import TrackingTask


class ControlEnv(BaseEnv):
    """
    ControlEnv is a fly-control env for single agent to do tracking task.
    """
    def __init__(self, num_envs=1, config='heading', model='F16', random_seed=None, device="cuda:0"):
        super().__init__(num_envs, config, model, random_seed, device)
    
    def load(self, random_seed, config, model):
        if random_seed is not None:
            self.seed(random_seed)
        if model == 'F16':
            self.model = F16Model(self.config, self.n, self.device, random_seed)
        elif model == 'UAV':
            self.model = UAVModel(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
        if config == 'heading':
            self.task = HeadingTask(self.config, self.n, self.device, random_seed)
        elif config == 'control':
            self.task = ControlTask(self.config, self.n, self.device, random_seed)
        elif config == 'tracking':
            self.task = TrackingTask(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
    