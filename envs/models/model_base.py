import torch
import numpy as np
import random
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base Model class
    Model-specific functions are implemented in subclasses
    """
    def __init__(self, config, n, device, random_seed):
        self.config = config
        self.n = n
        self.device = device
        if random_seed is not None:
            self.seed(random_seed)

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    @abstractmethod
    def reset(self, env):
        """Perform model function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_extended_state(self):
        """Compute the extended state at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): extended state
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, t, action):
        """Compute the next state at the current timestep.
        Overwritten by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self):
        """Compute the state at the current timestep.
        Overwritten by subclasses.
        
        Returns:
            (tensor): state
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_control(self):
        """Compute the control at the current timestep.
        Overwritten by subclasses.
        
        Returns:
            (tensor): control
        """
        raise NotImplementedError

    @abstractmethod
    def get_position(self):
        """Compute the position at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): npos, epos, altitude
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_ground_speed(self):
        """Compute the ground speed at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): ground speed
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_climb_rate(self):
        """Compute the climb rate at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): climb rate
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_posture(self):
        """Compute the posture at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): roll, pitch, yaw
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_euler_angular_velocity(self):
        """Compute the euler angular velocity at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): euler angular velocity
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_vt(self):
        """Compute the vt at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): vt
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_TAS(self):
        """Compute the TAS at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): TAS
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_AOA(self):
        """Compute the angle of attack(AOA) at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): AOA
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_AOS(self):
        """Compute the angle of sideslip(AOS) at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): AOS
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_angular_velocity(self):
        """Compute the angular velocity at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): angular velocity
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_thrust(self):
        """Compute the thrust at the current timestep.
        Overwritten by subclasses.
        
        Returns:
            (tensor): thrust
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_control_surface(self):
        """Compute the control surface at the current timestep.
        Overwritten by subclasses.
        
        Returns:
            (tensor): control surface
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_velocity(self):
        """Compute the velocity at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): velocity
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_acceleration(self):
        """Compute the acceleration at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): acceleration
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_G(self):
        """Compute the G at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): G
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_EAS2TAS(self):
        """Compute the EAS2TAS at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): EAS2TAS
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_accels(self):
        """Compute the accels at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): accels
        """
        raise NotImplementedError
    
    def get_atmos(self):
        """Compute the atmos at the current timestep.
        Overwritten by subclasses.

        Returns:
            (tensor): atmos
        """
        raise NotImplementedError
    