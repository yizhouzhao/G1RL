# define the locomotion task
import torch

class LocomotionTask:
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
    
    def get_observation(self):
        pass
    
    def get_reward(self):
        pass
    
    def is_done(self):
        pass

    @property
    def default_base_env_path(self):
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        return f"{self.default_base_env_path}/env_0"



class G1LocomotionTask(LocomotionTask):
    def __init__(self):
        super().__init__()
        self._g1_postions = [0, 0, 0.8]
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
    
    def get_observation(self):
        pass
    
    def get_reward(self):
        pass
    
    def is_done(self):
        pass

    def set_up_scene(self):
        self.get_humanoid()

    def get_humanoid(self):
        from .g1 import G1Robot
        g1 = G1Robot(prim_path=self.default_zero_env_path + "/g1", name="g1", translation=self._g1_postions)
        
        