# define the locomotion task
import torch

from pxr import Gf, PhysicsSchemaTools
import omni.usd

from isaacsim.core.cloner import Cloner, GridCloner
import isaacsim.core.utils.xforms as xform_utils 

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
        self._g1_default_height = 0.8
        self._num_envs = 16
        self._env_spacing = 4.0
    
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
        stage = omni.usd.get_context().get_stage()
        self.add_ground()
        self.get_humanoid()

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)

        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=True, copy_from_source=False
        )

        print("self._env_pos: ", self._env_pos)
        for prim_path, pos in zip(prim_paths, self._env_pos):
            print("prim_path: ", prim_path)
            print("pos: ", pos)
            translation = Gf.Vec3d(pos[0], pos[1], self._g1_default_height)
            orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)
            prim = stage.GetPrimAtPath(prim_path + "/g1")
            xform_utils.reset_and_set_xform_ops(prim, translation, orientation)

    def add_ground(self):
        PhysicsSchemaTools.addGroundPlane(
            omni.usd.get_context().get_stage(), "/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
        )

    def get_humanoid(self):
        from .g1 import G1Robot
        g1 = G1Robot(prim_path=self.default_zero_env_path + "/g1", name="g1")
        
        