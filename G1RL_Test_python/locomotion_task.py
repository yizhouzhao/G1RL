# define the locomotion task
import torch
import numpy as np

from pxr import Gf, PhysicsSchemaTools
import omni.usd

from isaacsim.core.cloner import Cloner, GridCloner
import isaacsim.core.utils.xforms as xform_utils 
from isaacsim.core.prims import Articulation

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
        self.articulation = None
        self.initialized = False
        
        # action scale and offset
        self.action_scale = 0.5
        self.action_offset = torch.tensor([-0.1000, -0.1000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3000, 0.3000, -0.2000, -0.2000, 0.0000, 0.0000], device='cuda:0')
        self.joint_ids = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18]
        self.joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint']


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


    def initialize(self):
        self.articulation = Articulation(
            prim_paths_expr="/World/envs/.*/g1", name="humanoid_view", reset_xform_properties=False
        )
        self.articulation.initialize()

        print("[G1LocomotionTask] num_dof", self.articulation.num_dof)
        print("[G1LocomotionTask] num_bodies", self.articulation.num_bodies)
        print("[G1LocomotionTask] num_joints", self.articulation.num_joints)
        print("[G1LocomotionTask] dof_names", self.articulation.dof_names)

        # get_joint_positions
        joint_positions = self.articulation.get_joint_positions()
        print("[G1LocomotionTask] joint_positions", joint_positions)

        # get_joint_velocities
        joint_velocities = self.articulation.get_joint_velocities()
        print("[G1LocomotionTask] joint_velocities", joint_velocities)

        # set default joint positions
        default_joint_positions = torch.zeros((self._num_envs, self.articulation.num_dof), device='cuda:0')
        
        # apply action_offset according to joint_ids
        default_joint_positions[:, self.joint_ids] = self.action_offset        
        default_velocities=np.zeros((self._num_envs, self.articulation.num_dof))
        default_efforts = np.zeros((self._num_envs, self.articulation.num_dof))
        self.articulation.set_joints_default_state(
            positions = default_joint_positions.cpu().data, 
            velocities = default_velocities, 
            efforts = default_efforts
        )


        self.initialized = True

    def reset(self):
        if not self.initialized:
            self.initialize()

        self.articulation.post_reset()
        
    def get_observation(self):
        joint_positions = self.articulation.get_joint_positions(joint_indices=self.joint_ids)
        joint_velocities = self.articulation.get_joint_velocities(joint_indices=self.joint_ids)
        positions, orientations = self.articulation.get_local_poses()

        obs = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "positions": positions,
            "orientations": orientations
        }

        print("[G1LocomotionTask] obs", obs)
        return obs
        