from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from robopoly.agents.multi_agent import MultiAgent
from robopoly.agents.robots.panda import Panda
from robopoly.envs.sapien_env import BaseEnv
from robopoly.sensors.camera import CameraConfig
from robopoly.utils import sapien_utils
from robopoly.utils.building import actors
from robopoly.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from robopoly.utils.scene_builder.table import TableSceneBuilder
from robopoly.utils.structs.actor import Actor
from robopoly.utils.structs.pose import Pose
from robopoly.utils.structs.types import GPUMemoryConfig, SimConfig


class TwoRobotKitchenReplicaCADBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]
    agent: MultiAgent[Tuple[Panda, Panda]]

    replicacad_build_config_idx = 0
    workspace_offset = (1.4, -1.2, 0.9196429)
    # Optional per-task in-plane (x, y) nudge of each arm base from its default
    # mounting point (workspace_offset + (0, -/+0.68)). Defaults to no shift so every
    # existing task is unaffected; a subclass overrides these to move an arm closer to
    # its own workspace (see TwoRobotPutObjectCabinet, which nudges both bases toward
    # the cabinet so the left arm reaches the fully-pulled-out drawer handle cleanly).
    arm_base_shift_left = (0.0, 0.0)
    arm_base_shift_right = (0.0, 0.0)
    # Optional per-task yaw (rad) of each arm base. Defaults to the standard facing (left +pi/2,
    # right -pi/2) so every existing task is unaffected; a subclass can rotate a base to face its
    # workspace head-on (see TwoRobotPutObjectCabinet, which angles the left base toward the drawer
    # so the pull becomes a single forward reach instead of a sideways sweep).
    arm_base_yaw_left = np.pi / 2
    arm_base_yaw_right = -np.pi / 2
    # Native fixed-view resolution shared by every two-robot task.
    global_camera_width = 512
    global_camera_height = 384
    hidden_replicacad_name_substrings = (
        "frl_apartment_table",
        "_cabinet-0",
        "frl_apartment_rug_01",
        "frl_apartment_rug_02",
        "frl_apartment_bowl_07",
        "frl_apartment_kitchen_utensil_01",
        "frl_apartment_kitchen_utensil_05",
        "frl_apartment_lamp_02",
        "frl_apartment_pan_01",
        "frl_apartment_choppingboard_02",
    )

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _workspace_target(self):
        return np.array(self.workspace_offset) + np.array([0.0, 0.0, 0.10])

    @property
    def _default_sensor_configs(self):
        target = self._workspace_target
        offset = np.array(self.workspace_offset)
        left_pose = sapien_utils.look_at(
            eye=(offset + np.array([1.25, -1.45, 1.05])).tolist(),
            target=(target + np.array([0.0, -0.05, -0.02])).tolist(),
        )
        right_pose = sapien_utils.look_at(
            eye=(offset + np.array([1.25, 1.45, 1.05])).tolist(),
            target=(target + np.array([0.0, 0.05, -0.02])).tolist(),
        )
        top_pose = sapien_utils.look_at(
            eye=(offset + np.array([1.20, -1.00, 2.10])).tolist(),
            target=(target + np.array([0.0, 0.0, -0.06])).tolist(),
        )
        visualization_pose = sapien_utils.look_at(
            eye=(offset + np.array([1.55, 0.0, 0.60])).tolist(),
            target=(target + np.array([0.0, 0.0, -0.04])).tolist(),
        )
        global_pose = sapien_utils.look_at(
            eye=(offset + np.array([1.40, 0.0, 1.1643])).tolist(),
            target=(target + np.array([-0.12, 0.0, 0.0])).tolist(),
        )
        room_pose = sapien_utils.look_at(
            eye=(offset + np.array([2.2, -2.6, 1.8])).tolist(),
            target=target.tolist(),
        )
        return [
            CameraConfig("left_side_camera", left_pose, 256, 256, np.pi / 2, 0.01, 100),
            CameraConfig(
                "right_side_camera", right_pose, 256, 256, np.pi / 2, 0.01, 100
            ),
            CameraConfig("overview_camera", top_pose, 1024, 768, 0.92, 0.01, 100),
            CameraConfig("global_camera", global_pose, self.global_camera_width,
                         self.global_camera_height, 0.70, 0.01, 100),
            CameraConfig(
                "visualization_camera",
                visualization_pose,
                512,
                384,
                0.90,
                0.01,
                100,
            ),
            CameraConfig("room_camera", room_pose, 1024, 768, 0.95, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        target = self._workspace_target
        offset = np.array(self.workspace_offset)
        pose = sapien_utils.look_at(
            eye=(offset + np.array([2.2, -2.6, 1.8])).tolist(),
            target=target.tolist(),
        )
        return CameraConfig("render_camera", pose, 1024, 768, 0.95, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -0.68, 0]), sapien.Pose(p=[0, 0.68, 0])]
        )

    def _load_scene(self, options: dict):
        self.replicacad_scene = ReplicaCADSceneBuilder(self)
        self.replicacad_scene.build(self.replicacad_build_config_idx)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._build_plain_workspace_table()
        self._load_task_scene(options)

    def _load_task_scene(self, options: dict):
        raise NotImplementedError

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        self._reset_replicacad_background()
        self._hide_replicacad_workspace_furniture()
        self.table_scene.table.set_pose(sapien.Pose(p=[0, 0, -100]))
        self._set_plain_workspace_table_pose()
        self._set_default_robot_poses()
        self._initialize_task_episode(env_idx, options)

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def workspace_offset_tensor(self):
        return torch.tensor(self.workspace_offset, device=self.device)

    def _world_pos(self, rel_xyz):
        return self.workspace_offset_tensor + torch.tensor(rel_xyz, device=self.device)

    def _reset_replicacad_background(self):
        """Restore every ReplicaCAD background object (decor, furniture, dynamic
        clutter) to its built pose on each reset. The scene builder's own
        ``initialize`` assumes a single fetch robot and can't be used here, so we
        replicate just the object-restoration part. Without this, dynamic decor
        settles/drifts under physics and never gets reset, so the background grows
        progressively more chaotic across successive episodes.
        """
        from robopoly.utils.structs import Articulation

        for obj, pose in self.replicacad_scene._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

    def _hide_replicacad_workspace_furniture(self):
        hidden_pose = sapien.Pose(p=[0, 0, -100])
        for actor_name, actor in self.replicacad_scene.scene_objects.items():
            if any(s in actor_name for s in self.hidden_replicacad_name_substrings):
                actor.set_pose(hidden_pose)

    def _set_default_robot_poses(self):
        offset = np.array(self.workspace_offset)
        lshift = np.array([*self.arm_base_shift_left, 0.0])
        rshift = np.array([*self.arm_base_shift_right, 0.0])
        self.left_agent.robot.set_pose(
            sapien.Pose(
                p=(offset + np.array([0.0, -0.68, 0.0]) + lshift).tolist(),
                q=euler2quat(0, 0, self.arm_base_yaw_left),
            )
        )
        self.right_agent.robot.set_pose(
            sapien.Pose(
                p=(offset + np.array([0.0, 0.68, 0.0]) + rshift).tolist(),
                q=euler2quat(0, 0, self.arm_base_yaw_right),
            )
        )

    def _build_plain_workspace_table(self):
        self.plain_table_top = actors.build_box(
            self.scene,
            half_sizes=[0.75, 1.10, 0.035],
            color=[0.72, 0.36, 0.16, 1],
            name=f"{self.__class__.__name__}_plain_table_top",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, -100]),
        )
        self.plain_table_legs = []
        for i, (x, y) in enumerate(
            [(-0.68, -1.03), (-0.68, 1.03), (0.68, -1.03), (0.68, 1.03)]
        ):
            self.plain_table_legs.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[0.035, 0.035, 0.425],
                    color=[0.58, 0.30, 0.16, 1],
                    name=f"{self.__class__.__name__}_plain_table_leg_{i}",
                    body_type="kinematic",
                    initial_pose=sapien.Pose(p=[x, y, -100]),
                )
            )

    def _set_plain_workspace_table_pose(self):
        surface_z = self.workspace_offset[2]
        ox, oy, _ = self.workspace_offset
        self.plain_table_top.set_pose(sapien.Pose(p=[ox, oy, surface_z - 0.035]))
        for leg, (x, y) in zip(
            self.plain_table_legs,
            [(-0.68, -1.03), (-0.68, 1.03), (0.68, -1.03), (0.68, 1.03)],
        ):
            leg.set_pose(sapien.Pose(p=[ox + x, oy + y, surface_z / 2 - 0.035]))

    def _set_actor_pose_rel(self, actor, rel_xyz, q=None):
        p = self._world_pos(rel_xyz).expand(self.num_envs, -1)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))

    def _build_kitchen_object(
        self,
        group,
        name,
        object_scale=1.0,
        seed=0,
        obj_registries=("objaverse", "aigen"),
        kinematic=False,
    ):
        from robopoly.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
            sample_kitchen_object,
        )
        from robopoly.utils.scene_builder.robocasa.objects.objects import MJCFObject

        try:
            kwargs, info = sample_kitchen_object(
                group,
                obj_registries=obj_registries,
                object_scale=object_scale,
                rng=np.random.default_rng(seed),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Could not find a RoboCasa {group} asset. Install/download the "
                "RoboCasa asset bundle, then recreate the environment."
            ) from exc

        actors_per_env = []
        spawn_z = None
        horizontal_radius = None
        for i in range(self.num_envs):
            obj = MJCFObject(self.scene, name=name, **kwargs)
            if spawn_z is None:
                spawn_z = float(-obj.bottom_offset[2] + 0.005)
                horizontal_radius = float(obj.horizontal_radius)
            body_type = "kinematic" if kinematic else "dynamic"
            actor = obj.build(scene_idxs=[i], body_type=body_type).actor
            actors_per_env.append(actor)
            self.remove_from_state_dict_registry(actor)

        merged_actor = Actor.merge(actors_per_env, name=name)
        self.add_to_state_dict_registry(merged_actor)
        return merged_actor, spawn_z, horizontal_radius, info

    def _inside_xy(self, obj, center_actor, half_x, half_y):
        delta_xy = torch.abs(obj.pose.p[:, :2] - center_actor.pose.p[:, :2])
        return (delta_xy[:, 0] < half_x) & (delta_xy[:, 1] < half_y)

    def _both_static(self):
        return self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

    def _get_obs_extra(self, info: dict):
        return dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].float()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
