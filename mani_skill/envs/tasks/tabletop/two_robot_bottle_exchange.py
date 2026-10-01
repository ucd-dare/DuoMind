from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose

from .two_robot_kitchen_base import TwoRobotKitchenReplicaCADBaseEnv


class TwoRobotBottleExchangeReplicaCADEnv(TwoRobotKitchenReplicaCADBaseEnv):
    """Internal exchange-scene base retained for Exchange Bread."""
    global_camera_width = 512
    global_camera_height = 384

    mug_model_id = 4
    # Keep the exchanged cups physically identical to Food Serve's cup.
    mug_scale = 0.050
    mug_mass = 0.003
    mug_radius = 0.048
    mug_height = 0.067
    object_yaw_range = 0.0
    mug_yaw_offset = np.pi / 2
    object_friction = 10.0
    gripper_friction = 10.0
    box_friction = 10.0

    box_half_size_xy = (0.18, 0.16)
    box_wall_height = 0.13
    box_wall_thickness = 0.015
    box_bottom_center_z = 0.015
    box_success_height = 0.18

    middle_stage_rel_xy = (0.0, 0.0)
    middle_half_size_xy = (0.12, 0.12)
    middle_stage_height = 0.10
    middle_stage_speed = 0.12

    left_mug_x_range = (-0.28, -0.24)
    right_mug_x_range = (0.24, 0.28)
    left_box_x_range = (0.34, 0.38)
    right_box_x_range = (-0.38, -0.34)
    mug_abs_y_range = (0.48, 0.52)
    box_abs_y_range = (0.44, 0.48)
    min_mug_arm_dist = 0.18
    max_mug_arm_dist = 0.50
    min_box_arm_dist = 0.20
    min_mug_middle_dist = 0.18
    min_box_middle_dist = 0.16

    # Start a bit more retracted than the shared kitchen-base default so the
    # arms have extra clearance before the first pregrasp.
    home_arm_qpos_left = (-0.30, -0.30, 0.0, -2.12, 0.0, 1.98, np.pi / 4)
    home_arm_qpos_right = (0.30, -0.30, 0.0, -2.12, 0.0, 1.98, np.pi / 4)

    @property
    def _workspace_target(self):
        return np.array(self.workspace_offset) + np.array([0.0, 0.0, 0.08])

    def _build_plain_workspace_table(self):
        # Match FoodServe's simple workspace table exactly.
        super()._build_plain_workspace_table()

    def _set_plain_workspace_table_pose(self):
        super()._set_plain_workspace_table_pose()

    def _load_task_scene(self, options: dict):
        self.left_mug = self._build_robotwin_mesh(
            "039_mug",
            self.mug_model_id,
            "left_exchange_mug",
            self.mug_scale,
            self.mug_mass,
        )
        self.right_mug = self._build_robotwin_mesh(
            "039_mug",
            self.mug_model_id,
            "right_exchange_mug",
            self.mug_scale,
            self.mug_mass,
        )
        self.left_mug_spawn_z = 0.005
        self.right_mug_spawn_z = 0.005

        # Compatibility aliases so existing bottle-exchange tooling keeps working.
        self.left_bottle = self.left_mug
        self.right_bottle = self.right_mug
        self.left_bottle_spawn_z = self.left_mug_spawn_z
        self.right_bottle_spawn_z = self.right_mug_spawn_z
        self.bottle_radius = self.mug_radius

        self.left_box_parts = self._build_transparent_box("left_exchange_box")
        self.right_box_parts = self._build_transparent_box("right_exchange_box")
        self._apply_task_contact_materials()

    def _build_robotwin_mesh(self, group, model_id, name, scale, mass):
        from mani_skill.utils.structs.actor import Actor

        mesh_scale = [scale] * 3 if np.isscalar(scale) else list(scale)
        asset_root = (
            Path(__file__).resolve().parents[3]
            / "assets"
            / "robotwin"
            / "objects"
            / group
        )
        collision_file = asset_root / "collision" / f"base{model_id}.glb"
        visual_file = asset_root / "visual" / f"base{model_id}.glb"
        if not collision_file.is_file() or not visual_file.is_file():
            raise FileNotFoundError(
                f"Missing RoboTwin {group}/base{model_id} assets under {asset_root}"
            )

        material = sapien.physx.PhysxMaterial(
            static_friction=self.object_friction,
            dynamic_friction=self.object_friction,
            restitution=0.0,
        )
        actors_per_env = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, -100])
            builder.add_multiple_convex_collisions_from_file(
                filename=str(collision_file),
                scale=mesh_scale,
                material=material,
                density=100.0,
            )
            builder.add_visual_from_file(
                filename=str(visual_file),
                scale=mesh_scale,
            )
            actor = builder.build(name=f"{name}-{i}")
            actor.mass = mass
            actors_per_env.append(actor)
            self.remove_from_state_dict_registry(actor)

        merged_actor = Actor.merge(actors_per_env, name=name)
        self.add_to_state_dict_registry(merged_actor)
        return merged_actor

    def _apply_task_contact_materials(self):
        gripper_material = sapien.physx.PhysxMaterial(
            static_friction=self.gripper_friction,
            dynamic_friction=self.gripper_friction,
            restitution=0.0,
        )
        box_material = sapien.physx.PhysxMaterial(
            static_friction=self.box_friction,
            dynamic_friction=self.box_friction,
            restitution=0.0,
        )
        for agent in (self.left_agent, self.right_agent):
            for link in (agent.finger1_link, agent.finger2_link):
                for body in link._bodies:
                    for shape in body.collision_shapes:
                        shape.set_physical_material(gripper_material)
        for parts in (self.left_box_parts, self.right_box_parts):
            for part in parts:
                for body in part._bodies:
                    for shape in body.collision_shapes:
                        shape.set_physical_material(box_material)

    def _build_transparent_box(self, prefix):
        hx, hy = self.box_half_size_xy
        t = self.box_wall_thickness
        wall_z = self.box_wall_height / 2
        color = [0.25, 0.25, 0.28, 0.55]
        bottom_color = [0.12, 0.12, 0.14, 0.75]
        z0 = self.box_bottom_center_z
        return [
            actors.build_box(
                self.scene,
                half_sizes=[hx, hy, t / 2],
                color=bottom_color,
                name=f"{prefix}_bottom",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, -100]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, t / 2, wall_z],
                color=color,
                name=f"{prefix}_front_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, -100]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, t / 2, wall_z],
                color=color,
                name=f"{prefix}_back_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, -100]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[t / 2, hy, wall_z],
                color=color,
                name=f"{prefix}_left_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, -100]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[t / 2, hy, wall_z],
                color=color,
                name=f"{prefix}_right_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, -100]),
            ),
        ]

    def _set_transparent_box_pose(self, parts, rel_xy):
        if torch.is_tensor(rel_xy):
            x = rel_xy[:, 0]
            y = rel_xy[:, 1]
        else:
            x, y = rel_xy
        hx, hy = self.box_half_size_xy
        z0 = self.box_bottom_center_z
        wall_z = self.box_wall_height / 2
        rels = [
            (x, y, z0),
            (x, y - hy, wall_z),
            (x, y + hy, wall_z),
            (x - hx, y, wall_z),
            (x + hx, y, wall_z),
        ]
        for part, rel in zip(parts, rels):
            if torch.is_tensor(rel_xy):
                self._set_actor_pose_rel_components(part, rel)
            else:
                self._set_actor_pose_rel(part, rel)

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        if not hasattr(self, "left_mug_staged_once"):
            self.left_mug_staged_once = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.right_mug_staged_once = torch.zeros_like(self.left_mug_staged_once)
            self.left_mug_received_once = torch.zeros_like(self.left_mug_staged_once)
            self.right_mug_received_once = torch.zeros_like(self.left_mug_staged_once)
        self.left_mug_staged_once[env_idx] = False
        self.right_mug_staged_once[env_idx] = False
        self.left_mug_received_once[env_idx] = False
        self.right_mug_received_once[env_idx] = False

        self._set_retracted_arm_qpos()

        b = len(env_idx)
        left_mug_xy = self._sample_mug_xy(
            b, self.left_mug_x_range, self.mug_abs_y_range, side=-1.0
        )
        right_mug_xy = self._sample_mug_xy(
            b, self.right_mug_x_range, self.mug_abs_y_range, side=1.0
        )
        left_box_xy = self._sample_box_xy(
            b, self.left_box_x_range, self.box_abs_y_range, side=-1.0
        )
        right_box_xy = self._sample_box_xy(
            b, self.right_box_x_range, self.box_abs_y_range, side=1.0
        )

        self._set_actor_pose_rel_xy(
            self.left_mug, left_mug_xy, self.left_mug_spawn_z, q=self._sample_upright_yaw(b)
        )
        self._set_actor_pose_rel_xy(
            self.right_mug,
            right_mug_xy,
            self.right_mug_spawn_z,
            q=self._sample_upright_yaw(b),
        )
        self._set_transparent_box_pose(self.left_box_parts, left_box_xy)
        self._set_transparent_box_pose(self.right_box_parts, right_box_xy)

    def _set_retracted_arm_qpos(self):
        for agent, home in (
            (self.left_agent, self.home_arm_qpos_left),
            (self.right_agent, self.home_arm_qpos_right),
        ):
            arm_q = (
                torch.tensor(
                    list(home) + [0.04, 0.04],
                    dtype=torch.float32,
                    device=self.device,
                )
                .expand(self.num_envs, -1)
                .clone()
            )
            agent.reset(arm_q)

    def _sample_upright_yaw(self, b):
        yaw = torch.full((b,), self.mug_yaw_offset, device=self.device)
        if self.object_yaw_range > 0:
            yaw = yaw + (
                torch.rand((b,), device=self.device) * 2.0 - 1.0
            ) * self.object_yaw_range
        flip = torch.randint(0, 2, (b,), device=self.device).to(torch.float32)
        yaw = yaw + flip * np.pi
        c = torch.cos(yaw / 2)
        s = torch.sin(yaw / 2)
        root_half = np.sqrt(0.5)
        return torch.stack(
            [c * root_half, c * root_half, s * root_half, s * root_half],
            dim=1,
        )

    def _sample_mug_xy(self, b, x_range, abs_y_range, side):
        xy = torch.zeros((b, 2), device=self.device)
        needs_sample = torch.ones((b,), dtype=torch.bool, device=self.device)
        arm_xy = torch.tensor([0.0, 0.68 * side], device=self.device)
        middle_xy = torch.tensor(self.middle_stage_rel_xy, device=self.device)
        for _ in range(20):
            sample_count = int(needs_sample.sum().item())
            if sample_count == 0:
                break
            candidate = torch.zeros((sample_count, 2), device=self.device)
            candidate[:, 0] = (
                torch.rand((sample_count,), device=self.device)
                * (x_range[1] - x_range[0])
                + x_range[0]
            )
            candidate[:, 1] = side * (
                torch.rand((sample_count,), device=self.device)
                * (abs_y_range[1] - abs_y_range[0])
                + abs_y_range[0]
            )
            dist_to_arm = torch.linalg.norm(candidate - arm_xy, axis=1)
            dist_to_middle = torch.linalg.norm(candidate - middle_xy, axis=1)
            accepted = (
                (dist_to_arm > self.min_mug_arm_dist)
                & (dist_to_arm < self.max_mug_arm_dist)
                & (dist_to_middle > self.min_mug_middle_dist)
            )
            sampled_indices = torch.nonzero(needs_sample, as_tuple=False).flatten()
            accepted_indices = sampled_indices[accepted]
            xy[accepted_indices] = candidate[accepted]
            needs_sample[accepted_indices] = False
        if needs_sample.any():
            sample_count = int(needs_sample.sum().item())
            xy[needs_sample, 0] = (
                torch.rand((sample_count,), device=self.device)
                * (x_range[1] - x_range[0])
                + x_range[0]
            )
            xy[needs_sample, 1] = side * abs_y_range[0]
        return xy

    def _sample_box_xy(self, b, x_range, abs_y_range, side):
        xy = torch.zeros((b, 2), device=self.device)
        needs_sample = torch.ones((b,), dtype=torch.bool, device=self.device)
        arm_xy = torch.tensor([0.0, 0.68 * side], device=self.device)
        middle_xy = torch.tensor(self.middle_stage_rel_xy, device=self.device)
        for _ in range(20):
            sample_count = int(needs_sample.sum().item())
            if sample_count == 0:
                break
            candidate = torch.zeros((sample_count, 2), device=self.device)
            candidate[:, 0] = (
                torch.rand((sample_count,), device=self.device)
                * (x_range[1] - x_range[0])
                + x_range[0]
            )
            candidate[:, 1] = side * (
                torch.rand((sample_count,), device=self.device)
                * (abs_y_range[1] - abs_y_range[0])
                + abs_y_range[0]
            )
            dist_to_arm = torch.linalg.norm(candidate - arm_xy, axis=1)
            dist_to_middle = torch.linalg.norm(candidate - middle_xy, axis=1)
            accepted = (dist_to_arm > self.min_box_arm_dist) & (
                dist_to_middle > self.min_box_middle_dist
            )
            sampled_indices = torch.nonzero(needs_sample, as_tuple=False).flatten()
            accepted_indices = sampled_indices[accepted]
            xy[accepted_indices] = candidate[accepted]
            needs_sample[accepted_indices] = False
        if needs_sample.any():
            sample_count = int(needs_sample.sum().item())
            xy[needs_sample, 0] = (
                torch.rand((sample_count,), device=self.device)
                * (x_range[1] - x_range[0])
                + x_range[0]
            )
            xy[needs_sample, 1] = side * abs_y_range[0]
        return xy

    def _set_actor_pose_rel_xy(self, actor, rel_xy, rel_z, q=None):
        b = rel_xy.shape[0]
        rel = torch.zeros((b, 3), device=self.device)
        rel[:, :2] = rel_xy
        rel[:, 2] = rel_z
        actor.set_pose(Pose.create_from_pq(p=self.workspace_offset_tensor + rel, q=q))

    def _set_actor_pose_rel_components(self, actor, rel):
        x, y, z = rel
        p = self.workspace_offset_tensor.expand(x.shape[0], -1).clone()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += z
        actor.set_pose(Pose.create_from_pq(p=p))

    def _mug_in_box(self, mug, box_parts):
        center = box_parts[0]
        in_xy = self._inside_xy(
            mug, center, self.box_half_size_xy[0], self.box_half_size_xy[1]
        )
        low = mug.pose.p[:, 2] < center.pose.p[:, 2] + self.box_success_height
        return in_xy & low

    def _mug_in_middle(self, mug):
        rel = mug.pose.p - self.workspace_offset_tensor
        delta_xy = torch.abs(
            rel[:, :2]
            - torch.tensor(self.middle_stage_rel_xy, device=self.device).expand(
                self.num_envs, -1
            )
        )
        in_xy = (delta_xy[:, 0] < self.middle_half_size_xy[0]) & (
            delta_xy[:, 1] < self.middle_half_size_xy[1]
        )
        low = rel[:, 2] < self.middle_stage_height
        slow = torch.linalg.norm(mug.linear_velocity, axis=1) < self.middle_stage_speed
        not_grasped = (~self.left_agent.is_grasping(mug)) & (
            ~self.right_agent.is_grasping(mug)
        )
        return in_xy & low & slow & not_grasped

    def _mug_upright(self, mug):
        q = mug.pose.q
        local_up_dot_world_up = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
        return local_up_dot_world_up > 0.70

    def evaluate(self):
        left_staged = self._mug_in_middle(self.left_mug)
        right_staged = self._mug_in_middle(self.right_mug)
        self.left_mug_staged_once = self.left_mug_staged_once | left_staged
        self.right_mug_staged_once = self.right_mug_staged_once | right_staged

        left_received = self.left_mug_staged_once & self.right_agent.is_grasping(
            self.left_mug
        )
        right_received = self.right_mug_staged_once & self.left_agent.is_grasping(
            self.right_mug
        )
        self.left_mug_received_once = self.left_mug_received_once | left_received
        self.right_mug_received_once = self.right_mug_received_once | right_received

        left_in_right_box = self._mug_in_box(self.left_mug, self.right_box_parts)
        right_in_left_box = self._mug_in_box(self.right_mug, self.left_box_parts)
        left_upright = self._mug_upright(self.left_mug)
        right_upright = self._mug_upright(self.right_mug)

        # The task outcome is the exchanged placement itself: each cup must be
        # inside the opposite-side box.  Handoff, upright, and static predicates
        # remain available below as diagnostics and dataset quality gates.
        success = left_in_right_box & right_in_left_box
        return {
            "success": success,
            "left_mug_staged_once": self.left_mug_staged_once,
            "right_mug_staged_once": self.right_mug_staged_once,
            "left_mug_received_once": self.left_mug_received_once,
            "right_mug_received_once": self.right_mug_received_once,
            "left_mug_in_right_box": left_in_right_box,
            "right_mug_in_left_box": right_in_left_box,
            "left_mug_upright": left_upright,
            "right_mug_upright": right_upright,
            # Compatibility aliases for existing bottle-exchange tooling.
            "left_bottle_handoff_once": self.left_mug_received_once,
            "right_bottle_handoff_once": self.right_mug_received_once,
            "left_bottle_in_right_box": left_in_right_box,
            "right_bottle_in_left_box": right_in_left_box,
        }
