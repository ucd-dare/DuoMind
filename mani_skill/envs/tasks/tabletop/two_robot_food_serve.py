from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .two_robot_kitchen_base import TwoRobotKitchenReplicaCADBaseEnv


def _robotwin_asset_paths(group: str, model_id: int) -> tuple[Path, Path]:
    """Resolve packaged assets first, then the full RoboTwin workspace archive."""
    roots = (
        Path(__file__).resolve().parents[3] / "assets" / "robotwin" / "objects" / group,
        Path(__file__).resolve().parents[3] / "assets" / "objects" / group,
        Path(__file__).resolve().parents[6] / "assets" / "objects" / group,
    )
    for root in roots:
        collision_file = root / "collision" / f"base{model_id}.glb"
        visual_file = root / "visual" / f"base{model_id}.glb"
        if collision_file.is_file() and visual_file.is_file():
            return collision_file, visual_file
    raise FileNotFoundError(
        f"Missing RoboTwin {group}/base{model_id} assets; searched: "
        + ", ".join(str(root) for root in roots)
    )


@register_env(
    "TwoRobotFoodServeReplicaCAD-v1",
    max_episode_steps=1500,
    asset_download_ids=["ReplicaCAD", "RoboCasa"],
)
class TwoRobotFoodServeReplicaCADEnv(TwoRobotKitchenReplicaCADBaseEnv):
    """Cooperative serving task: put a mug and bread on a fixed raised tray."""

    # Spawn ranges narrowed to the boxes the demo-gen solver was validated on.
    # Bread is intentionally farther from the right robot than before; rejection
    # sampling (below) still keeps each object clear of the tray and its own arm.
    mug_x_range = (-0.28, -0.16)
    bread_x_range = (0.12, 0.24)
    mug_abs_y_range = (0.27, 0.39)
    bread_abs_y_range = (0.18, 0.28)
    min_object_tray_dist = 0.24
    min_object_arm_dist = 0.18
    object_yaw_range = np.deg2rad(15.0)
    # +x is closer to the fixed global camera; -y is toward the left robot.
    # This keeps the tray outside the right robot's useful reach.
    tray_center_rel = (0.25, -0.31)
    tray_lifter_half_sizes = (0.12, 0.08, 0.045)
    tray_success_half_xy = (0.11, 0.078)
    # Object origins sit at their bottoms. A placed bread is only a few
    # centimetres above the support top; the previous 0.15 m allowance could
    # label bread passing through the tray footprint in mid-air as successful.
    bread_success_max_clearance = 0.04

    # RoboTwin assets used by the original Hanging Mug and Place Bread Basket
    # tasks. base4 is light ivory and thick-walled; base0 is the round braided bread.
    mug_model_id = 4
    mug_scale = 0.050
    mug_mass = 0.003
    mug_radius = 0.048
    mug_height = 0.067
    bread_model_id = 0
    bread_scale = (0.0265, 0.0500, 0.0265)
    bread_mass = 0.010
    bread_radius = 0.035
    bread_height = 0.040
    object_friction = 10.0
    gripper_friction = 10.0

    # Dataset camera setup: the observation uses exactly THREE views -- the two
    # wrist cameras (added automatically by the ``panda_wristcam`` arms:
    # ``panda_wristcam-0-hand_camera`` = LEFT, ``panda_wristcam-1-hand_camera`` =
    # RIGHT) plus the fixed ``global_camera``. The other static kitchen cameras are
    # dropped here so an ``obs_mode="rgb"`` dataset is small and exactly 3-view.
    # Resolution of the global view (4:3). 512x384 is the standard dataset
    # global_camera resolution shared across tasks (CleanTable matches this);
    # raise both to 1024x768 if full resolution is ever needed.
    global_camera_width = 512
    global_camera_height = 384

    @property
    def _default_sensor_configs(self):
        cameras = {
            c.uid: c for c in super()._default_sensor_configs
            if c.uid in {"global_camera", "visualization_camera"}
        }
        global_cam = cameras["global_camera"]
        return [
            CameraConfig(
                "global_camera",
                global_cam.pose,
                self.global_camera_width,
                self.global_camera_height,
                global_cam.fov,
                global_cam.near,
                global_cam.far,
            ),
            cameras["visualization_camera"],
        ]

    def _load_task_scene(self, options: dict):
        self.mug = self._build_robotwin_mesh(
            "039_mug",
            self.mug_model_id,
            "serve_mug",
            self.mug_scale,
            self.mug_mass,
        )
        self.bread = self._build_robotwin_mesh(
            "075_bread",
            self.bread_model_id,
            "serve_bread",
            self.bread_scale,
            self.bread_mass,
        )
        self.tray, self.tray_spawn_z, self.tray_radius, _ = self._build_kitchen_object(
            "tray", "serve_tray", object_scale=1.0, seed=13, kinematic=True
        )
        self.tray_lifter = actors.build_box(
            self.scene,
            half_sizes=self.tray_lifter_half_sizes,
            color=[0.015, 0.015, 0.018, 1],
            name="serve_tray_lifter",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, -100]),
        )
        self._apply_task_contact_materials()

        # Compatibility aliases for downstream tooling that previously inspected
        # FoodServe's bowl/fruit actors directly.
        self.bowl = self.mug
        self.fruit = self.bread
        self.bowl_radius = self.mug_radius
        self.fruit_radius = self.bread_radius

    def _build_robotwin_mesh(self, group, model_id, name, scale, mass):
        """Build one RoboTwin GLB actor per sub-scene and merge the actor view."""
        from mani_skill.utils.structs.actor import Actor

        mesh_scale = [scale] * 3 if np.isscalar(scale) else list(scale)
        # Packaged ManiSkill assets are preferred, but the full RoboTwin checkout
        # keeps its object archive at the workspace root.  The repository
        # reorganization that introduced the packaged bread asset did not stage
        # 039_mug, so retain the original RoboTwin location as a fallback.
        collision_file, visual_file = _robotwin_asset_paths(group, model_id)

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
        for agent in (self.left_agent, self.right_agent):
            for link in (agent.finger1_link, agent.finger2_link):
                for body in link._bodies:
                    for shape in body.collision_shapes:
                        shape.set_physical_material(gripper_material)

    def _set_actor_pose_rel_xy(self, actor, rel_xy, rel_z, q=None):
        """Set an actor's pose from per-env relative xy (b,2 tensor) + scalar z."""
        b = rel_xy.shape[0]
        rel = torch.zeros((b, 3), device=self.device)
        rel[:, :2] = rel_xy
        rel[:, 2] = rel_z
        actor.set_pose(Pose.create_from_pq(p=self.workspace_offset_tensor + rel, q=q))

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        mug_xy = self._sample_object_xy(
            b,
            x_range=self.mug_x_range,
            abs_y_range=self.mug_abs_y_range,
            side=-1.0,
        )
        bread_xy = self._sample_object_xy(
            b,
            x_range=self.bread_x_range,
            abs_y_range=self.bread_abs_y_range,
            side=1.0,
        )
        self._set_actor_pose_rel_xy(
            self.mug, mug_xy, 0.005, q=self._sample_upright_yaw(b)
        )
        self._set_actor_pose_rel_xy(
            self.bread, bread_xy, 0.005, q=self._sample_upright_yaw(b)
        )

        # The fixed tray is exactly centered on the fixed black support.
        lifter_z = self.tray_lifter_half_sizes[2]
        self._set_actor_pose_rel(
            self.tray_lifter,
            [self.tray_center_rel[0], self.tray_center_rel[1], lifter_z],
        )
        self._set_actor_pose_rel(
            self.tray,
            [
                self.tray_center_rel[0],
                self.tray_center_rel[1],
                2 * lifter_z + self.tray_spawn_z,
            ],
        )

    def _sample_upright_yaw(self, b):
        """RoboTwin +y-up mesh rotation with a small randomized world yaw."""
        yaw = (
            torch.rand((b,), device=self.device) * 2.0 - 1.0
        ) * self.object_yaw_range
        c = torch.cos(yaw / 2)
        s = torch.sin(yaw / 2)
        root_half = np.sqrt(0.5)
        # q_world_yaw * q_x(+90deg), in wxyz order.
        return torch.stack(
            [c * root_half, c * root_half, s * root_half, s * root_half],
            dim=1,
        )

    def _sample_object_xy(self, b, x_range, abs_y_range, side):
        xy = torch.zeros((b, 2), device=self.device)
        needs_sample = torch.ones((b,), dtype=torch.bool, device=self.device)
        tray_xy = torch.tensor(self.tray_center_rel, device=self.device)
        arm_xy = torch.tensor([0.0, 0.68 * side], device=self.device)
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
            far_from_tray = (
                torch.linalg.norm(candidate - tray_xy, axis=1)
                > self.min_object_tray_dist
            )
            far_from_arm = (
                torch.linalg.norm(candidate - arm_xy, axis=1)
                > self.min_object_arm_dist
            )
            accepted = far_from_tray & far_from_arm
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

    def _object_on_tray(self, actor, max_clearance=0.15):
        delta_xy = torch.abs(actor.pose.p[:, :2] - self.tray.pose.p[:, :2])
        # The serving tray is rectangular. An axis-aligned footprint check
        # correctly accepts objects near its corners, unlike the old circle test.
        on_xy = (delta_xy[:, 0] < self.tray_success_half_xy[0]) & (
            delta_xy[:, 1] < self.tray_success_half_xy[1]
        )
        # The RoboCasa tray origin is above its bottom/support contact, while the
        # RoboTwin actors' origins sit at their bottoms. Compare against the fixed
        # support top rather than comparing these unlike mesh origins directly.
        support_top = self.workspace_offset[2] + 2 * self.tray_lifter_half_sizes[2]
        resting_height = (actor.pose.p[:, 2] > support_top - 0.004) & (
            actor.pose.p[:, 2] < support_top + max_clearance
        )
        return on_xy & resting_height

    def _mug_on_tray(self):
        return self._object_on_tray(self.mug)

    def _bread_on_tray(self):
        contact_force = self.scene.get_pairwise_contact_forces(self.bread, self.tray)
        touching_tray = torch.linalg.norm(contact_force, axis=1) > 1e-5
        released = ~(
            self.left_agent.is_grasping(self.bread)
            | self.right_agent.is_grasping(self.bread)
        )
        bread_static = self.bread.is_static(lin_thresh=0.02, ang_thresh=1.0)
        return (
            self._object_on_tray(
                self.bread, max_clearance=self.bread_success_max_clearance
            )
            & touching_tray
            & released
            & bread_static
        )

    def _mug_upright(self):
        # RoboTwin mug meshes use local +y as their vertical axis. For a wxyz
        # quaternion, R[world_z, local_y] = 2 * (w*x + y*z).
        q = self.mug.pose.q
        local_up_dot_world_up = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
        return local_up_dot_world_up > 0.85

    def evaluate(self):
        mug_on_tray = self._mug_on_tray()
        bread_on_tray = self._bread_on_tray()
        mug_upright = self._mug_upright()
        separated = (
            torch.linalg.norm(self.mug.pose.p[:, :2] - self.bread.pose.p[:, :2], axis=1)
            > 0.065
        )
        objects_static = (
            torch.linalg.norm(self.mug.linear_velocity, axis=1) < 0.10
        ) & (torch.linalg.norm(self.bread.linear_velocity, axis=1) < 0.10)
        # The task succeeds exactly when both served objects are on the fixed tray.
        success = mug_on_tray & bread_on_tray
        return {
            "success": success,
            "mug_on_tray": mug_on_tray,
            "bread_on_tray": bread_on_tray,
            "mug_upright": mug_upright,
            "objects_separated": separated,
        }
