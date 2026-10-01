"""Exchange two square breads between the Panda workspaces."""

from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from robopoly.utils.building import actors
from robopoly.utils.registration import register_env
from robopoly.utils.structs.actor import Actor

from .two_robot_bottle_exchange import TwoRobotBottleExchangeReplicaCADEnv


@register_env(
    "TwoRobotBreadExchangeReplicaCAD-v1",
    max_episode_steps=3000,
    asset_download_ids=["ReplicaCAD", "RoboCasa"],
)
class TwoRobotBreadExchangeReplicaCADEnv(TwoRobotBottleExchangeReplicaCADEnv):
    """Put each square bread in the box on the opposite side."""

    # Use Prepare Snack's square bread geometry, but match Food Serve's bread
    # mass and its object/gripper contact frictions as requested.
    square_bread_model_id = 1
    square_bread_scale = (0.0267, 0.0232, 0.0267)
    square_bread_mass = 0.010
    object_friction = 10.0
    gripper_friction = 10.0

    # Lift the thin breads clear of the tabletop so a top-down pinch can close
    # without the fingertips scraping the table. Each pedestal is fixed and
    # follows its bread's randomized starting XY position.
    bread_pedestal_half_sizes = (0.065, 0.065, 0.035)
    # Broader than the cup task: the bread and its pedestal are sampled together
    # across a useful area on each side of the table.
    left_mug_x_range = (-0.38, -0.16)
    right_mug_x_range = (0.16, 0.38)
    mug_abs_y_range = (0.40, 0.57)
    # Account for the pedestal footprint, not only the bread centre, when the
    # inherited sampler rejects positions near a robot base or the middle line.
    min_mug_arm_dist = 0.28
    min_mug_middle_dist = 0.40

    def _apply_task_contact_materials(self):
        super()._apply_task_contact_materials()
        pedestal_material = sapien.physx.PhysxMaterial(
            static_friction=self.object_friction,
            dynamic_friction=self.object_friction,
            restitution=0.0,
        )
        for pedestal in (
            self.left_bread_pedestal,
            self.right_bread_pedestal,
        ):
            for body in pedestal._bodies:
                for shape in body.collision_shapes:
                    shape.set_physical_material(pedestal_material)

    def _build_square_bread(self, name):
        asset_root = Path(__file__).resolve().parents[6] / "assets/objects/075_bread"
        collision_file = asset_root / "collision" / f"base{self.square_bread_model_id}.glb"
        visual_file = asset_root / "visual" / f"base{self.square_bread_model_id}.glb"
        if not collision_file.is_file() or not visual_file.is_file():
            asset_root = Path(__file__).resolve().parents[3] / "assets/objects/075_bread"
            collision_file = asset_root / "collision" / f"base{self.square_bread_model_id}.glb"
            visual_file = asset_root / "visual" / f"base{self.square_bread_model_id}.glb"
        if not collision_file.is_file() or not visual_file.is_file():
            raise FileNotFoundError(f"Missing square-bread assets under {asset_root}")

        material = sapien.physx.PhysxMaterial(
            static_friction=self.object_friction,
            dynamic_friction=self.object_friction,
            restitution=0.0,
        )
        actors_per_env = []
        for index in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([index])
            builder.initial_pose = sapien.Pose(p=[0, 0, -100])
            builder.add_multiple_convex_collisions_from_file(
                filename=str(collision_file),
                scale=list(self.square_bread_scale),
                material=material,
                density=100.0,
            )
            builder.add_visual_from_file(
                filename=str(visual_file), scale=list(self.square_bread_scale)
            )
            actor = builder.build(name=f"{name}-{index}")
            actor.mass = self.square_bread_mass
            actors_per_env.append(actor)
            self.remove_from_state_dict_registry(actor)
        bread = Actor.merge(actors_per_env, name=name)
        # The serving face is local Y: allow yaw but prevent edge-over rolling,
        # exactly as Prepare Snack does.
        bread.set_locked_motion_axes([False, False, False, True, False, True])
        self.add_to_state_dict_registry(bread)
        return bread

    def _load_task_scene(self, options: dict):
        pedestal_color = [0.12, 0.12, 0.14, 1.0]
        self.left_bread_pedestal = actors.build_box(
            self.scene,
            half_sizes=self.bread_pedestal_half_sizes,
            color=pedestal_color,
            name="left_exchange_bread_pedestal",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, -100]),
        )
        self.right_bread_pedestal = actors.build_box(
            self.scene,
            half_sizes=self.bread_pedestal_half_sizes,
            color=pedestal_color,
            name="right_exchange_bread_pedestal",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, -100]),
        )
        self.left_bread = self._build_square_bread("left_exchange_bread")
        self.right_bread = self._build_square_bread("right_exchange_bread")
        pedestal_height = 2 * self.bread_pedestal_half_sizes[2]
        self.left_bread_spawn_z = pedestal_height + 0.005
        self.right_bread_spawn_z = pedestal_height + 0.005

        # Reuse the proven exchange geometry and solver through compatibility
        # aliases; user-facing task state below uses bread terminology.
        self.left_mug = self.left_bread
        self.right_mug = self.right_bread
        self.left_mug_spawn_z = self.left_bread_spawn_z
        self.right_mug_spawn_z = self.right_bread_spawn_z
        self.left_bottle = self.left_bread
        self.right_bottle = self.right_bread
        self.left_bottle_spawn_z = self.left_bread_spawn_z
        self.right_bottle_spawn_z = self.right_bread_spawn_z
        self.bottle_radius = 0.05

        self.left_box_parts = self._build_transparent_box("left_exchange_box")
        self.right_box_parts = self._build_transparent_box("right_exchange_box")
        self._apply_task_contact_materials()

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_task_episode(env_idx, options)

        # The parent samples both bread positions. Reuse those XY coordinates so
        # every bread begins centered on its own fixed square pedestal.
        pedestal_z = self.bread_pedestal_half_sizes[2]
        for bread, pedestal in (
            (self.left_bread, self.left_bread_pedestal),
            (self.right_bread, self.right_bread_pedestal),
        ):
            relative_xy = bread.pose.p[..., :2] - self.workspace_offset_tensor[:2]
            if relative_xy.ndim == 1:
                relative_xy = relative_xy.unsqueeze(0)
            self._set_actor_pose_rel_xy(pedestal, relative_xy, pedestal_z)

    def _sample_upright_yaw(self, batch_size):
        # Prepare Snack uses this exact face-up orientation.
        quat = torch.tensor(
            euler2quat(np.pi / 2, 0.0, 0.0),
            dtype=torch.float32,
            device=self.device,
        )
        return quat.expand(batch_size, -1).clone()

    def evaluate(self):
        info = super().evaluate()
        info.update(
            left_bread_staged_once=info["left_mug_staged_once"],
            right_bread_staged_once=info["right_mug_staged_once"],
            left_bread_received_once=info["left_mug_received_once"],
            right_bread_received_once=info["right_mug_received_once"],
            left_bread_in_right_box=info["left_mug_in_right_box"],
            right_bread_in_left_box=info["right_mug_in_left_box"],
            left_bread_face_up=info["left_mug_upright"],
            right_bread_face_up=info["right_mug_upright"],
        )
        return info
