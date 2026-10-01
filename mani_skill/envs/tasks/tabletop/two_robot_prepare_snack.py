"""Prepare Snack: place four identical RoboTwin square breads on two plates."""

from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR

from .two_robot_prepare_fruit import TwoRobotPrepareFruitReplicaCADEnv


@register_env(
    "TwoRobotPrepareSnackReplicaCAD-v1",
    max_episode_steps=2000,
    asset_download_ids=["ReplicaCAD", "RoboCasa"],
)
class TwoRobotPrepareSnackReplicaCADEnv(TwoRobotPrepareFruitReplicaCADEnv):
    """Put two identical square breads onto each plate."""

    # Place Bread Skillet's square 075_bread/base1. All four breads use this exact
    # model, scale, mass, collision geometry, and contact material.
    square_bread_model_id = 1
    square_bread_scale = (0.0267, 0.0232, 0.0267)
    square_bread_mass = 0.080

    def _load_task_scene(self, options: dict):
        self.plates = []
        self.plate_spawn_zs = []
        self.plate_radii = []
        plate_paths = [
            str(ROBOCASA_ASSET_DIR / "objects/objaverse/plate/plate_8/model.xml"),
            str(ROBOCASA_ASSET_DIR / "objects/objaverse/plate/plate_9/model.xml"),
        ]
        for i, plate_path in enumerate(plate_paths):
            plate, spawn_z, radius, _ = self._build_kitchen_object(
                plate_path,
                f"snack_plate_{i}",
                object_scale=self.plate_scale,
                seed=32 + i,
                obj_registries=("objaverse",),
                kinematic=True,
            )
            self.plates.append(plate)
            self.plate_spawn_zs.append(spawn_z)
            self.plate_radii.append(radius)

        self.left_breads = [
            self._build_square_bread(f"snack_square_bread_left_{i}") for i in range(2)
        ]
        self.right_breads = [
            self._build_square_bread(f"snack_square_bread_right_{i}") for i in range(2)
        ]
        # Compatibility aliases retained for the shared Prepare Fruit solver.
        self.round_breads = self.left_breads
        self.square_breads = self.right_breads
        self.breads = self.left_breads
        self.bananas = self.left_breads
        self.oranges = self.right_breads
        self._apply_task_contact_materials()

    def _build_square_bread(self, name):
        """Build RoboTwin Place Bread Skillet's square bread at matched size."""
        from mani_skill.utils.structs.actor import Actor

        asset_root = Path(__file__).resolve().parents[6] / "assets/objects/075_bread"
        collision_file = asset_root / "collision" / f"base{self.square_bread_model_id}.glb"
        visual_file = asset_root / "visual" / f"base{self.square_bread_model_id}.glb"
        if not collision_file.is_file() or not visual_file.is_file():
            asset_root = Path(__file__).resolve().parents[3] / "assets/objects/075_bread"
            collision_file = asset_root / "collision" / f"base{self.square_bread_model_id}.glb"
            visual_file = asset_root / "visual" / f"base{self.square_bread_model_id}.glb"
        material = sapien.physx.PhysxMaterial(
            static_friction=self.fruit_friction,
            dynamic_friction=self.fruit_friction,
            restitution=0.0,
        )
        actors_per_env = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.set_scene_idxs([i])
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
            actor = builder.build(name=f"{name}-{i}")
            actor.mass = self.square_bread_mass
            actors_per_env.append(actor)
            self.remove_from_state_dict_registry(actor)
        bread = Actor.merge(actors_per_env, name=name)
        bread.set_locked_motion_axes([False, False, False, True, False, True])
        self.add_to_state_dict_registry(bread)
        return bread

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        plate_xy = self._sample_plate_xy(b)
        left_xy = self._sample_fruit_side_xy(b, side=-1.0, plate_xy=plate_xy)
        right_xy = self._sample_fruit_side_xy(b, side=1.0, plate_xy=plate_xy)
        self._set_actor_pose_rel_xy(self.plates[0], plate_xy[:, 0], self.plate_spawn_zs[0])
        self._set_actor_pose_rel_xy(self.plates[1], plate_xy[:, 1], self.plate_spawn_zs[1])
        upright = euler2quat(np.pi / 2, 0, 0)
        for i in range(2):
            self._set_actor_pose_rel_xy(self.left_breads[i], left_xy[:, i], 0.005, q=upright)
            self._set_actor_pose_rel_xy(self.right_breads[i], right_xy[:, i], 0.005, q=upright)

    def evaluate(self):
        plate_flags = []
        info = {}
        for i, plate in enumerate(self.plates):
            left_on = torch.stack(
                [self._fruit_on_plate(x, plate, self.plate_radii[i]) for x in self.left_breads],
                dim=1,
            )
            right_on = torch.stack(
                [self._fruit_on_plate(x, plate, self.plate_radii[i]) for x in self.right_breads],
                dim=1,
            )
            has_left, has_right = torch.any(left_on, 1), torch.any(right_on, 1)
            plate_flags.append((left_on.sum(1) == 1) & (right_on.sum(1) == 1))
            info[f"plate_{i}_has_left_bread"] = has_left
            info[f"plate_{i}_has_right_bread"] = has_right
        all_plates_ready = torch.stack(plate_flags, dim=1).all(dim=1)
        all_breads = (*self.left_breads, *self.right_breads)
        breads_face_up = torch.stack([self._bread_face_up(x) for x in all_breads], 1).all(1)
        food_static = torch.stack(
            [x.is_static(lin_thresh=0.015, ang_thresh=0.5) for x in all_breads], 1
        ).all(1)
        success = all_plates_ready & breads_face_up & food_static & self._both_static()
        info.update(
            success=success,
            all_plates_ready=all_plates_ready,
            breads_face_up=breads_face_up,
            food_static=food_static,
        )
        return info
