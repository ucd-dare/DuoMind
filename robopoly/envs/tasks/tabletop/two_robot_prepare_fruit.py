import numpy as np
import sapien
import torch
from pathlib import Path
from transforms3d.euler import euler2quat

from robopoly.utils.building import actors
from robopoly.utils.registration import register_env
from robopoly.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR
from robopoly.utils.structs.pose import Pose

from .two_robot_kitchen_base import TwoRobotKitchenReplicaCADBaseEnv


class TwoRobotPrepareFruitReplicaCADEnv(TwoRobotKitchenReplicaCADBaseEnv):
    fruit_friction = 10.0
    gripper_friction = 10.0
    # Match the bread used by TwoRobotFoodServeReplicaCAD-v1 exactly.
    bread_model_id = 0
    bread_scale = (0.0265, 0.0500, 0.0265)
    # Realistic food-scale masses. The previous 10 g bread (and the orange's
    # asset-default mass) were light enough for a fingertip graze to send them
    # sliding across the table.
    bread_mass = 0.080
    orange_mass = 0.150
    orange_scale = 1.0
    # Plates sit in the central strip both arms must reach. The original spread
    # (-0.06, 0.34) put the far plate at world x~1.74 -- ~0.76 m from the opposite
    # arm's base, right at the Panda's reach limit, where placements land off-plate
    # or go unstable. Centring the pair on the arms' x (1.4) at a tighter spread
    # keeps BOTH plates ~0.70 m away (comfortably reachable) while still leaving a
    # clear gap between the (now larger, see plate_scale) plates.
    # (Env change for motion-planning solvability.)
    plate_x_centers = (-0.18, 0.18)
    plate_x_noise = 0.03
    plate_y_noise = 0.035
    min_plate_dist = 0.34
    # The default plates (flat radius ~0.10-0.12 m) are too small to hold a 0.16 m
    # banana AND an orange without one knocking the other off. Scaling them up gives
    # room for both fruits side by side. (Env change for solvability.)
    plate_scale = 1.2
    # Compact, reachable food zones: clear of each pedestal but not at the outer
    # IK envelope where the second top-down grasp becomes unreliable.
    fruit_x_range = (-0.18, 0.18)
    fruit_abs_y_range = (0.28, 0.36)
    min_fruit_plate_dist = 0.18
    min_fruit_fruit_dist = 0.14
    # Keep food away from the robot pedestal/base, where top-down IK is difficult,
    # while retaining it in the owning arm's nearby workspace.
    min_fruit_arm_dist = 0.28

    def __init__(self, *args, orange_scale=None, **kwargs):
        if orange_scale is not None:
            orange_scale = float(orange_scale)
            if orange_scale <= 0:
                raise ValueError(f"orange_scale must be positive, got {orange_scale}")
            self.orange_scale = orange_scale
        super().__init__(*args, **kwargs)

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
                f"fruit_plate_{i}",
                object_scale=self.plate_scale,
                seed=32 + i,
                obj_registries=("objaverse",),
                # kinematic so a placed fruit brushing the rim cannot shove the plate
                # across the table (it stays a stable placement target).
                kinematic=True,
            )
            self.plates.append(plate)
            self.plate_spawn_zs.append(spawn_z)
            self.plate_radii.append(radius)

        self.breads = []
        for i in range(2):
            self.breads.append(self._build_food_serve_bread(f"food_bread_{i}"))
        # Compatibility alias for older tooling; the actors are breads, not bananas.
        self.bananas = self.breads

        self.oranges = []
        self.orange_spawn_zs = []
        for i, seed in enumerate([35, 36]):
            orange, spawn_z, _, _ = self._build_kitchen_object(
                "orange", f"fruit_orange_{i}", object_scale=self.orange_scale, seed=seed
            )
            for body in orange._bodies:
                body.mass = self.orange_mass
            self.oranges.append(orange)
            self.orange_spawn_zs.append(spawn_z)

        self._apply_task_contact_materials()

    def _build_food_serve_bread(self, name):
        """Build the same RoboTwin bread mesh and physical body as Food Serve."""
        from robopoly.utils.structs.actor import Actor

        asset_root = Path(__file__).resolve().parents[3] / "assets/objects/075_bread"
        collision_file = asset_root / "collision" / f"base{self.bread_model_id}.glb"
        visual_file = asset_root / "visual" / f"base{self.bread_model_id}.glb"
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
                scale=list(self.bread_scale),
                material=material,
                density=100.0,
            )
            builder.add_visual_from_file(
                filename=str(visual_file), scale=list(self.bread_scale)
            )
            actor = builder.build(name=f"{name}-{i}")
            actor.mass = self.bread_mass
            actors_per_env.append(actor)
            self.remove_from_state_dict_registry(actor)
        bread = Actor.merge(actors_per_env, name=name)
        # Task constraint: bread may translate and yaw, but its serving face must
        # remain upward rather than rolling onto an edge during grasp/release.
        # Locks are expressed in the body's local frame. The serving-face normal
        # is local Y, so allow only rotation about Y (in-plane yaw) and lock local
        # X/Z tilting.
        bread.set_locked_motion_axes([False, False, False, True, False, True])
        self.add_to_state_dict_registry(bread)
        return bread

    def _apply_task_contact_materials(self):
        """Give all fruit and both Panda grippers high-friction contact."""
        fruit_material = sapien.physx.PhysxMaterial(
            static_friction=self.fruit_friction,
            dynamic_friction=self.fruit_friction,
            restitution=0.0,
        )
        for fruit in (*self.bananas, *self.oranges):
            for body in fruit._bodies:
                for shape in body.collision_shapes:
                    shape.set_physical_material(fruit_material)

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

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        plate_xy = self._sample_plate_xy(b)
        bread_xy = self._sample_fruit_side_xy(b, side=-1.0, plate_xy=plate_xy)
        orange_xy = self._sample_fruit_side_xy(b, side=1.0, plate_xy=plate_xy)

        self._set_actor_pose_rel_xy(self.plates[0], plate_xy[:, 0], self.plate_spawn_zs[0])
        self._set_actor_pose_rel_xy(self.plates[1], plate_xy[:, 1], self.plate_spawn_zs[1])
        self._set_actor_pose_rel_xy(
            self.breads[0], bread_xy[:, 0], 0.005,
            q=euler2quat(np.pi / 2, 0, 0),
        )
        self._set_actor_pose_rel_xy(
            self.breads[1], bread_xy[:, 1], 0.005,
            q=euler2quat(np.pi / 2, 0, 0),
        )
        self._set_actor_pose_rel_xy(
            self.oranges[0], orange_xy[:, 0], self.orange_spawn_zs[0]
        )
        self._set_actor_pose_rel_xy(
            self.oranges[1], orange_xy[:, 1], self.orange_spawn_zs[1]
        )

    def _sample_plate_xy(self, b):
        plate_xy = torch.zeros((b, 2, 2), device=self.device)
        centers = torch.tensor(self.plate_x_centers, device=self.device).expand(b, -1).clone()
        # Plate identity is stable: brown is always at the first center and white at
        # the second. Only the small within-position jitter below is randomized.
        plate_xy[:, :, 0] = centers
        plate_xy[:, :, 0] += (
            torch.rand((b, 2), device=self.device) * 2 - 1
        ) * self.plate_x_noise
        plate_xy[:, :, 1] = (
            torch.rand((b, 2), device=self.device) * 2 - 1
        ) * self.plate_y_noise
        plate_dist = torch.linalg.norm(plate_xy[:, 0] - plate_xy[:, 1], axis=1)
        too_close = plate_dist < self.min_plate_dist
        if too_close.any():
            plate_xy[too_close, 0, 0] = centers[too_close, 0]
            plate_xy[too_close, 1, 0] = centers[too_close, 1]
            plate_xy[too_close, :, 1] = 0
        return plate_xy

    def _sample_fruit_side_xy(self, b, side, plate_xy):
        fruit_xy = torch.zeros((b, 2, 2), device=self.device)
        arm_xy = torch.tensor([0.0, 0.68 * side], device=self.device)
        for fruit_idx in range(2):
            needs_sample = torch.ones((b,), dtype=torch.bool, device=self.device)
            for _ in range(30):
                sample_count = int(needs_sample.sum().item())
                if sample_count == 0:
                    break
                candidate = torch.zeros((sample_count, 2), device=self.device)
                candidate[:, 0] = (
                    torch.rand((sample_count,), device=self.device)
                    * (self.fruit_x_range[1] - self.fruit_x_range[0])
                    + self.fruit_x_range[0]
                )
                candidate[:, 1] = side * (
                    torch.rand((sample_count,), device=self.device)
                    * (self.fruit_abs_y_range[1] - self.fruit_abs_y_range[0])
                    + self.fruit_abs_y_range[0]
                )
                sampled_indices = torch.nonzero(needs_sample, as_tuple=False).flatten()
                far_from_plates = (
                    torch.linalg.norm(
                        candidate[:, None, :] - plate_xy[sampled_indices], axis=2
                    ).min(dim=1).values
                    > self.min_fruit_plate_dist
                )
                far_from_arm = (
                    torch.linalg.norm(candidate - arm_xy, axis=1)
                    > self.min_fruit_arm_dist
                )
                if fruit_idx == 0:
                    far_from_fruit = torch.ones_like(far_from_arm)
                else:
                    far_from_fruit = (
                        torch.linalg.norm(
                            candidate - fruit_xy[sampled_indices, 0], axis=1
                        )
                        > self.min_fruit_fruit_dist
                    )
                accepted = far_from_plates & far_from_arm & far_from_fruit
                accepted_indices = sampled_indices[accepted]
                fruit_xy[accepted_indices, fruit_idx] = candidate[accepted]
                needs_sample[accepted_indices] = False
            if needs_sample.any():
                sample_count = int(needs_sample.sum().item())
                fruit_xy[needs_sample, fruit_idx, 0] = (
                    torch.rand((sample_count,), device=self.device)
                    * (self.fruit_x_range[1] - self.fruit_x_range[0])
                    + self.fruit_x_range[0]
                )
                fruit_xy[needs_sample, fruit_idx, 1] = side * self.fruit_abs_y_range[0]
        return fruit_xy

    def _set_actor_pose_rel_xy(self, actor, rel_xy, rel_z, q=None):
        p = self.workspace_offset_tensor.expand(rel_xy.shape[0], -1).clone()
        p[:, :2] += rel_xy
        p[:, 2] += rel_z
        actor.set_pose(Pose.create_from_pq(p=p, q=q))

    def _fruit_on_plate(self, fruit, plate, plate_radius):
        dist = torch.linalg.norm(fruit.pose.p[:, :2] - plate.pose.p[:, :2], axis=1)
        near_z = fruit.pose.p[:, 2] < plate.pose.p[:, 2] + 0.13
        return (dist < plate_radius * 0.8) & near_z

    def _bread_face_up(self, bread):
        """Require the bread's visible serving face to remain pointed upward.

        RoboTwin bread base0's braided face is normal to local +Y (not local +Z).
        """
        q = bread.pose.q
        local_y_dot_world_z = 2.0 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        return local_y_dot_world_z > 0.85

    def evaluate(self):
        plate_flags = []
        info = {}
        for i, plate in enumerate(self.plates):
            breads_on_plate = torch.stack(
                [
                    self._fruit_on_plate(bread, plate, self.plate_radii[i])
                    for bread in self.breads
                ],
                dim=1,
            )
            oranges_on_plate = torch.stack(
                [
                    self._fruit_on_plate(orange, plate, self.plate_radii[i])
                    for orange in self.oranges
                ],
                dim=1,
            )
            has_bread = torch.any(breads_on_plate, dim=1)
            has_orange = torch.any(oranges_on_plate, dim=1)
            exactly_one_bread = breads_on_plate.sum(dim=1) == 1
            exactly_one_orange = oranges_on_plate.sum(dim=1) == 1
            plate_ok = has_bread & has_orange & exactly_one_bread & exactly_one_orange
            plate_flags.append(plate_ok)
            info[f"plate_{i}_has_bread"] = has_bread
            info[f"plate_{i}_has_orange"] = has_orange
        all_plates_ready = torch.stack(plate_flags, dim=1).all(dim=1)
        breads_face_up = torch.stack(
            [self._bread_face_up(bread) for bread in self.breads], dim=1
        ).all(dim=1)
        food_static = torch.stack(
            [
                food.is_static(lin_thresh=0.015, ang_thresh=0.5)
                for food in (*self.breads, *self.oranges)
            ],
            dim=1,
        ).all(dim=1)
        success = all_plates_ready & breads_face_up & food_static & self._both_static()
        info.update(
            success=success,
            all_plates_ready=all_plates_ready,
            breads_face_up=breads_face_up,
            food_static=food_static,
        )
        return info
