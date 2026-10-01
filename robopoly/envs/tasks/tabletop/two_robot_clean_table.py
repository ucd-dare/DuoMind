from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from robopoly.agents.multi_agent import MultiAgent
from robopoly.agents.robots.panda import Panda
from robopoly.envs.sapien_env import BaseEnv
from robopoly.envs.utils import randomization
from robopoly.sensors.camera import CameraConfig
from robopoly.utils import sapien_utils
from robopoly.utils.building import actors
from robopoly.utils.registration import register_env
from robopoly.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from robopoly.utils.scene_builder.table import TableSceneBuilder
from robopoly.utils.structs.pose import Pose
from robopoly.utils.structs.types import GPUMemoryConfig, SimConfig


class TwoRobotCleanTableEnv(BaseEnv):
    """
    **Task Description:**
    Two Franka/Panda arms cooperate to clean a tabletop. Several small objects
    start on the left/center side of the table. A basket is fixed on the right
    side, outside the left robot's direct working area, so the robots need to
    pass or relay objects to put every item into the basket.

    One red cube starts in front of the LEFT arm and one green cube starts in
    front of the RIGHT arm. The left arm relays red to the table middle while
    the right arm places green in the basket; the right arm then places the
    relayed red cube in the basket.

    **Success Conditions:**
    - every cleanup object center is inside the basket footprint
    - every cleanup object has been dropped low enough to be considered inside
      the basket
    - both arms are approximately static
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]
    agent: MultiAgent[Tuple[Panda, Panda]]

    basket_center_xyz = (-0.46, 0.62, 0.015)
    basket_half_size_xy = (0.18, 0.16)
    basket_wall_height = 0.13
    basket_wall_thickness = 0.015
    basket_success_height = 0.18
    item_half_size = 0.032 * 0.8
    cube_friction = 10.0
    gripper_friction = 10.0

    # Per-side cube spawn x-range (workspace-relative). The basket footprint is at
    # x in [-0.64, -0.28], so the RIGHT cubes (which start next to the basket) are
    # kept to x >= -0.05 -- a >=0.23 m gap from the basket edge -- so the arm never
    # grazes the basket wall while grasping/carrying them. Left cubes are far from
    # the basket and keep the full range.
    left_item_x_range = (-0.28, 0.28)
    right_item_x_range = (-0.05, 0.36)
    # Keep both cube groups close to the table centerline and away from the
    # robot bases at y = +/-0.68. In particular, move the right pair farther
    # from the right robot base, where near-base spawns interfere with evaluation.
    left_item_y_range = (-0.28, -0.14)
    right_item_y_range = (0.14, 0.28)
    right_item_min_separation = 0.12

    # Each cleanup cube: (name, rgba, side, initial xy). ``side`` selects the
    # spawn y-range at reset ("left" -> in front of agents[0], "right" -> in
    # front of agents[1]).
    ITEMS = [
        ("cleanup_cube_red", [0.9, 0.1, 0.08, 1], "left", (-0.28, -0.3)),
        ("cleanup_cube_green", [0.05, 0.9, 0.08, 1], "right", (0.04, -0.16)),
    ]

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
    def _default_sensor_configs(self):
        left_pose = sapien_utils.look_at(
            eye=[0.65, -1.05, 0.75], target=[-0.05, -0.05, 0.08]
        )
        right_pose = sapien_utils.look_at(
            eye=[0.65, 1.05, 0.75], target=[0.0, 0.25, 0.08]
        )
        top_pose = sapien_utils.look_at(
            eye=[0.75, -0.65, 1.55], target=[0.0, 0.06, 0.02]
        )
        visualization_pose = sapien_utils.look_at(
            eye=[1.05, 0.0, 0.55], target=[0.0, 0.0, 0.06]
        )
        global_pose = sapien_utils.look_at(
            eye=[1.25, 0.0, 1.0053], target=[-0.10, 0.0, 0.06]
        )
        return [
            CameraConfig("left_side_camera", left_pose, 256, 256, np.pi / 2, 0.01, 100),
            CameraConfig(
                "right_side_camera", right_pose, 256, 256, np.pi / 2, 0.01, 100
            ),
            CameraConfig("overview_camera", top_pose, 1024, 768, 0.85, 0.01, 100),
            CameraConfig("global_camera", global_pose, 512, 384, 0.70, 0.01, 100),
            # Preserve the former global view for visualization/debugging only.
            # Dataset re-rendering, conversion, training, and eval select
            # global_camera and the two wrist cameras explicitly.
            CameraConfig(
                "visualization_camera",
                visualization_pose,
                512,
                384,
                0.90,
                0.01,
                100,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.05, 0.85, 0.85], target=[0.0, 0.05, 0.08])
        return CameraConfig("render_camera", pose, 768, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -0.68, 0]), sapien.Pose(p=[0, 0.68, 0])]
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.items = []
        self.item_sides = []
        for name, color, side, (ix, iy) in self.ITEMS:
            self.items.append(
                actors.build_cube(
                    self.scene,
                    half_size=self.item_half_size,
                    color=color,
                    name=name,
                    initial_pose=sapien.Pose(p=[ix, iy, self.item_half_size]),
                )
            )
            self.item_sides.append(side)
        self._build_basket()
        self._apply_task_contact_materials()

    def _apply_task_contact_materials(self):
        """Give clean-table cubes and Panda fingers high-friction contact."""
        cube_material = sapien.physx.PhysxMaterial(
            static_friction=self.cube_friction,
            dynamic_friction=self.cube_friction,
            restitution=0.0,
        )
        for item in self.items:
            for body in item._bodies:
                for shape in body.collision_shapes:
                    shape.set_physical_material(cube_material)

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

    def _build_basket(self):
        cx, cy, cz = self.basket_center_xyz
        hx, hy = self.basket_half_size_xy
        t = self.basket_wall_thickness
        wall_z = self.basket_wall_height / 2
        color = [0.25, 0.25, 0.28, 0.55]
        bottom_color = [0.12, 0.12, 0.14, 0.75]

        self.basket_parts = [
            actors.build_box(
                self.scene,
                half_sizes=[hx, hy, t / 2],
                color=bottom_color,
                name="basket_bottom",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx, cy, cz]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, t / 2, self.basket_wall_height / 2],
                color=color,
                name="basket_front_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx, cy - hy, wall_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[hx + t, t / 2, self.basket_wall_height / 2],
                color=color,
                name="basket_back_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx, cy + hy, wall_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[t / 2, hy, self.basket_wall_height / 2],
                color=color,
                name="basket_left_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx - hx, cy, wall_z]),
            ),
            actors.build_box(
                self.scene,
                half_sizes=[t / 2, hy, self.basket_wall_height / 2],
                color=color,
                name="basket_right_wall",
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[cx + hx, cy, wall_z]),
            ),
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            right_item_xy = []

            for i, item in enumerate(self.items):
                if self.item_sides[i] == "left":
                    y_lo, y_hi = self.left_item_y_range
                    x_lo, x_hi = self.left_item_x_range
                else:
                    y_lo, y_hi = self.right_item_y_range
                    x_lo, x_hi = self.right_item_x_range
                xy = torch.rand((b, 2), device=self.device)
                xy[:, 0] = xy[:, 0] * (x_hi - x_lo) + x_lo
                xy[:, 1] = xy[:, 1] * (y_hi - y_lo) + y_lo
                if self.item_sides[i] == "right" and right_item_xy:
                    previous_xy = torch.stack(right_item_xy, dim=1)
                    too_close = torch.linalg.vector_norm(
                        xy[:, None, :] - previous_xy, dim=-1
                    ).lt(self.right_item_min_separation).any(dim=1)
                    for _ in range(32):
                        if not too_close.any():
                            break
                        count = int(too_close.sum().item())
                        replacement = torch.rand((count, 2), device=self.device)
                        replacement[:, 0] = replacement[:, 0] * (x_hi - x_lo) + x_lo
                        replacement[:, 1] = replacement[:, 1] * (y_hi - y_lo) + y_lo
                        xy[too_close] = replacement
                        too_close = torch.linalg.vector_norm(
                            xy[:, None, :] - previous_xy, dim=-1
                        ).lt(self.right_item_min_separation).any(dim=1)
                    if too_close.any():
                        first_xy = previous_xy[:, 0]
                        xy[too_close, 0] = torch.where(
                            first_xy[too_close, 0] < (x_lo + x_hi) / 2,
                            x_hi,
                            x_lo,
                        )
                        xy[too_close, 1] = torch.where(
                            first_xy[too_close, 1] < (y_lo + y_hi) / 2,
                            y_hi,
                            y_lo,
                        )
                if self.item_sides[i] == "right":
                    right_item_xy.append(xy)
                qs = randomization.random_quaternions(
                    b, lock_x=True, lock_y=True, lock_z=False
                )
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = xy
                xyz[:, 2] = self.item_half_size
                item.set_pose(Pose.create_from_pq(xyz, qs))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def basket_center(self):
        return torch.tensor(self.basket_center_xyz, device=self.device).expand(
            self.num_envs, -1
        )

    def _items_in_basket(self):
        center = self.basket_center
        success_height = self.basket_success_height + getattr(
            self, "workspace_offset", (0.0, 0.0, 0.0)
        )[2]
        flags = []
        for item in self.items:
            delta_xy = torch.abs(item.pose.p[:, :2] - center[:, :2])
            in_xy = (delta_xy[:, 0] < self.basket_half_size_xy[0]) & (
                delta_xy[:, 1] < self.basket_half_size_xy[1]
            )
            low_enough = item.pose.p[:, 2] < success_height
            flags.append(in_xy & low_enough)
        return torch.stack(flags, dim=1)

    def evaluate(self):
        items_in_basket = self._items_in_basket()
        num_items_in_basket = items_in_basket.sum(dim=1)
        # The task succeeds only when both declared cubes are in the basket.
        all_items_in_basket = num_items_in_basket == len(self.items)
        left_static = self.left_agent.is_static(0.2)
        right_static = self.right_agent.is_static(0.2)
        success = all_items_in_basket & left_static & right_static
        info = {
            "success": success,
            "all_items_in_basket": all_items_in_basket,
            "left_arm_static": left_static,
            "right_arm_static": right_static,
            "num_items_in_basket": num_items_in_basket,
            "required_num_items_in_basket": len(self.items),
        }
        for i in range(len(self.items)):
            info[f"item_{i}_in_basket"] = items_in_basket[:, i]
        return info

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
            basket_center=self.basket_center,
        )
        if "state" in self.obs_mode:
            for i, item in enumerate(self.items):
                obs.update(
                    {
                        f"item_{i}_pose": item.pose.raw_pose,
                        f"left_arm_tcp_to_item_{i}_pos": item.pose.p
                        - self.left_agent.tcp.pose.p,
                        f"right_arm_tcp_to_item_{i}_pos": item.pose.p
                        - self.right_agent.tcp.pose.p,
                        f"item_{i}_to_basket_pos": self.basket_center - item.pose.p,
                    }
                )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        item_positions = torch.stack([item.pose.p for item in self.items], dim=1)
        center = self.basket_center[:, None, :]
        item_to_basket = torch.linalg.norm(
            item_positions[..., :2] - center[..., :2], axis=2
        )
        basket_progress = 1 - torch.tanh(3 * item_to_basket)

        left_dists = torch.linalg.norm(
            item_positions - self.left_agent.tcp.pose.p[:, None, :], axis=2
        )
        right_dists = torch.linalg.norm(
            item_positions - self.right_agent.tcp.pose.p[:, None, :], axis=2
        )
        reach_reward = 1 - torch.tanh(5 * torch.minimum(left_dists, right_dists))

        items_in_basket = self._items_in_basket()
        per_item_reward = 0.25 * reach_reward + 0.75 * basket_progress
        per_item_reward[items_in_basket] = 2.0
        reward = per_item_reward.mean(dim=1)
        reward += info["num_items_in_basket"] / len(self.items)

        static_reward = (
            1
            - torch.tanh(
                5 * torch.linalg.norm(self.left_agent.robot.get_qvel()[..., :-2], axis=1)
            )
            + 1
            - torch.tanh(
                5
                * torch.linalg.norm(
                    self.right_agent.robot.get_qvel()[..., :-2], axis=1
                )
            )
        ) / 2
        reward[info["all_items_in_basket"]] += static_reward[
            info["all_items_in_basket"]
        ]
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0


@register_env(
    "TwoRobotCleanTableReplicaCAD-v1",
    max_episode_steps=200,
    asset_download_ids=["ReplicaCAD"],
)
class TwoRobotCleanTableReplicaCADEnv(TwoRobotCleanTableEnv):
    """Two-robot clean-table task staged inside a ReplicaCAD apartment."""

    replicacad_build_config_idx = 0
    workspace_offset = (1.4, -1.2, 0.9196429)
    hidden_replicacad_name_substrings = (
        "frl_apartment_table",
        "_cabinet-0",
        "frl_apartment_rug_01",
        "frl_apartment_rug_02",
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
        # Aim 12 cm toward the far side so distal arm reaches stay visible.
        # Relative to that target, the 1.52 m forward and 1.0643 m vertical
        # offsets retain a 35-degree downward view. The modestly wider FOV
        # includes both robot bases without returning to the legacy wide shot.
        global_pose = sapien_utils.look_at(
            eye=(offset + np.array([1.40, 0.0, 1.1643])).tolist(),
            target=(target + np.array([-0.12, 0.0, 0.0])).tolist(),
        )
        room_pose = sapien_utils.look_at(
            eye=(offset + np.array([2.2, -2.6, 1.8])).tolist(),
            target=target.tolist(),
        )
        front_room_pose = sapien_utils.look_at(
            eye=(offset + np.array([-2.3, -2.0, 1.55])).tolist(),
            target=target.tolist(),
        )
        side_room_pose = sapien_utils.look_at(
            eye=(offset + np.array([-1.4, 1.7, 2.1])).tolist(),
            target=target.tolist(),
        )
        return [
            CameraConfig("left_side_camera", left_pose, 256, 256, np.pi / 2, 0.01, 100),
            CameraConfig(
                "right_side_camera", right_pose, 256, 256, np.pi / 2, 0.01, 100
            ),
            CameraConfig("overview_camera", top_pose, 1024, 768, 0.92, 0.01, 100),
            CameraConfig("global_camera", global_pose, 512, 384, 0.70, 0.01, 100),
            # Exact legacy global-camera pose and intrinsics, retained only for
            # future visualization/debugging.
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
            CameraConfig(
                "front_room_camera",
                front_room_pose,
                1024,
                768,
                0.95,
                0.01,
                100,
            ),
            CameraConfig(
                "side_room_camera",
                side_room_pose,
                1024,
                768,
                0.95,
                0.01,
                100,
            ),
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

    @property
    def basket_center(self):
        return (
            torch.tensor(self.basket_center_xyz, device=self.device)
            + torch.tensor(self.workspace_offset, device=self.device)
        ).expand(self.num_envs, -1)

    def _load_scene(self, options: dict):
        self.replicacad_scene = ReplicaCADSceneBuilder(self)
        self.replicacad_scene.build(self.replicacad_build_config_idx)
        # ReplicaCAD's loose decor is background-only in this task. Its original
        # supporting furniture overlaps the replacement task table and must stay
        # hidden; leaving the decor dynamic therefore makes it fall or explode.
        # Pin only these pre-existing background props at their authored poses.
        # Task cubes are built later by ``super`` and remain fully dynamic.
        seen = set()
        for actor in self.replicacad_scene.movable_objects.values():
            if id(actor) in seen:
                continue
            seen.add(id(actor))
            for body in actor._bodies:
                body.kinematic = True
        self._kinematic_replicacad_background_props = len(seen)
        super()._load_scene(options)
        self._build_plain_workspace_table()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self._reset_replicacad_background()
        self._hide_replicacad_workspace_furniture()
        offset_np = np.array(self.workspace_offset)
        offset = torch.tensor(self.workspace_offset, device=self.device)

        self.table_scene.table.set_pose(sapien.Pose(p=[0, 0, -100]))
        self._set_plain_workspace_table_pose()
        self.left_agent.robot.set_pose(
            sapien.Pose(
                p=(offset_np + np.array([0.0, -0.68, 0.0])).tolist(),
                q=euler2quat(0, 0, np.pi / 2),
            )
        )
        self.right_agent.robot.set_pose(
            sapien.Pose(
                p=(offset_np + np.array([0.0, 0.68, 0.0])).tolist(),
                q=euler2quat(0, 0, -np.pi / 2),
            )
        )

        for actor in self.items:
            actor.set_pose(Pose.create_from_pq(p=actor.pose.p + offset, q=actor.pose.q))
        self._set_basket_pose()

    def _reset_replicacad_background(self):
        """Restore loose ReplicaCAD props instead of removing them.

        Motion-planning generation resets the environment many times. The stock
        ReplicaCAD initializer assumes a single Fetch robot and is not usable in
        this two-Panda scene, so restore the builder's recorded object poses and
        velocities explicitly. This keeps all background decor visible while
        preventing motion from one attempt carrying into the next.
        """
        from robopoly.utils.structs import Articulation

        seen = set()
        for obj, pose in self.replicacad_scene._default_object_poses:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)
        self._restored_replicacad_background_objects = len(seen)

    def _hide_replicacad_workspace_furniture(self):
        # Keep hidden objects far apart. Placing several dynamic actors at the
        # same below-scene pose makes them collide and launch back into view.
        hidden_idx = 0
        for actor_name, actor in self.replicacad_scene.scene_objects.items():
            if any(s in actor_name for s in self.hidden_replicacad_name_substrings):
                actor.set_pose(sapien.Pose(p=[hidden_idx * 20.0, 0, -100]))
                hidden_idx += 1

    def _build_plain_workspace_table(self):
        self.plain_table_top = actors.build_box(
            self.scene,
            half_sizes=[0.75, 1.10, 0.035],
            color=[0.72, 0.36, 0.16, 1],
            name="replicacad_clean_plain_table_top",
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
                    name=f"replicacad_clean_plain_table_leg_{i}",
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

    def _set_basket_pose(self):
        cx, cy, cz = self.basket_center_xyz
        hx, hy = self.basket_half_size_xy
        t = self.basket_wall_thickness
        wall_z = self.basket_wall_height / 2
        offset = torch.tensor(self.workspace_offset, device=self.device)
        rel_positions = torch.tensor(
            [
                [cx, cy, cz],
                [cx, cy - hy, wall_z],
                [cx, cy + hy, wall_z],
                [cx - hx, cy, wall_z],
                [cx + hx, cy, wall_z],
            ],
            device=self.device,
        )
        for actor, rel_pos in zip(self.basket_parts, rel_positions):
            p = (rel_pos + offset).expand(self.num_envs, -1)
            actor.set_pose(Pose.create_from_pq(p=p))
