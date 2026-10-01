from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from robopoly import PACKAGE_ASSET_DIR
from robopoly.agents.multi_agent import MultiAgent
from robopoly.agents.robots.panda import Panda
from robopoly.envs.sapien_env import BaseEnv
from robopoly.sensors.camera import CameraConfig
from robopoly.utils import sapien_utils
from robopoly.utils.building import actors
from robopoly.utils.registration import register_env
from robopoly.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from robopoly.utils.scene_builder.table import TableSceneBuilder
from robopoly.utils.structs.actor import Actor
from robopoly.utils.structs.pose import Pose
from robopoly.utils.structs.types import GPUMemoryConfig, SimConfig


class TwoRobotCookPotEnv(BaseEnv):
    """
    **Task Description:**
    Two Franka/Panda arms cooperate to cook with a pot. A lidded pot and a
    square target platform start near the table center, separated along the
    line perpendicular to the two robots. A carrot starts in front of one
    randomly selected robot. The intended sequence is to open the lid, put the
    carrot in the pot, close the lid, then lift the pot onto the target platform
    with both arms.

    **Success Conditions:**
    - the lid has been opened at least once
    - the carrot is inside the pot footprint
    - the lid is closed on the pot
    - both robots are grasping the pot
    - the pot is centered over the red/white target
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]
    agent: MultiAgent[Tuple[Panda, Panda]]

    goal_radius = 0.15
    pot_radius = 0.075
    # The (flipped) pot mesh spans z [-0.0815, +0.0856] about its origin, so resting
    # it with the base on the table puts the origin at +0.0815.
    pot_half_height = 0.0815
    pot_scale = 0.25
    # Lid open/closed are now measured from the (decoupled) lid actor's pose
    # relative to the pot body, in metres. "Open" = the lid has been removed from
    # the pot mouth (lifted clear above the rim, or set aside off to the side);
    # "closed" = the lid is seated back on the pot, centred and at resting height.
    lid_open_lift_z = 0.05
    lid_open_xy = 0.07
    lid_closed_xy = 0.06       # the lid is a free body placed by the gripper; this
                               # tolerance accepts it visually covering the mouth
    lid_closed_dz = 0.025
    meat_half_size = 0.025
    kitchen_object_group = "carrot"
    kitchen_object_name = "carrot"
    carrot_scale = 1.2
    pot_station_offset_x = 0.16
    target_station_offset_x = 0.28
    station_center_y = 0.0
    # Pot and target are deterministic. Only the carrot POSITION is randomized,
    # in a compact left-arm region with clearance from the left base, pot, and
    # target; its orientation is fixed below.
    station_y_noise = 0.0
    meat_x_bounds = (-0.05, 0.08)
    meat_abs_y_bounds = (0.30, 0.36)
    meat_min_pot_dist = 0.24
    meat_min_target_dist = 0.22
    target_square_half_size = 0.18
    target_square_thickness = 0.05

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
            eye=[0.65, -1.05, 0.75], target=[0.0, -0.05, 0.08]
        )
        right_pose = sapien_utils.look_at(
            eye=[0.65, 1.05, 0.75], target=[0.0, 0.05, 0.08]
        )
        top_pose = sapien_utils.look_at(
            eye=[0.75, -0.60, 1.55], target=[0.0, 0.0, 0.02]
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
            CameraConfig(
                "visualization_camera", visualization_pose, 512, 384, 0.90, 0.01, 100
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[1.05, 0.85, 0.85], target=[0.0, 0.0, 0.10]
        )
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

        # The pot body and lid are built as TWO SEPARATE free rigid actors (the
        # original URDF joins them with a prismatic slider whose 0.25-scaled travel
        # is only 1.5 cm -- too little to ever open the lid). Decoupling them lets a
        # robot lift the lid clear off the pot, drop the carrot in from the top, and
        # set the lid back on. ``pot_body_link`` / ``lid_link`` keep their names so
        # the rest of the task code is unchanged.
        self.pot_body, self.lid = self._build_pot()
        self.pot_body_link = self.pot_body
        self.lid_link = self.lid
        self.meat = self._build_carrot()
        self.target_square = actors.build_box(
            self.scene,
            half_sizes=[
                self.target_square_half_size,
                self.target_square_half_size,
                self.target_square_thickness / 2,
            ],
            color=[0.18, 0.28, 0.32, 1],
            name="target_square",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, self.target_square_thickness / 2]),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=min(self.goal_radius, self.target_square_half_size),
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

    # Densities for the decoupled actors. The default URDF density (1000) makes the
    # convex-decomposed body ~21 kg, far past what two Panda arms can lift; a lighter
    # body keeps the coordinated two-arm lift feasible.
    pot_body_density = 80.0
    lid_density = 200.0        # heavy enough that a grasp approach doesn't flick it
    # All offsets below are relative to the pot origin and aligned to the (flipped)
    # VISUAL mesh: base -0.0815, rim +0.0856, side handle bars at z +0.079 / |y| 0.21
    # (running along x), mouth opening radius ~0.147.
    pot_base_z = -0.0815       # mesh base (rests on the table)
    pot_rim_z = 0.0856         # mesh rim / mouth top
    pot_lift = 0.0             # >0 sinks the base into the table (lowers the whole pot)
    well_inner = 0.125         # well wall inner half-extent (contains the carrot)
    well_thick = 0.013
    # Pot-body handle bars (primitive collision) -> placed ON the visual handle loops.
    handle_off_x = 0.0
    handle_off_y = 0.205       # |Δy| of the handle bar centre (on the loop's outer bar)
    handle_off_z = 0.079       # handle-bar centre z (matches the visual handle bar)
    handle_hx = 0.050          # long in x (matches the visual handle bar)
    handle_hy = 0.015          # thin in y -> graspable across y
    handle_hz = 0.020
    # Lid primitive-collision geometry (relative to the lid origin).
    lid_plate_z = 0.093        # plate centre z (bottom rests just on the rim)
    lid_plate_half = 0.135     # plate half-width: rests on walls, clears handle posts
    lid_plate_thick = 0.005    # plate half-thickness
    lid_visual_dz = 0.015      # raise lid mesh so its underside sits ON the rim
    lid_visual_scale = 1.10    # widen the lid mesh so its rim meets the pot rim
    # the lid's visual handle is a bar LONG in x, THIN in y -> match it so the gripper
    # grips across y (aligned with the bar's long side).
    lid_bar_z = 0.125          # handle-bar centre z (clearly above the lid plate)
    lid_bar_hx = 0.045         # bar half-length along x (long side, grippable centre)
    lid_bar_hy = 0.012         # bar half-size across y (grippable width)
    lid_bar_hz = 0.018

    def _build_pot(self):
        """Build the pot body and lid as two independent free rigid actors.

        The URDF links carry their meshes in link-local frames joined by fixed
        (base->link_1) and prismatic (link_1->link_0) joints. We read each link's
        mesh-file records and rebuild them as standalone actors, baking the
        link-to-base rotation into each shape's pose so that, with the actor placed
        at the pot xyz, the geometry lands exactly where the original articulation
        put it (lid resting on the body).
        """
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.scale = self.pot_scale
        loader.load_multiple_collisions_from_file = True
        model_dir = PACKAGE_ASSET_DIR / "robotwin/objects/060_kitchenpot/100015"
        urdf_path = model_dir / "mobility.urdf"
        art = loader.parse(str(urdf_path), package_dir=str(model_dir))[
            "articulation_builders"
        ][0]
        link_builders = {lb.name: lb for lb in art.link_builders}
        # The URDF base->link_1 rotation leaves the pot's mouth + handles at the
        # BOTTOM (the dense rim/handle end faces down). Flip the body mesh 180 deg so
        # the mouth and the two side handles are both at the TOP -- matching the
        # open-top collision well, which always had them up. The lid mesh is laid FLAT
        # (90 deg about x) so it caps the mouth horizontally at the rim.
        q_body = sapien.Pose(q=[0.5, -0.5, 0.5, -0.5])
        q_body_up = sapien.Pose(q=euler2quat(np.pi, 0, 0)) * q_body  # mouth+handles up
        q_lid = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))  # lid laid flat on the mouth

        pot_body = self._build_pot_body_actor(link_builders["link_1"], q_body_up)
        lid = self._build_lid_actor(link_builders["link_0"], q_lid)
        return pot_body, lid

    def _build_pot_body_actor(self, body_link_builder, frame_pose):
        """Build the pot body with the full mesh as VISUAL but SIMPLE primitive
        collision: a floor, four walls forming an open-top square well (contains the
        carrot, gives the lid plate a flat rim to rest on and lift straight off), and
        two side handle bars for the two-arm lift. All offsets are relative to the pot
        origin (which sits at ``pot_half_height`` with the well floor on the table).
        """
        builder = self.scene.create_actor_builder()
        for rec in body_link_builder.visual_records:
            builder.add_visual_from_file(
                filename=rec.filename, pose=frame_pose * rec.pose, scale=rec.scale
            )
        d = self.pot_body_density
        wi, wt = self.well_inner, self.well_thick
        floor_b = self.pot_base_z + self.pot_lift  # floor bottom (rests on the table)
        floor_top = floor_b + 0.024   # 0.012 half-thickness floor
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, floor_b + 0.012]),
            half_size=[wi + wt, wi + wt, 0.012], density=d,
        )
        # four walls of the well, from just above the floor up to the rim.
        wz = (floor_top + self.pot_rim_z) / 2
        whz = (self.pot_rim_z - floor_top) / 2
        for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if sx:
                pose = sapien.Pose(p=[sx * (wi + wt), 0, wz])
                hs = [wt, wi + 2 * wt, whz]
            else:
                pose = sapien.Pose(p=[0, sy * (wi + wt), wz])
                hs = [wi + 2 * wt, wt, whz]
            builder.add_box_collision(pose=pose, half_size=hs, density=d)
        # two side handle bars (grasped for the lift)
        for sy in (1, -1):
            builder.add_box_collision(
                pose=sapien.Pose(p=[self.handle_off_x, sy * self.handle_off_y,
                                    self.handle_off_z]),
                half_size=[self.handle_hx, self.handle_hy, self.handle_hz], density=d,
            )
        builder.initial_pose = sapien.Pose(p=[0, 0, self.pot_half_height])
        return builder.build(name="kitchenpot_body")

    def _build_lid_actor(self, lid_link_builder, frame_pose):
        """Build the lid with the full mesh as VISUAL but SIMPLE primitive collision.

        The lid mesh's convex decomposition deeply overlaps the pot body, so once the
        slider joint is removed the two free bodies explode apart on contact. Instead
        we give the lid two clean collision primitives, both expressed relative to the
        lid origin (which coincides with the pot-body centre when seated):
          * a thin square plate that rests flat on the pot rim (rim top ≈ +0.065), and
          * a thin bar matching the lid's top handle, so a gripper can grasp it.
        The plate rests cleanly on the rim (a shallow surface contact, no deep
        interpenetration), so the seated lid is stable.
        """
        builder = self.scene.create_actor_builder()
        # raise the lid mesh so its flat underside sits ON the rim (at the collision
        # plate height), and widen it slightly so its rim meets the pot rim (the raw
        # lid is a touch smaller than the mouth and otherwise leaves a visible gap).
        vis_lift = sapien.Pose(p=[0, 0, self.lid_visual_dz])
        for rec in lid_link_builder.visual_records:
            builder.add_visual_from_file(
                filename=rec.filename, pose=vis_lift * frame_pose * rec.pose,
                scale=np.asarray(rec.scale) * self.lid_visual_scale,
            )
        # high-friction material so the gripper holds the thin handle bar firmly.
        grip_mat = sapien.physx.PhysxMaterial(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0
        )
        # resting plate (covers the mouth, sits on the rim ring)
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.lid_plate_z]),
            half_size=[self.lid_plate_half, self.lid_plate_half, self.lid_plate_thick],
            density=self.lid_density,
        )
        # graspable top handle bar (thin in x, long in y)
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.lid_bar_z]),
            half_size=[self.lid_bar_hx, self.lid_bar_hy, self.lid_bar_hz],
            density=self.lid_density,
            material=grip_mat,
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self.pot_half_height])
        return builder.build(name="kitchenpot_lid")

    def _build_carrot(self):
        from robopoly.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
            sample_kitchen_object,
        )
        from robopoly.utils.scene_builder.robocasa.objects.objects import MJCFObject

        try:
            carrot_kwargs, self.carrot_info = sample_kitchen_object(
                self.kitchen_object_group,
                obj_registries=("objaverse", "aigen"),
                object_scale=self.carrot_scale,
                rng=np.random.default_rng(0),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Could not find a RoboCasa {self.kitchen_object_group} asset. "
                "Install/download the RoboCasa asset bundle, then recreate the "
                "environment."
            ) from exc

        carrots = []
        self.carrot_spawn_z = None
        for i in range(self.num_envs):
            carrot = MJCFObject(
                self.scene, name=self.kitchen_object_name, **carrot_kwargs
            )
            if self.carrot_spawn_z is None:
                self.carrot_spawn_z = float(-carrot.bottom_offset[2] + 0.005)
            actor = carrot.build(scene_idxs=[i]).actor
            carrots.append(actor)
            self.remove_from_state_dict_registry(actor)

        carrot_actor = Actor.merge(carrots, name=self.kitchen_object_name)
        self.add_to_state_dict_registry(carrot_actor)
        return carrot_actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # Start both arms RETRACTED so the grippers rest back from the pot (not hovering
            # right over it) -- bases are unchanged; only the arm configuration is pulled back
            # (~13 cm toward the base, ~5 cm up) from the default Panda rest pose. This is a
            # symmetric pose (base yaw 0), so the same qpos mirrors to both arms.
            rest_back = torch.tensor(
                [0.0, -0.0295, 0.0, -2.4222, 0.0, 2.3927, 0.7854, 0.04, 0.04],
                device=self.device,
            )
            rest_qpos = rest_back.unsqueeze(0).repeat(b, 1)
            if self.robot_init_qpos_noise:
                rest_qpos[:, :7] = rest_qpos[:, :7] + (
                    torch.rand((b, 7), device=self.device) * 2 - 1
                ) * self.robot_init_qpos_noise
            self.left_agent.robot.set_qpos(rest_qpos)
            self.right_agent.robot.set_qpos(rest_qpos.clone())
            if not hasattr(self, "lid_opened_once"):
                self.lid_opened_once = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.lid_opened_once[env_idx] = False

            # Keep the scene ordering fixed relative to the global camera, whose
            # eye is on +x: pot on the far (-x) side and target on the near (+x)
            # side. Previously this sign was randomized, visually exchanging the
            # pot and target between episodes.
            pot_side = torch.full((b,), -1.0, device=self.device)
            pot_xyz = torch.zeros((b, 3), device=self.device)
            pot_xyz[:, 0] = pot_side * self.pot_station_offset_x
            pot_xyz[:, 1] = self.station_center_y
            pot_xyz[:, 1] += (
                torch.rand((b,), device=self.device) * 2 - 1
            ) * self.station_y_noise
            pot_xyz[:, 2] = self.pot_half_height
            pot_q = torch.tensor(euler2quat(0, 0, 0), device=self.device)
            # The lid rests on the body: both actors share the same root pose (the
            # original articulation reported identical link poses for body and lid).
            self.pot_body.set_pose(Pose.create_from_pq(pot_xyz, pot_q))
            self.lid.set_pose(Pose.create_from_pq(pot_xyz.clone(), pot_q))

            target_xyz = pot_xyz.clone()
            target_xyz[:, 0] = -pot_side * self.target_station_offset_x
            target_xyz[:, 2] = self.target_square_thickness / 2
            self.target_square.set_pose(Pose.create_from_pq(p=target_xyz))

            target_region_xyz = target_xyz.clone()
            target_region_xyz[:, 2] = self.target_square_thickness + 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            # Keep the carrot on the -y side so the left arm always performs the
            # carrot subtask. Its sampled distance from the left base is 0.32-0.38 m.
            meat_side = torch.full((b,), -1.0, device=self.device)
            meat_xyz = torch.zeros((b, 3), device=self.device)
            needs_sample = torch.ones((b,), dtype=torch.bool, device=self.device)
            for _ in range(20):
                sample_count = int(needs_sample.sum().item())
                if sample_count == 0:
                    break
                candidate_xy = torch.zeros((sample_count, 2), device=self.device)
                candidate_xy[:, 0] = (
                    torch.rand((sample_count,), device=self.device)
                    * (self.meat_x_bounds[1] - self.meat_x_bounds[0])
                    + self.meat_x_bounds[0]
                )
                candidate_xy[:, 1] = meat_side[needs_sample] * (
                    torch.rand((sample_count,), device=self.device)
                    * (self.meat_abs_y_bounds[1] - self.meat_abs_y_bounds[0])
                    + self.meat_abs_y_bounds[0]
                )
                far_from_pot = (
                    torch.linalg.norm(candidate_xy - pot_xyz[needs_sample, :2], axis=1)
                    > self.meat_min_pot_dist
                )
                far_from_target = (
                    torch.linalg.norm(
                        candidate_xy - target_xyz[needs_sample, :2], axis=1
                    )
                    > self.meat_min_target_dist
                )
                sampled_indices = torch.nonzero(needs_sample, as_tuple=False).flatten()
                accepted = far_from_pot & far_from_target
                accepted_indices = sampled_indices[accepted]
                meat_xyz[accepted_indices, :2] = candidate_xy[accepted]
                needs_sample[accepted_indices] = False
            if needs_sample.any():
                sample_count = int(needs_sample.sum().item())
                meat_xyz[needs_sample, 0] = (
                    torch.rand((sample_count,), device=self.device)
                    * (self.meat_x_bounds[1] - self.meat_x_bounds[0])
                    + self.meat_x_bounds[0]
                )
                meat_xyz[needs_sample, 1] = (
                    meat_side[needs_sample] * self.meat_abs_y_bounds[1]
                )
            meat_xyz[:, 2] = self.carrot_spawn_z
            # Only position is randomized; keep carrot orientation deterministic.
            meat_q = torch.tensor(
                euler2quat(0, 0, 0), dtype=torch.float32, device=self.device
            ).expand(b, -1)
            self.meat.set_pose(Pose.create_from_pq(meat_xyz, meat_q))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def pot_top_z(self):
        return self.pot_body_link.pose.p[:, 2] + self.pot_half_height

    def _lid_xy_offset(self):
        """Horizontal distance of the lid from the pot-body centre."""
        return torch.linalg.norm(
            self.lid.pose.p[:, :2] - self.pot_body.pose.p[:, :2], axis=1
        )

    def _lid_z_offset(self):
        """Signed height of the lid above its resting height on the pot body."""
        return self.lid.pose.p[:, 2] - self.pot_body.pose.p[:, 2]

    def _lid_is_open(self):
        """The lid has been taken off the pot: lifted clear above the rim, or set
        aside horizontally off the pot mouth."""
        return (self._lid_z_offset() > self.lid_open_lift_z) | (
            self._lid_xy_offset() > self.lid_open_xy
        )

    def _lid_is_closed(self):
        """The lid is seated back on the pot (centred over it, at resting height)."""
        return (self._lid_xy_offset() < self.lid_closed_xy) & (
            torch.abs(self._lid_z_offset()) < self.lid_closed_dz
        )

    def _meat_inside_pot(self):
        meat_xy_dist = torch.linalg.norm(
            self.meat.pose.p[:, :2] - self.pot_body_link.pose.p[:, :2], axis=1
        )
        meat_low_enough = (
            self.meat.pose.p[:, 2] < self.pot_body_link.pose.p[:, 2] + 0.08
        )
        meat_high_enough = (
            self.meat.pose.p[:, 2]
            > self.pot_body_link.pose.p[:, 2] - self.pot_half_height
        )
        return (
            (meat_xy_dist < self.pot_radius * 0.75)
            & meat_low_enough
            & meat_high_enough
        )

    def evaluate(self):
        lid_open = self._lid_is_open()
        if hasattr(self, "lid_opened_once"):
            self.lid_opened_once = self.lid_opened_once | lid_open
        else:
            self.lid_opened_once = lid_open
        meat_inside = self._meat_inside_pot()
        lid_closed = self._lid_is_closed()
        left_grasping_pot = self.left_agent.is_grasping(self.pot_body_link)
        right_grasping_pot = self.right_agent.is_grasping(self.pot_body_link)
        both_grasping_pot = left_grasping_pot & right_grasping_pot
        pot_delta_xy = torch.abs(
            self.pot_body_link.pose.p[:, :2] - self.target_square.pose.p[:, :2]
        )
        pot_on_target = (pot_delta_xy[:, 0] < self.target_square_half_size) & (
            pot_delta_xy[:, 1] < self.target_square_half_size
        )
        table_surface_z = getattr(self, "workspace_offset", (0.0, 0.0, 0.0))[2]
        pot_lifted = (
            self.pot_body_link.pose.p[:, 2]
            > table_surface_z
            + self.pot_half_height
            + self.target_square_thickness * 0.75
        )
        success = (
            self.lid_opened_once
            & meat_inside
            & lid_closed
            & both_grasping_pot
            & pot_on_target
            & pot_lifted
        )
        return {
            "success": success,
            "lid_open": lid_open,
            "lid_opened_once": self.lid_opened_once,
            "meat_inside_pot": meat_inside,
            "lid_closed": lid_closed,
            "left_grasping_pot": left_grasping_pot,
            "right_grasping_pot": right_grasping_pot,
            "both_grasping_pot": both_grasping_pot,
            "pot_on_target": pot_on_target,
            "pot_lifted": pot_lifted,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
            goal_region_pos=self.goal_region.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                pot_pose=self.pot_body_link.pose.raw_pose,
                lid_pose=self.lid_link.pose.raw_pose,
                meat_pose=self.meat.pose.raw_pose,
                left_arm_tcp_to_lid_pos=self.lid_link.pose.p
                - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_lid_pos=self.lid_link.pose.p
                - self.right_agent.tcp.pose.p,
                left_arm_tcp_to_meat_pos=self.meat.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_meat_pos=self.meat.pose.p
                - self.right_agent.tcp.pose.p,
                meat_to_pot_pos=self.pot_body_link.pose.p - self.meat.pose.p,
                lid_to_pot_pos=self.pot_body_link.pose.p - self.lid_link.pose.p,
                pot_to_goal_pos=self.goal_region.pose.p - self.pot_body_link.pose.p,
            )
            for key in [
                "lid_opened_once",
                "meat_inside_pot",
                "lid_closed",
                "both_grasping_pot",
                "pot_on_target",
                "pot_lifted",
            ]:
                obs[key] = info[key]
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        lid_reach_dist = torch.minimum(
            torch.linalg.norm(
                self.lid_link.pose.p - self.left_agent.tcp.pose.p, axis=1
            ),
            torch.linalg.norm(
                self.lid_link.pose.p - self.right_agent.tcp.pose.p, axis=1
            ),
        )
        meat_reach_dist = torch.minimum(
            torch.linalg.norm(self.meat.pose.p - self.left_agent.tcp.pose.p, axis=1),
            torch.linalg.norm(self.meat.pose.p - self.right_agent.tcp.pose.p, axis=1),
        )
        pot_reach_reward = (
            1
            - torch.tanh(
                5
                * torch.linalg.norm(
                    self.pot_body_link.pose.p - self.left_agent.tcp.pose.p, axis=1
                )
            )
            + 1
            - torch.tanh(
                5
                * torch.linalg.norm(
                    self.pot_body_link.pose.p - self.right_agent.tcp.pose.p, axis=1
                )
            )
        ) / 2

        lid_open_reward = 1 - torch.tanh(5 * lid_reach_dist)
        meat_place_dist = torch.linalg.norm(
            self.meat.pose.p[:, :2] - self.pot_body_link.pose.p[:, :2], axis=1
        )
        meat_place_reward = 1 - torch.tanh(5 * meat_place_dist)
        lid_close_dist = self._lid_xy_offset() + torch.abs(self._lid_z_offset())
        lid_close_reward = 1 - torch.tanh(5 * lid_close_dist)
        pot_to_goal_dist = torch.linalg.norm(
            self.pot_body_link.pose.p[:, :2] - self.goal_region.pose.p[:, :2], axis=1
        )
        pot_goal_reward = 1 - torch.tanh(5 * pot_to_goal_dist)

        reward = 0.5 * lid_open_reward
        reward[info["lid_opened_once"]] = 1.0 + 0.5 * (
            1 - torch.tanh(5 * meat_reach_dist[info["lid_opened_once"]])
        )
        reward[info["meat_inside_pot"]] = (
            2.0 + lid_close_reward[info["meat_inside_pot"]]
        )
        ready_to_lift = (
            info["lid_opened_once"] & info["meat_inside_pot"] & info["lid_closed"]
        )
        reward[ready_to_lift] = 4.0 + pot_reach_reward[ready_to_lift]
        reward[info["both_grasping_pot"]] = (
            6.0 + pot_goal_reward[info["both_grasping_pot"]]
        )
        placed = info["both_grasping_pot"] & info["pot_on_target"] & info["pot_lifted"]
        reward[placed] = 8.0
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0


@register_env(
    "TwoRobotCookPotReplicaCAD-v1",
    max_episode_steps=250,
    asset_download_ids=["ReplicaCAD"],
)
class TwoRobotCookPotReplicaCADEnv(TwoRobotCookPotEnv):
    """Two-robot cook-pot task staged inside a ReplicaCAD apartment scene."""

    replicacad_build_config_idx = 0
    workspace_offset = (1.4, -1.2, 0.9196429)
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
            CameraConfig(
                "visualization_camera", visualization_pose, 512, 384, 0.90, 0.01, 100
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

    def _load_scene(self, options: dict):
        self.replicacad_scene = ReplicaCADSceneBuilder(self)
        self.replicacad_scene.build(self.replicacad_build_config_idx)
        super()._load_scene(options)
        self._build_plain_workspace_table()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
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

        for actor in [self.meat, self.target_square, self.goal_region,
                      self.pot_body, self.lid]:
            actor.set_pose(Pose.create_from_pq(p=actor.pose.p + offset, q=actor.pose.q))

    def _hide_replicacad_workspace_furniture(self):
        hidden_pose = sapien.Pose(p=[0, 0, -100])
        for actor_name, actor in self.replicacad_scene.scene_objects.items():
            if any(s in actor_name for s in self.hidden_replicacad_name_substrings):
                actor.set_pose(hidden_pose)

    def _build_plain_workspace_table(self):
        self.plain_table_top = actors.build_box(
            self.scene,
            half_sizes=[0.75, 1.10, 0.035],
            color=[0.72, 0.36, 0.16, 1],
            name="replicacad_plain_table_top",
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
                    name=f"replicacad_plain_table_leg_{i}",
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
