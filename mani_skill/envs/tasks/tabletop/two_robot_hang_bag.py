import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import get_articulation_builder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .two_robot_kitchen_base import TwoRobotKitchenReplicaCADBaseEnv


@register_env(
    "TwoRobotHangBagReplicaCAD-v1",
    max_episode_steps=1500,
    asset_download_ids=["ReplicaCAD"],
)
class TwoRobotHangBagReplicaCADEnv(TwoRobotKitchenReplicaCADBaseEnv):
    """Two-arm collaborative bag hanging.

    The LEFT arm can only reach a soft bag (PartNet-Mobility "suitcase" 101673 --
    a body ``link_1`` with a rigid carry ``handle`` ``link_0`` on top), spawned in
    front of the left base. The RIGHT arm can only reach a fixed hook ("hooker")
    standing in front of and to the right of the left base. Because neither arm can
    reach both objects, the two arms must cooperate (left picks up the bag and hands
    it toward the centre, right takes it and drops the bag's handle loop over the
    hook peg) so the bag ends up hanging from the hook.

    Env modifications from a plain asset load (user requests):
      * The bag's handle joint (``joint_0``, a +/-90 deg revolute) is LOCKED at 0 so
        the handle stays upright/rigid at all times -- the arm always grabs, carries
        and hangs the same fixed loop.
      * The hook is built from simple boxes and is kinematic (fixed) so it can never
        fall or be pushed over while the bag is hung on it.
    """

    # PartNet-Mobility "suitcase" used as the bag. Imported exactly like the oven
    # (7220) / cabinet (19179): try the registered partnet-mobility builder first,
    # then fall back to loading the vendored URDF directly. Unlike those two the bag
    # is DYNAMIC (movable) -- the arms pick it up and carry it -- so its root link is
    # NOT fixed; only its handle joint is locked (see ``_lock_bag_handle``).
    bag_model_id = "101673"
    bag_scale = 0.25
    bag_friction = 10.0
    gripper_friction = 10.0
    # Bag spawn: in front of the LEFT base (rel -y), but shifted forward (+x, toward
    # the global camera) so the right arm has a cleaner handoff approach.
    # Yaw orients the bag so its handle "through" axis (the direction a peg threads the
    # handle loop, = the bag's local x) points along world x -- matching the hook peg,
    # which points toward the FRONT camera (+x). (env change 2026-07-21, user chose to
    # have the hook face the front: peg +x + bag yaw 0, so the arm threads the loop by
    # reaching forward and the hook's post is out of its path -- a peg toward the centre
    # put the post between the right arm and the seat, blocking every approach.)
    bag_center_rel = (0.28, -0.34)
    bag_yaw = 0.0
    bag_spawn_xy_noise = 0.025
    bag_min_arm_dist = 0.38
    bag_max_arm_dist = 0.50
    # Handle-joint friction/drive used to keep the handle rigidly upright.
    handle_lock_stiffness = 1e5
    handle_lock_damping = 1e3

    # Hook ("hooker"): fixed, shifted toward the right-arm side (+y) and farther from
    # the global camera (-x). The peg still faces the camera (+x) so the handle is
    # threaded toward the viewer, but the post itself sits behind the main arm workspace.
    hook_center_rel = (-0.24, 0.201)
    hook_post_half = (0.022, 0.022, 0.24)   # vertical post half-sizes
    hook_peg_len = 0.16                      # peg length (cantilever, along its axis)
    hook_peg_half_cross = 0.013              # peg cross-section half-size
    hook_peg_z = 0.46                        # peg root height above the table top
    hook_peg_tilt_deg = 20.0                 # single straight peg, tilted upward
    # Peg cantilevers along +x (toward the front camera); the right arm slides the bag's
    # handle loop onto it by moving in -x (from beyond the tip toward the post).
    hook_peg_axis = np.array([1.0, 0.0, 0.0])
    hook_color = [0.20, 0.22, 0.26, 1]

    # Bases remain unchanged. These joint homes simply park the end-effectors farther
    # apart at reset, with grippers fully open, so the handoff starts less crowded.
    home_arm_qpos_left = (-0.35, -0.35, 0.0, -2.15, 0.0, 1.95, np.pi / 4)
    home_arm_qpos_right = (0.35, -0.35, 0.0, -2.15, 0.0, 1.95, np.pi / 4)

    # Success tolerances (see ``evaluate``).
    # The handle-link origin is offset from the loop opening that physically rests
    # on the peg. This radius accepts the observed stable hanging pose while staying
    # far smaller than the distance to the middle/table placement region.
    hang_xy_radius = 0.14
    # When hung, the handle loop rests on the peg by its TOP bar, so the handle's
    # geometric centre sits ~one loop-half BELOW the peg. Accept a band that spans
    # from a little above the peg (loop bar just cresting it) to below it.
    hang_handle_z_above = 0.06
    hang_handle_z_below = 0.20
    bag_hang_min_clearance = 0.02   # bag bottom must sit above the table (hanging)

    def _load_task_scene(self, options: dict):
        self.bag = self._build_bag()
        self.hook = self._build_hook()
        self._apply_task_contact_materials()

    def _apply_task_contact_materials(self):
        """Give the bag and both Panda grippers high-friction contact."""
        bag_material = sapien.physx.PhysxMaterial(
            static_friction=self.bag_friction,
            dynamic_friction=self.bag_friction,
            restitution=0.0,
        )
        for link in self.bag.links:
            for body in link._bodies:
                for shape in body.collision_shapes:
                    shape.set_physical_material(bag_material)

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

    # ------------------------------------------------------------------ bag ----
    def _build_bag(self):
        try:
            builder = get_articulation_builder(
                self.scene, f"partnet-mobility:{self.bag_model_id}"
            )
        except Exception as exc:
            builder = self._get_direct_partnet_builder(exc)
        if builder is None:
            raise RuntimeError(
                f"Could not locate the PartNet-Mobility bag '{self.bag_model_id}'. "
                f"Expected a 'mobility.urdf' under "
                f"partnet_mobility/dataset/{self.bag_model_id} in either the package "
                f"assets or the downloaded asset directory."
            )
        builder.initial_pose = sapien.Pose(
            p=[0, 0, -100], q=euler2quat(0, 0, self.bag_yaw)
        )
        bag = builder.build(name=f"partnet_bag_{self.bag_model_id}")
        self.remove_from_state_dict_registry(bag)
        self.add_to_state_dict_registry(bag)
        self._lock_bag_handle(bag)
        return bag

    def _lock_bag_handle(self, bag):
        """Keep the carry handle (``joint_0``) rigidly upright. The handle is a
        +/-90 deg revolute; we hold it at 0 with a very stiff position drive plus
        high friction so it never flops while the bag is grabbed/carried/hung."""
        for joint in bag.active_joints:
            joint.set_friction(self.bag_friction)
            joint.set_drive_properties(
                self.handle_lock_stiffness, self.handle_lock_damping, mode="force"
            )
            joint.set_drive_target(0.0)

    def _get_direct_partnet_builder(self, original_exc):
        model_dirs = [
            PACKAGE_ASSET_DIR / "partnet_mobility/dataset" / self.bag_model_id,
            ASSET_DIR / "partnet_mobility/dataset" / self.bag_model_id,
        ]
        for model_dir in model_dirs:
            for urdf_name in (
                "mobility_cvx.urdf",
                "mobility_fixed.urdf",
                "mobility.urdf",
            ):
                urdf_path = model_dir / urdf_name
                if urdf_path.exists():
                    loader = self.scene.create_urdf_loader()
                    # DYNAMIC bag: the root link is free so the arms can carry it.
                    loader.fix_root_link = False
                    loader.scale = self.bag_scale
                    # mobility_cvx.urdf ships PRECOMPUTED convex-decomposition
                    # collision pieces (``col_body_*`` / ``col_handle_*``): the
                    # carry handle is a closed loop, so a single convex hull would
                    # fill its opening solid (nothing could grab it or thread the
                    # hook peg through it). The pieces were made with COACD offline
                    # (SAPIEN's runtime COACD wrapper crashes on this asset's
                    # meshes), so each collision mesh is already convex and is loaded
                    # as-is -- no runtime decomposition.
                    loader.load_multiple_collisions_from_file = True
                    sapien_utils.apply_urdf_config(
                        loader,
                        sapien_utils.parse_urdf_config(
                            dict(
                                material=dict(
                                    static_friction=self.bag_friction,
                                    dynamic_friction=self.bag_friction,
                                    restitution=0,
                                )
                            )
                        ),
                    )
                    return loader.parse(str(urdf_path))["articulation_builders"][0]
        return None

    def _select_bag_links(self):
        """Identify the handle link (child of the single revolute joint) and the
        body link (its parent). Run after reconfigure so link poses are valid."""
        handle_joint = self.bag.active_joints[0]
        self.bag_handle_link = handle_joint.child_link
        self.bag_body_link = handle_joint.parent_link

    # ----------------------------------------------------------------- hook ----
    def _build_hook(self):
        """A fixed (kinematic) hook: a base plate + vertical post + one straight peg
        tilted upward toward the camera. Built as one actor so it moves/holds as a
        rigid, immovable unit."""
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=self.hook_color)

        post_hz = self.hook_post_half[2]
        # base plate flush on the table
        base_hz = 0.012
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, base_hz]),
            half_size=[0.06, 0.06, base_hz],
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, base_hz]),
            half_size=[0.06, 0.06, base_hz],
            material=mat,
        )
        # vertical post, standing on the base plate
        post_cz = 2 * base_hz + post_hz
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, post_cz]),
            half_size=list(self.hook_post_half),
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, post_cz]),
            half_size=list(self.hook_post_half),
            material=mat,
        )
        # single tilted peg cantilevering out from the post top toward the camera
        axis = self._peg_axis_local()
        peg_center = axis * (self.hook_peg_len / 2)
        peg_center[2] += self.hook_peg_z
        peg_half = [
            self.hook_peg_len / 2,
            self.hook_peg_half_cross,
            self.hook_peg_half_cross,
        ]
        tilt = -np.deg2rad(self.hook_peg_tilt_deg)
        peg_q = euler2quat(0, tilt, 0)
        builder.add_box_collision(
            pose=sapien.Pose(p=peg_center.tolist(), q=peg_q), half_size=peg_half
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=peg_center.tolist(), q=peg_q),
            half_size=peg_half,
            material=mat,
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, -100])
        return builder.build_kinematic(name="hook")

    def _peg_axis_local(self):
        tilt = np.deg2rad(self.hook_peg_tilt_deg)
        horizontal = self.hook_peg_axis / np.linalg.norm(self.hook_peg_axis)
        return np.array(
            [
                horizontal[0] * np.cos(tilt),
                horizontal[1] * np.cos(tilt),
                np.sin(tilt),
            ],
            dtype=np.float64,
        )

    def _peg_axis_world(self):
        return self._peg_axis_local()

    def _peg_tip_world(self):
        """World-frame position of the hook peg's far tip (where the handle loop
        rests when hung). Shape ``[num_envs, 3]``."""
        axis = self._peg_axis_local()
        local_tip = torch.tensor(
            [axis[0] * self.hook_peg_len * 0.6,
             axis[1] * self.hook_peg_len * 0.6,
             self.hook_peg_z + axis[2] * self.hook_peg_len * 0.6],
            dtype=torch.float32,
            device=self.device,
        )
        tip_pose = Pose.create_from_pq(p=local_tip.expand(self.num_envs, -1))
        return (self.hook.pose * tip_pose).p

    # -------------------------------------------------------------- lifecycle --
    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        self._select_bag_links()
        # Height needed so the bag's lowest collision point rests on the table.
        collision_mesh = self.bag.get_first_collision_mesh()
        local_min_z = (
            collision_mesh.bounding_box.bounds[0, 2]
            - self.bag.pose.p[:, 2].min().item()
        )
        self.bag_spawn_z = -local_min_z + 0.003
        # Bag total height (used for the "hanging clear of the table" check).
        bounds = collision_mesh.bounding_box.bounds
        self.bag_height = float(bounds[1, 2] - bounds[0, 2])

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)

        self._set_wide_arm_qpos()

        # --- hook (fixed) ---
        hook_p = self.workspace_offset_tensor.expand(b, -1).clone()
        hook_p[:, 0] += self.hook_center_rel[0]
        hook_p[:, 1] += self.hook_center_rel[1]
        self.hook.set_pose(Pose.create_from_pq(p=hook_p))

        # --- bag (dynamic), handle locked upright ---
        bag_rel_xy = self._sample_bag_xy(b)
        bag_p = self.workspace_offset_tensor.expand(b, -1).clone()
        bag_p[:, :2] += bag_rel_xy
        bag_p[:, 2] += self.bag_spawn_z
        self.bag.set_pose(
            Pose.create_from_pq(p=bag_p, q=euler2quat(0, 0, self.bag_yaw))
        )
        # Latch the handle joint closed (upright) at reset.
        zero_q = torch.zeros((b, self.bag.max_dof), device=self.device)
        self.bag.set_qpos(zero_q)
        self.bag.set_qvel(torch.zeros_like(zero_q))

    def _sample_bag_xy(self, b):
        """Small spawn jitter, bounded away from and near the owning LEFT base."""
        center = torch.tensor(self.bag_center_rel, device=self.device)
        left_base = torch.tensor([0.0, -0.68], device=self.device)
        result = center.expand(b, -1).clone()
        pending = torch.ones((b,), dtype=torch.bool, device=self.device)
        for _ in range(30):
            count = int(pending.sum().item())
            if count == 0:
                break
            candidate = center + (
                torch.rand((count, 2), device=self.device) * 2 - 1
            ) * self.bag_spawn_xy_noise
            dist = torch.linalg.norm(candidate - left_base, axis=1)
            accepted = (dist >= self.bag_min_arm_dist) & (dist <= self.bag_max_arm_dist)
            indices = torch.nonzero(pending, as_tuple=False).flatten()
            result[indices[accepted]] = candidate[accepted]
            pending[indices[accepted]] = False
        return result

    def _set_wide_arm_qpos(self):
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

    # -------------------------------------------------------------- evaluate ---
    def _bag_hung(self):
        """The bag hangs on the hook: the handle loop is around the peg (handle near
        the peg in xy, and in a z band around the peg -- the loop rests on the peg by
        its top bar, so its centre sits just below the peg), the body dangles below
        the handle, and the bag bottom is clear of the table (supported by the peg,
        not resting on the table)."""
        peg = self._peg_tip_world()
        handle_p = self.bag_handle_link.pose.p
        body_p = self.bag_body_link.pose.p

        handle_near_peg_xy = (
            torch.linalg.norm(handle_p[:, :2] - peg[:, :2], axis=1)
            < self.hang_xy_radius
        )
        dz = handle_p[:, 2] - peg[:, 2]
        handle_at_peg_z = (dz < self.hang_handle_z_above) & (
            dz > -self.hang_handle_z_below
        )
        body_below_handle = body_p[:, 2] < handle_p[:, 2]
        table_z = self.workspace_offset[2]
        bag_off_table = (
            body_p[:, 2] - self.bag_height / 2
            > table_z + self.bag_hang_min_clearance
        )
        return (
            handle_near_peg_xy & handle_at_peg_z & body_below_handle & bag_off_table
        )

    def _bag_static(self):
        # A freely hanging bag keeps a small residual pendulum motion even after the
        # arms retreat; 0.1 m/s still rejects an active carry/drop while allowing a
        # visibly settled bag supported by the hook.
        return torch.linalg.norm(self.bag.root_linear_velocity, axis=1) < 0.1

    def evaluate(self):
        bag_hung = self._bag_hung()
        success = bag_hung & self._bag_static() & self._both_static()
        return {
            "success": success,
            "bag_hung": bag_hung,
        }
