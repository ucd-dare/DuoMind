import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building import get_articulation_builder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .two_robot_kitchen_base import TwoRobotKitchenReplicaCADBaseEnv


@register_env(
    "TwoRobotPutObjectCabinetReplicaCAD-v1",
    max_episode_steps=250,
    asset_download_ids=["ReplicaCAD"],
)
class TwoRobotPutObjectCabinetReplicaCADEnv(TwoRobotKitchenReplicaCADBaseEnv):
    # PartNet-Mobility cabinet (two sliding drawers). Imported exactly like the
    # oven (7220) in ``two_robot_use_oven``: try the registered partnet-mobility
    # builder first, then fall back to loading the vendored URDF directly.
    cabinet_model_id = "19179"
    # Shift the cabinet 8 cm toward the visual left (-y from the global camera).
    # This moves the drawer/body away from the right arm's cube-pick corridor while
    # remaining comfortably inside the left arm's drawer-handle workspace.
    cabinet_center_rel = (-0.4, -0.08)
    cabinet_yaw = np.pi
    cabinet_scale = 0.4
    # The PartNet drawer handle is a thin, nearly flush bar. The demonstration
    # solver had to kinematically couple it because the stock contact material
    # and a heavily damped slider let the fingers slide along the handle instead
    # of pulling the drawer. Evaluation uses normal contact physics, so make that
    # contact deliberately forgiving while keeping the task geometry unchanged.
    drawer_joint_friction = 0.01
    drawer_drive_damping = 0.05
    drawer_mass = 1.0
    cabinet_sliding_friction = 0.01
    handle_friction = 10.0
    gripper_friction = 10.0
    upper_drawer_lock_stiffness = 5000.0
    upper_drawer_lock_damping = 200.0
    upper_drawer_lock_force = 5000.0
    upper_drawer_lock_friction = 10.0
    drawer_pair_no_collision_bit = 28
    # Fraction of the drawer's travel used to define "opened" / "closed".
    drawer_open_frac = 0.85
    drawer_closed_frac = 0.15
    # Compact spawn band: closer to the +x global camera, centered between the
    # cabinet (y=-1.28) and right-arm base (y=-0.52). This stays clear of both
    # while giving the raised right arm a short, nearly vertical pick approach.
    cube_spawn_x_range = (0.12, 0.18)
    cube_spawn_y_range = (0.26, 0.32)
    # Nudge BOTH arm bases +x (toward the cabinet / drawer pull-out direction) so the
    # arms' motions are cleaner. The drawer pulls out to x~1.66 -- at the default base
    # x=1.4 the fully-extended handle sits right at the LEFT arm's front-grip reach
    # limit (near-singular), which is what forced all the extension/descent workarounds.
    # Shifting +0.08 m centers each arm on its own workspace (left: the 1.26->1.66
    # handle travel; right: the far cube spawns) without meaningfully cramping the near
    # end. Base y (arm separation) is unchanged, so arm-vs-arm clearance is preserved.
    # The LEFT base is ANGLED to face the cabinet: nudged +0.20 m in x and +0.20 m in y and rotated
    # from the default +pi/2 (facing +y) to +3pi/4 (facing the -x/+y diagonal, TOWARD the cabinet
    # body / the drawer it operates -- not out toward the pulled-out end). Rationale: the handle
    # travel is along +x; facing +y made the pull a sideways SWEEP (uneven, arm near full
    # extension), so the coupled drawer stuttered. Facing the base head-on toward the cabinet turns
    # the pull into a single clean reach -> combined with _coupled_pull segs=1 (one screw segment)
    # the drawer opens/closes in ONE continuous motion (per-frame CoV ~0.65, no mid-motion pause),
    # and the approach to the handle is short/direct (~35 cm). Accommodations for the angled/nearer
    # base (its opening reaches near the cube): the co-play restores the cube to its spawn before
    # the right-pick record (see solve()), and home_arm_qpos_left is retuned for the rotated base.
    arm_base_shift_left = (0.20, 0.20)
    arm_base_shift_right = (0.08, 0.0)
    arm_base_yaw_left = 3 * np.pi / 4
    # Retracted spawn pose per arm (7 joints; grippers open below). The stock ready
    # pose spawns the two hands crammed together over the drawer opening (~0.05 m
    # link gap), so BOTH the left arm and the pulled-out drawer hit the right arm.
    # Each arm bends the elbow more / eases the shoulder to pull its hand back toward
    # its own base. (The right arm is additionally lifted clear of the opening drawer
    # by the SOLVER at run time -- see put_object_cabinet.py -- not here.) Both bases
    # are additionally nudged +x (see arm_base_shift_* above).
    # LEFT home tuned for the angled base so the approach is a plain reach with NO wrist spin:
    #  * joint 0 (shoulder pan) -0.5 rad (was 0.0): re-points the arm at the handle after the base
    #    was rotated to +3pi/4 -- without it the arm faces off, can't grasp, whole episode fails.
    #  * joint 4 (forearm roll) +1.74 rad (was 0.0): pre-orients the forearm to the grasp value so
    #    it doesn't wind ~100 deg during the descent. (Joint 6 / wrist roll is handled on the grasp
    #    side -- the flipped closing axis puts it near home; see phase_open.)
    # Result: yield ~11/12 solve, the grasp descent barely rotates the wrist, and the pull is
    # smoothest yet (CoV ~0.52).
    home_arm_qpos_left = (-0.5, 0.2, 0.0, -2.05, 1.74, 2.25, -np.pi / 4)
    # Start already in the elevated pre-pick posture. The old low home first had
    # to climb, then descend again, and could touch/displace a nearby cube during
    # the initial settle.
    home_arm_qpos_right = (-0.05, -0.55, -0.08, -2.75, -0.05, 2.20, -0.88)

    def _load_task_scene(self, options: dict):
        self.cube = actors.build_cube(
            self.scene,
            half_size=0.035,
            color=[0.12, 0.48, 0.95, 1],
            name="cabinet_task_cube",
            initial_pose=sapien.Pose(p=[0, 0, -100]),
        )
        self.cabinet = self._build_cabinet()

    def _build_cabinet(self):
        try:
            builder = get_articulation_builder(
                self.scene, f"partnet-mobility:{self.cabinet_model_id}"
            )
        except Exception as exc:
            builder = self._get_direct_partnet_builder(exc)
        if builder is None:
            raise RuntimeError(
                f"Could not locate the PartNet-Mobility cabinet "
                f"'{self.cabinet_model_id}'. Expected a 'mobility.urdf' under "
                f"partnet_mobility/dataset/{self.cabinet_model_id} in either the "
                f"package assets or the downloaded asset directory."
            )
        builder.initial_pose = sapien.Pose(
            p=[0, 0, -100], q=euler2quat(0, 0, self.cabinet_yaw)
        )
        cabinet = builder.build(name=f"partnet_cabinet_{self.cabinet_model_id}")
        self.remove_from_state_dict_registry(cabinet)
        self.add_to_state_dict_registry(cabinet)
        for joint in cabinet.active_joints:
            joint.set_friction(self.drawer_joint_friction)
            joint.set_drive_properties(0.0, self.drawer_drive_damping)
        return cabinet

    def _get_direct_partnet_builder(self, original_exc):
        model_dirs = [
            PACKAGE_ASSET_DIR / "partnet_mobility/dataset" / self.cabinet_model_id,
            ASSET_DIR / "partnet_mobility/dataset" / self.cabinet_model_id,
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
                    loader.fix_root_link = True
                    loader.scale = self.cabinet_scale
                    loader.load_multiple_collisions_from_file = True
                    # The raw drawer meshes are open-top trays; a single convex
                    # hull would fill them in solid (cube can't go inside). COACD
                    # decomposition preserves the hollow interior so the cube
                    # actually drops into the drawer.
                    loader.multiple_collisions_decomposition = "coacd"
                    loader.multiple_collisions_decomposition_params = dict(
                        threshold=0.05
                    )
                    sapien_utils.apply_urdf_config(
                        loader,
                        sapien_utils.parse_urdf_config(
                            dict(
                                material=dict(
                                    static_friction=1,
                                    dynamic_friction=1,
                                    restitution=0,
                                )
                            )
                        ),
                    )
                    return loader.parse(str(urdf_path))["articulation_builders"][0]
        return None

    def _link_local_geom_center(self, link):
        """AABB centre of a link's collision geometry, expressed in the link's
        own local frame. Static (built from the collision-shape vertices), so it
        is independent of the sim backend / GPU pose state."""
        sapien_link = link._objs[0]
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        for cs in sapien_link.get_collision_shapes():
            verts = np.asarray(cs.vertices) * np.asarray(cs.scale)
            T = cs.get_local_pose().to_transformation_matrix()
            world_verts = (T[:3, :3] @ verts.T).T + T[:3, 3]
            mins = np.minimum(mins, world_verts.min(0))
            maxs = np.maximum(maxs, world_verts.max(0))
        return (mins + maxs) / 2

    def _apply_easy_open_contact_materials(self):
        """Increase friction only at the fingers and target drawer handle."""
        # The scaled PartNet target drawer is 10.33 kg by default. That is far
        # heavier than a small kitchen drawer and makes a shallow pinch on its
        # flush handle unnecessarily demanding. Preserve the mass distribution
        # while scaling total mass and inertia to a realistic, forgiving value.
        for drawer_obj in self.drawer_link._objs:
            mass_scale = self.drawer_mass / drawer_obj.mass
            drawer_obj.set_mass(self.drawer_mass)
            drawer_obj.set_inertia(np.asarray(drawer_obj.inertia) * mass_scale)

        # PartNet's drawer and cabinet collision meshes touch along their guide
        # surfaces. Give those internal sliding contacts low friction; the target
        # handle is overridden with a high-friction material below.
        sliding_material = sapien.physx.PhysxMaterial(
            static_friction=self.cabinet_sliding_friction,
            dynamic_friction=self.cabinet_sliding_friction,
            restitution=0.0,
        )
        for cabinet_link in self.cabinet.links:
            for cabinet_obj in cabinet_link._objs:
                for shape in cabinet_obj.get_collision_shapes():
                    shape.set_physical_material(sliding_material)

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

        # Identify the frontmost y-long collision shape, matching the handle
        # geometry used by the motion-planning solution, then apply the material
        # to that same shape in every sub-scene. Avoid changing the drawer tray
        # material because high tray/cabinet friction could resist sliding.
        reference_shapes = self.drawer_link._objs[0].get_collision_shapes()
        candidates = []
        for shape_idx, shape in enumerate(reference_shapes):
            vertices = np.asarray(shape.vertices) * np.asarray(shape.scale)
            transform = shape.get_local_pose().to_transformation_matrix()
            local_vertices = (
                transform[:3, :3] @ vertices.T
            ).T + transform[:3, 3]
            vertex_pose = Pose.create_from_pq(
                p=torch.tensor(
                    local_vertices, dtype=torch.float32, device=self.device
                )
            )
            world_vertices = (self.drawer_link.pose * vertex_pose).p.cpu().numpy()
            center_x = 0.5 * (
                world_vertices[:, 0].min() + world_vertices[:, 0].max()
            )
            y_length = world_vertices[:, 1].max() - world_vertices[:, 1].min()
            if y_length > 0.08:
                candidates.append((center_x, shape_idx))
        if not candidates:
            raise RuntimeError("Could not identify the target drawer handle collision shape")
        handle_shape_idx = max(candidates)[1]
        handle_material = sapien.physx.PhysxMaterial(
            static_friction=self.handle_friction,
            dynamic_friction=self.handle_friction,
            restitution=0.0,
        )
        for drawer_obj in self.drawer_link._objs:
            drawer_obj.get_collision_shapes()[handle_shape_idx].set_physical_material(
                handle_material
            )

    def _select_drawer_joints(self):
        """Pick the prismatic drawer joints. Both drawer links share the cabinet
        root as their link origin, so we rank them by the world height of their
        *geometry* centre (link pose applied to the local collision centre). The
        upper drawer is the target: it is the one modelled with a full tray floor
        (``drawer_bottom``) that can actually hold the cube, whereas the lower
        drawer is only a front panel. Every drawer is closed at reset."""
        if len(self.cabinet.active_joints) < 1:
            raise RuntimeError(
                f"Cabinet {self.cabinet_model_id} exposes no movable drawers."
            )

        def geom_world_z(joint):
            link = joint.child_link
            center = self._link_local_geom_center(link)
            center_pose = Pose.create_from_pq(
                p=torch.tensor(center, dtype=torch.float32, device=self.device)
            )
            return float((link.pose * center_pose).p[:, 2].mean().item())

        drawer_joints = sorted(
            self.cabinet.active_joints, key=geom_world_z, reverse=True
        )
        # Target the LOWER drawer -- the one the co-play solver actually opens, fills,
        # and closes. (The env used to target the upper drawer for success; success is
        # now the cube resting in the closed LOWER drawer, matching the demo.)
        self.drawer_joint = drawer_joints[-1]  # lower drawer
        self.fixed_drawer_joints = drawer_joints[:-1]
        # The imported upper/lower drawer collision meshes overlap enough that
        # contact drags them together. Disable collision only between this pair;
        # their contacts with the cabinet body, grippers, and task cube remain.
        self.drawer_joint.child_link.set_collision_group_bit(
            group=2, bit_idx=self.drawer_pair_no_collision_bit, bit=1
        )
        for joint in self.fixed_drawer_joints:
            joint.child_link.set_collision_group_bit(
                group=2, bit_idx=self.drawer_pair_no_collision_bit, bit=1
            )
            joint.set_friction(self.upper_drawer_lock_friction)
            joint.set_drive_properties(
                self.upper_drawer_lock_stiffness,
                self.upper_drawer_lock_damping,
                force_limit=self.upper_drawer_lock_force,
                mode="force",
            )
            joint.set_drive_target(0.0)
        self.drawer_link = self.drawer_joint.child_link
        self.drawer_joint_idx = int(
            self.drawer_joint.active_index.flatten()[0].item()
        )
        self.all_drawer_joint_idxs = [
            int(j.active_index.flatten()[0].item())
            for j in self.cabinet.active_joints
        ]
        # Local-frame offset from the drawer link origin to its cavity centre,
        # used at evaluation time to locate where the cube must rest.
        self.drawer_local_center = torch.tensor(
            self._link_local_geom_center(self.drawer_link),
            dtype=torch.float32,
            device=self.device,
        )

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        # Forward kinematics has run, so child-link world poses are now valid;
        # this is where we can reliably tell the upper drawer from the lower one.
        self._select_drawer_joints()
        self._apply_easy_open_contact_materials()
        # Lift the cabinet so its lowest collision point rests on the table top.
        collision_mesh = self.cabinet.get_first_collision_mesh()
        local_min_z = (
            collision_mesh.bounding_box.bounds[0, 2]
            - self.cabinet.pose.p[:, 2].min().item()
        )
        self.cabinet_z = -local_min_z

    def _drawer_cavity_center(self):
        """World-frame centre of the target drawer's cavity, tracking the drawer
        as it slides open/closed. Shape ``[num_envs, 3]``."""
        center_pose = Pose.create_from_pq(
            p=self.drawer_local_center.expand(self.num_envs, -1)
        )
        return (self.drawer_link.pose * center_pose).p

    def _set_retracted_arm_qpos(self):
        """Spawn each arm at its retracted home pose (``home_arm_qpos_left/right``),
        grippers open. Uses the agent reset path (zeros qpos/qvel/qf together) so no
        residual joint force is left behind. Base poses are set separately and are
        left unchanged. The env's controller reset (post-``_initialize_episode``)
        then latches the drive target to this qpos, so the arms hold here."""
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

    def _initialize_task_episode(self, env_idx: torch.Tensor, options: dict):
        if not hasattr(self, "drawer_opened_once"):
            self.drawer_opened_once = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self.drawer_opened_once[env_idx] = False
        b = len(env_idx)

        self._set_retracted_arm_qpos()

        p = self.workspace_offset_tensor.expand(b, -1).clone()
        p[:, 0] += self.cabinet_center_rel[0]
        p[:, 1] += self.cabinet_center_rel[1]
        p[:, 2] += self.cabinet_z
        self.cabinet.set_pose(
            Pose.create_from_pq(p=p, q=euler2quat(0, 0, self.cabinet_yaw))
        )
        self.close_drawer()

        cube_xyz = self.workspace_offset_tensor.expand(b, -1).clone()
        cube_xyz[:, 0] += (
            torch.rand((b,), device=self.device)
            * (self.cube_spawn_x_range[1] - self.cube_spawn_x_range[0])
            + self.cube_spawn_x_range[0]
        )
        cube_xyz[:, 1] += (
            torch.rand((b,), device=self.device)
            * (self.cube_spawn_y_range[1] - self.cube_spawn_y_range[0])
            + self.cube_spawn_y_range[0]
        )
        cube_xyz[:, 2] += 0.035
        self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

    def _drawer_qpos_target(self, frac):
        qlimits = self.cabinet.get_qlimits()
        qpos = qlimits[:, :, 0].clone()
        for joint_idx in self.all_drawer_joint_idxs:
            qmin = qlimits[:, joint_idx, 0]
            qmax = qlimits[:, joint_idx, 1]
            if joint_idx == self.drawer_joint_idx:
                qpos[:, joint_idx] = qmin + (qmax - qmin) * frac
            else:
                qpos[:, joint_idx] = qmin
        return qpos

    def set_drawer_open(self):
        qpos = self._drawer_qpos_target(self.drawer_open_frac)
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_qvel(torch.zeros_like(qpos))

    def close_drawer(self):
        qpos = self._drawer_qpos_target(0.0)
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_qvel(torch.zeros_like(qpos))

    def _drawer_open(self):
        qlimits = self.cabinet.get_qlimits()
        qmin = qlimits[:, self.drawer_joint_idx, 0]
        qmax = qlimits[:, self.drawer_joint_idx, 1]
        qpos = self.cabinet.qpos[:, self.drawer_joint_idx]
        return qpos >= qmin + (qmax - qmin) * self.drawer_open_frac

    def _drawer_closed(self):
        qlimits = self.cabinet.get_qlimits()
        qmin = qlimits[:, self.drawer_joint_idx, 0]
        qmax = qlimits[:, self.drawer_joint_idx, 1]
        qpos = self.cabinet.qpos[:, self.drawer_joint_idx]
        return qpos <= qmin + (qmax - qmin) * self.drawer_closed_frac

    def _cube_in_drawer(self):
        # The drawer interior spans ~0.45 m; the walls physically bound the cube,
        # so anywhere inside that footprint (and within the drawer's height band)
        # counts. The closed-drawer cavity sits ~0.4 m from any table spawn, so a
        # generous xy radius cannot false-trigger on a cube left on the table.
        cavity = self._drawer_cavity_center()
        in_xy = (
            torch.linalg.norm(self.cube.pose.p[:, :2] - cavity[:, :2], axis=1) < 0.19
        )
        in_z = torch.abs(self.cube.pose.p[:, 2] - cavity[:, 2]) < 0.13
        return in_xy & in_z

    def evaluate(self):
        drawer_open = self._drawer_open()
        drawer_closed = self._drawer_closed()
        self.drawer_opened_once = self.drawer_opened_once | drawer_open
        cube_in_drawer = self._cube_in_drawer()
        success = (
            self.drawer_opened_once
            & drawer_closed
            & cube_in_drawer
            & self._both_static()
        )
        return {
            "success": success,
            "drawer_open": drawer_open,
            "drawer_closed": drawer_closed,
            "drawer_opened_once": self.drawer_opened_once,
            "cube_in_drawer": cube_in_drawer,
        }
