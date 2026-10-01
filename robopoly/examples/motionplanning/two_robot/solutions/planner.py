"""Shared keypoint-based motion-planning toolkit for the two-robot tasks.

The solvers in this package describe each task as a short sequence of
*object-relative keypoints* (grasp / pre-grasp / place / handoff poses computed
from the current object poses) and schedule them across the two Panda arms over
time.  This module holds the pieces every solver reuses:

* :class:`MultiPandaArmPlanner` - a thin ``mplib`` wrapper that plans for one arm
  of the :class:`~robopoly.agents.multi_agent.MultiAgent` while holding the
  other arm(s) still and emitting the correctly-shaped flat action.
* helper functions for building grasp poses and for the common
  pick-top / pick-edge / place primitives.

Originally lived inside ``food_serve.py``; extracted so every two-robot solver
shares one implementation.
"""

from __future__ import annotations

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from robopoly.utils.structs.pose import to_sapien_pose


class MultiPandaArmPlanner:
    """Single-arm ``mplib`` planner for one agent of a multi-agent env.

    ``gripper_states`` is a shared list (one entry per agent) so each planner can
    emit a full multi-agent action that keeps the *other* arms frozen at their
    current qpos while this arm follows its planned path.
    """

    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "panda_hand_tcp"

    def __init__(
        self,
        env,
        agent_idx,
        gripper_states,
        debug=False,
        vis=True,
        print_env_info=False,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        import mplib  # local import: only needed when a solver actually runs

        self.env = env
        self.base_env = env.unwrapped
        self.agent_idx = agent_idx
        self.agent = self.base_env.agent.agents[agent_idx]
        self.robot = self.agent.robot
        self.gripper_states = gripper_states
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.elapsed_steps = 0
        self.path_step_repeats = 1
        self.path_max_joint_step = None
        self._mplib = mplib
        # Per-step "tape" recording: when ``recording`` is on, every env.step this
        # arm drives appends its own [arm_qpos(7), gripper_cmd] sub-action. Used to
        # run each arm SOLO (other arm frozen) and later replay both tapes together
        # so the two robots move simultaneously (see food_serve.py co-play).
        self.recording = False
        self.tape = []
        # Optional no-arg callback invoked after every env.step this arm drives. Used
        # to stream a video frame while a single-pass solver runs (see cook_pot.py).
        self.frame_cb = None
        # Optional "partner" tape: while THIS arm executes a LIVE move, the partner arm
        # (``partner_idx``) is driven from a pre-recorded tape instead of frozen, so the
        # two arms move simultaneously while this arm keeps a firm LIVE grasp (used by
        # cook_pot.py to pick the carrot while the lid is grasped live). The cursor
        # advances one tape entry per env.step this arm drives; it clamps (holds the last
        # entry) once the tape is exhausted.
        self.partner_tape = None
        self.partner_idx = None
        self._partner_cursor = 0

        self.planner = self._setup_planner(joint_vel_limits, joint_acc_limits)
        self.arm_dof = len(self.planner.joint_vel_limits)
        self.use_point_cloud = False
        self._collision_pts = None

    def _setup_planner(self, joint_vel_limits, joint_acc_limits):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = self._mplib.Planner(
            urdf=self.agent.urdf_path,
            srdf=self.agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        base_pose = to_sapien_pose(self.robot.pose)
        # mplib >=0.2 expects its Pose object; older releases accepted a flat
        # [p, q] vector. Keep both conventions so dataset generation is portable.
        try:
            planner.set_base_pose(self._mplib.Pose(base_pose.p, base_pose.q))
        except (TypeError, AttributeError):
            planner.set_base_pose(np.hstack([base_pose.p, base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * joint_acc_limits
        return planner

    def _arm_qpos(self, agent):
        return agent.robot.get_qpos()[0, : self.arm_dof].cpu().numpy()

    def _planner_qpos(self):
        return self.robot.get_qpos()[0].cpu().numpy()

    def _flat_action(self, moving_qpos=None):
        per_agent_actions = []
        partner_entry = None
        if self.partner_tape is not None:
            i = min(self._partner_cursor, len(self.partner_tape) - 1)
            partner_entry = self.partner_tape[i]
        for idx, agent in enumerate(self.base_env.agent.agents):
            qpos = self._arm_qpos(agent)
            grip = self.gripper_states[idx]
            if idx == self.agent_idx and moving_qpos is not None:
                qpos = moving_qpos[: self.arm_dof]
            elif partner_entry is not None and idx == self.partner_idx:
                # drive the partner arm from its tape (qpos + gripper), keeping the shared
                # gripper_states in sync so its grasp persists after the tape ends.
                qpos = partner_entry[: self.arm_dof]
                grip = float(partner_entry[self.arm_dof])
                self.gripper_states[idx] = grip
            per_agent_actions.append(np.hstack([qpos, grip]))
        if partner_entry is not None:
            self._partner_cursor += 1
        return np.hstack(per_agent_actions)

    def start_recording(self):
        """Begin a fresh tape of this arm's per-step [qpos(7), gripper] sub-actions."""
        self.recording = True
        self.tape = []
        return self

    def stop_recording(self):
        """Stop recording and return the accumulated tape (list of (8,) arrays)."""
        self.recording = False
        return self.tape

    def _record(self, arm_qpos):
        if self.recording:
            self.tape.append(
                np.hstack(
                    [
                        np.asarray(arm_qpos[: self.arm_dof], dtype=np.float64),
                        self.gripper_states[self.agent_idx],
                    ]
                )
            )

    def _render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def _path_step_targets(self, start_qpos, target_qpos):
        start_qpos = np.asarray(start_qpos, dtype=np.float64)
        target_qpos = np.asarray(target_qpos, dtype=np.float64)
        max_step = self.path_max_joint_step
        if max_step is None or max_step <= 0:
            return [target_qpos[: self.arm_dof]]
        delta = target_qpos[: self.arm_dof] - start_qpos[: self.arm_dof]
        n = max(1, int(np.ceil(np.max(np.abs(delta)) / float(max_step))))
        return [
            start_qpos[: self.arm_dof] + delta * (i / n)
            for i in range(1, n + 1)
        ]

    def follow_path(self, result, refine_steps=0):
        n_step = result["position"].shape[0]
        prev_qpos = self._arm_qpos(self.agent)
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            targets = self._path_step_targets(prev_qpos, qpos)
            if i >= n_step:
                targets = [qpos[: self.arm_dof]]
            for target_qpos in targets:
                for _ in range(max(1, int(self.path_step_repeats))):
                    obs, reward, terminated, truncated, info = self.env.step(
                        self._flat_action(target_qpos)
                    )
                    self._record(target_qpos)
                    self.elapsed_steps += 1
                    if self.print_env_info:
                        print(f"[{self.elapsed_steps:3}] reward={reward} info={info}")
                    if self.vis:
                        self.base_env.render_human()
                    if self.frame_cb is not None:
                        self.frame_cb()
            prev_qpos = qpos[: self.arm_dof]
        return obs, reward, terminated, truncated, info

    def step_gripper(self, gripper_state, steps=8):
        self.gripper_states[self.agent_idx] = gripper_state
        qpos = self._arm_qpos(self.agent)
        for _ in range(steps):
            obs, reward, terminated, truncated, info = self.env.step(
                self._flat_action(qpos)
            )
            self._record(qpos)
            self.elapsed_steps += 1
            if self.vis:
                self.base_env.render_human()
            if self.frame_cb is not None:
                self.frame_cb()
        return obs, reward, terminated, truncated, info

    def open_gripper(self, steps=8):
        return self.step_gripper(self.OPEN, steps=steps)

    def close_gripper(self, steps=10):
        return self.step_gripper(self.CLOSED, steps=steps)

    def move_to_pose_with_screw(self, pose, dry_run=False, refine_steps=0):
        pose = to_sapien_pose(pose)
        goal = np.concatenate([pose.p, pose.q])

        def plan():
            try:
                return self.planner.plan_screw(
                    goal,
                    self._planner_qpos(),
                    time_step=self.base_env.control_timestep,
                    use_point_cloud=self.use_point_cloud,
                )
            except TypeError as exc:
                if "use_point_cloud" not in str(exc):
                    raise
                return self.planner.plan_screw(
                    self._mplib.Pose(pose.p, pose.q),
                    self._planner_qpos(),
                    time_step=self.base_env.control_timestep,
                    wrt_world=True,
                )

        result = plan()
        if result["status"] != "Success":
            result = plan()
            if result["status"] != "Success":
                print(result["status"])
                self._render_wait()
                return -1
        self._render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose(self, pose, dry_run=False, refine_steps=0, attempts=10):
        pose = to_sapien_pose(pose)
        # RRT (and the collision-aware IK it relies on) is randomized, so a single
        # attempt can spuriously fail on a tight, near-obstacle goal. Retry a few
        # times before giving up.
        result = None
        for _ in range(max(1, attempts)):
            if hasattr(self.planner, "plan_qpos_to_pose"):
                result = self.planner.plan_qpos_to_pose(
                    np.concatenate([pose.p, pose.q]),
                    self._planner_qpos(),
                    time_step=self.base_env.control_timestep,
                    use_point_cloud=self.use_point_cloud,
                    planning_time=3,
                    wrt_world=True,
                )
            else:
                result = self.planner.plan_pose(
                    self._mplib.Pose(pose.p, pose.q),
                    self._planner_qpos(),
                    time_step=self.base_env.control_timestep,
                    planning_time=3,
                    wrt_world=True,
                )
            if result["status"] == "Success":
                break
        if result["status"] != "Success":
            print(result["status"])
            self._render_wait()
            return -1
        self._render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    # ------------------------------------------------------------------
    # Collision obstacles (point-cloud based, world frame)
    # ------------------------------------------------------------------
    def add_box_collision(self, extents, pose, n_points=512):
        """Register an axis-aligned-ish box obstacle (world frame) for planning.

        Samples surface points of the box and feeds them to ``mplib`` so RRT /
        screw plans route the arm around it. Enables point-cloud checking.
        """
        import trimesh

        pose = to_sapien_pose(pose)
        box = trimesh.creation.box(
            extents=np.asarray(extents, dtype=np.float64),
            transform=pose.to_transformation_matrix(),
        )
        pts, _ = trimesh.sample.sample_surface(box, n_points)
        pts = np.asarray(pts, dtype=np.float64)
        if self._collision_pts is None:
            self._collision_pts = pts
        else:
            self._collision_pts = np.vstack([self._collision_pts, pts])
        self.use_point_cloud = True
        self.planner.update_point_cloud(self._collision_pts)

    def clear_collisions(self):
        """Drop all registered obstacles and disable point-cloud checking."""
        self._collision_pts = None
        self.use_point_cloud = False

    def in_collision(self, qpos=None):
        """Return the list of collisions for ``qpos`` (default current) vs obstacles."""
        if qpos is None:
            qpos = self._arm_qpos(self.agent)
        try:
            return self.planner.check_for_env_collision(
                qpos=qpos, with_point_cloud=self.use_point_cloud
            )
        except TypeError:
            return self.planner.check_for_env_collision(state=qpos)


# ---------------------------------------------------------------------------
# Geometry / grasp helpers
# ---------------------------------------------------------------------------

def np_pos(actor):
    """World xyz of an actor as float64 numpy (single-env)."""
    return np.asarray(actor.pose.sp.p, dtype=np.float64)


def tcp_closing(agent):
    """Current gripper closing axis (world frame) for an agent's tcp."""
    return agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()


def make_grasp_pose(agent, approaching, closing, center):
    """Orthonormalize (approaching, closing) and build a grasp pose at ``center``."""
    approaching = np.asarray(approaching, dtype=np.float64)
    approaching = approaching / np.linalg.norm(approaching)
    closing = np.asarray(closing, dtype=np.float64)
    closing = closing - approaching * np.dot(approaching, closing)
    closing = closing / np.linalg.norm(closing)
    return agent.build_grasp_pose(approaching, closing, center)


def try_pose_variants(planner, pose, yaw_angles=None):
    """Return the first wrist-yaw variant of ``pose`` that the planner can reach."""
    if yaw_angles is None:
        yaw_angles = [0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4]
    for yaw in yaw_angles:
        candidate = pose * sapien.Pose(q=euler2quat(0, 0, yaw))
        if planner.move_to_pose_with_screw(candidate, dry_run=True) != -1:
            return candidate
    return pose


def move_or_fail(planner, pose, use_rrt=False, refine_steps=0, label="pose"):
    """Move to ``pose`` (screw first, RRT fallback); raise if both fail."""
    print(f"Planning {label}: p={np.array2string(np.asarray(pose.p), precision=4)}")
    if use_rrt:
        res = planner.move_to_pose(pose, refine_steps=refine_steps)
    else:
        res = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
        if res == -1:
            res = planner.move_to_pose(pose, refine_steps=refine_steps)
    if res == -1:
        raise RuntimeError(f"motion planning failed for {label}")
    return res


def move_optional(planner, pose, use_rrt=False, refine_steps=0, label="pose"):
    """Like :func:`move_or_fail` but returns ``-1`` instead of raising on failure."""
    if use_rrt:
        res = planner.move_to_pose(pose, refine_steps=refine_steps)
    else:
        res = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
        if res == -1:
            res = planner.move_to_pose(pose, refine_steps=refine_steps)
    return res


def wait_steps(planner, steps=20):
    """Step the env for ``steps`` frames holding all arms still (lets physics settle)."""
    qpos = planner._arm_qpos(planner.agent)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = planner.env.step(
            planner._flat_action(qpos)
        )
        planner._record(qpos)
        if planner.vis:
            planner.base_env.render_human()
        if planner.frame_cb is not None:
            planner.frame_cb()
    return obs, reward, terminated, truncated, info


def pick_top(planner, actor, z_offset, pregrasp_dist, close_steps=10, lift=0.12):
    """Top-down pick of ``actor``: open, pregrasp above, descend, close, lift."""
    center = np_pos(actor) + np.array([0, 0, z_offset])
    grasp_pose = make_grasp_pose(
        planner.agent,
        approaching=np.array([0, 0, -1]),
        closing=tcp_closing(planner.agent),
        center=center,
    )
    grasp_pose = try_pose_variants(planner, grasp_pose)
    pregrasp_pose = grasp_pose * sapien.Pose([0, 0, -pregrasp_dist])

    planner.open_gripper()
    move_or_fail(planner, pregrasp_pose, use_rrt=True, label="top pregrasp")
    move_or_fail(planner, grasp_pose, label="top grasp")
    planner.close_gripper(steps=close_steps)
    lift_pose = sapien.Pose([0, 0, lift]) * grasp_pose
    move_or_fail(planner, lift_pose, label="top lift")
    return grasp_pose


def place_at(planner, target_xyz, grasp_q, clearance, open_gripper=True, open_steps=15):
    """Move above ``target_xyz``, descend, then release (or retreat)."""
    above_pose = sapien.Pose(target_xyz + np.array([0, 0, clearance]), grasp_q)
    place_pose = sapien.Pose(target_xyz, grasp_q)
    move_or_fail(planner, above_pose, use_rrt=True, label="place above")
    move_or_fail(planner, place_pose, use_rrt=True, label="place")
    if open_gripper:
        return planner.open_gripper(steps=open_steps)
    return move_or_fail(planner, above_pose, label="retreat")


def pick_edge(planner, actor, radius, side_sign, edge_z, edge_depth, close_steps=14):
    """Side grasp of an actor's rim (used for trays / flat objects)."""
    center = np_pos(actor)
    grasp_center = center + np.array([0, side_sign * radius * edge_depth, edge_z])
    approaching = np.array([0, -side_sign, 0])
    grasp_pose = make_grasp_pose(
        planner.agent,
        approaching=approaching,
        closing=np.array([1, 0, 0]),
        center=grasp_center,
    )
    grasp_pose = try_pose_variants(planner, grasp_pose)
    pregrasp_pose = sapien.Pose(grasp_pose.p - approaching * 0.08, grasp_pose.q)

    planner.open_gripper()
    move_or_fail(planner, pregrasp_pose, use_rrt=True, label="edge pregrasp")
    move_or_fail(planner, grasp_pose, label="edge grasp")
    planner.close_gripper(steps=close_steps)
    lift_pose = sapien.Pose([0, 0, 0.12]) * grasp_pose
    return move_or_fail(planner, lift_pose, label="edge lift")


def make_planners(env, n=2, debug=False, vis=False, **kwargs):
    """Build one :class:`MultiPandaArmPlanner` per agent sharing a gripper-state list.

    Returns ``(planners, gripper_states)``. ``planners[0]`` is the left arm
    (agents[0], -y side) and ``planners[1]`` the right arm (agents[1], +y side).
    """
    gripper_states = [MultiPandaArmPlanner.OPEN] * n
    planners = [
        MultiPandaArmPlanner(env, idx, gripper_states, debug=debug, vis=vis, **kwargs)
        for idx in range(n)
    ]
    return planners, gripper_states
