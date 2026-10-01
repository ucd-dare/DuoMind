"""Keypoint solver for ``TwoRobotBottleExchangeReplicaCAD-v1``.

This revised task uses two cups and two transparent clean-table bins. Each cup is
picked with the same top-down body grasp, placed on the middle of the table, then
picked up by the opposite arm and dropped into the opposite bin:

* Phase 1:
    - LEFT  ``pick up mug and place it in the middle``
    - RIGHT ``pick up staged mug and place it in the box``
* Phase 2:
    - RIGHT ``pick up mug and place it in the middle``
    - LEFT  ``pick up staged mug and place it in the box``

The solver runs these motions sequentially: after a mug is set down in the
middle, that arm withdraws before the opposite arm moves in to pick it up. The
recorded tape keeps the non-moving arm static for each segment to avoid the
collisions caused by overlapping retreat and receive motions.
"""

from dataclasses import dataclass

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from robopoly.examples.motionplanning.two_robot.solutions.planner import (
    MultiPandaArmPlanner,
    make_grasp_pose as _make_grasp_pose,
    move_optional as _move_optional,
    move_or_fail as _move_or_fail,
    np_pos as _np_pos,
    tcp_closing as _tcp_closing,
    try_pose_variants as _try_pose_variants,
    wait_steps as _wait,
)

LABEL_LEFT_1 = "pick up cup and place in middle"
LABEL_RIGHT_1 = "pick up cup and place in box"
LABEL_LEFT_2 = "pick up cup and place in box"
LABEL_RIGHT_2 = "pick up cup and place in middle"

SETTLE = 12
FINAL_HOLD = 30


@dataclass
class BottleExchangeConfig:
    mug_grasp_z: float = 0.035
    mug_body_center_local_z: float = 0.003
    left_second_pregrasp_body_center_local_z: float = 0.010
    left_second_grasp_body_center_local_z: float = 0.0065
    mug_pregrasp: float = 0.10
    mug_lift: float = 0.18
    left_to_right_middle_rel_xy: tuple = (0.0, 0.04)
    right_to_left_middle_rel_xy: tuple = (0.0, -0.04)
    middle_stage_z: float = 0.005
    middle_transit: float = 0.24
    middle_hover: float = 0.12
    box_transit: float = 0.22
    box_hover: float = 0.14
    box_place_clear: float = 0.003
    box_release_wait: int = 30
    box_post_release_wait: int = 20
    box_release_heights: tuple = (0.0, 0.006, 0.012, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10)
    box_release_from_above: bool = False
    box_retreat_after_release: bool = True
    box_release_clear_up: float = 0.16
    right_box_release_xy: tuple = (0.0, 0.0)
    left_box_release_xy: tuple = (0.0, 0.0)
    open_steps: int = 56
    close_steps: int = 50
    retreat_up: float = 0.14
    retreat_side: float = 0.25
    box_retreat_side: float = 0.12
    max_joint_step: float = 0.040
    bread_grasp_above: float = 0.006


def _mug_rotation_matrix(mug):
    q = np.asarray(mug.pose.sp.q, dtype=np.float64)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _configure_motion_smoothing(planners, cfg):
    for planner in planners:
        planner.path_max_joint_step = cfg.max_joint_step


def _top_grasp_pose(planner, center, closing, minimize_wrist=False):
    base = _make_grasp_pose(
        planner.agent,
        approaching=np.array([0.0, 0.0, -1.0]),
        closing=np.asarray(closing, dtype=np.float64),
        center=np.asarray(center, dtype=np.float64),
    )
    if not minimize_wrist:
        return _try_pose_variants(planner, base, yaw_angles=[0.0, np.pi])

    # The left arm retains its first-cup top-down grasp orientation throughout
    # retreat, and cup yaw differs only by the jaws' 180-degree symmetry. Choose
    # the ideal handle-safe frame closest to that current orientation; do not
    # prefer the other frame merely because its endpoint screw IK is easier.
    current_q = np.asarray(planner.agent.tcp.pose.sp.q, dtype=np.float64)
    candidates = [
        base * sapien.Pose(q=euler2quat(0.0, 0.0, yaw))
        for yaw in (0.0, np.pi)
    ]
    return max(
        candidates,
        key=lambda candidate: abs(
            float(np.dot(current_q, np.asarray(candidate.q, dtype=np.float64)))
        ),
    )


def _upright_mug_body_closing_axis(mug):
    """Pinch across the mug body while avoiding the handle.

    The RoboTwin mug handle protrudes on the mesh-local -z side.  The clean body
    diameter is therefore mesh-local x, so the two fingers land on opposite cup
    walls instead of a handle-side protrusion or the narrower lower geometry.
    """
    rotation = _mug_rotation_matrix(mug)
    closing = rotation[:, 0]
    closing[2] = 0.0
    return closing / np.linalg.norm(closing)


def _upright_mug_body_grasp_center(mug, cfg, body_center_local_z=None):
    """Center the top-down grasp on the cup body, shifted away from the handle."""
    rotation = _mug_rotation_matrix(mug)
    vertical = rotation[:, 1]
    if body_center_local_z is None:
        body_center_local_z = cfg.mug_body_center_local_z
    body_offset = rotation[:, 2] * body_center_local_z
    center = _np_pos(mug) + body_offset + vertical * cfg.mug_grasp_z
    center[2] = _np_pos(mug)[2] + cfg.mug_grasp_z
    return center


def _collision_vertices_world(actor):
    """Return world-space collision vertices for a merged task actor."""
    entity = actor._objs[0]
    entity_tf = entity.get_pose().to_transformation_matrix()
    points = []
    for component in entity.get_components():
        for shape in getattr(component, "collision_shapes", ()) or ():
            vertices = np.asarray(shape.get_vertices()) * np.asarray(shape.get_scale())
            local_tf = shape.get_local_pose().to_transformation_matrix()
            vertices = (local_tf[:3, :3] @ vertices.T).T + local_tf[:3, 3]
            vertices = (entity_tf[:3, :3] @ vertices.T).T + entity_tf[:3, 3]
            points.append(vertices)
    return np.vstack(points)


def _bread_grasp_center(actor, cfg):
    vertices = _collision_vertices_world(actor)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    # Match Prepare Snack exactly: grasp 6 mm above the live collision-box
    # midpoint, not 6 mm above its bottom (which makes the fingers hit the table).
    center[2] += cfg.bread_grasp_above
    return center


def _move_orientation_locked(planner, pose, refine_steps=0, label="fixed-pose move"):
    result = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
    if result != -1:
        return result
    result = planner.move_to_pose(pose, refine_steps=refine_steps)
    if result == -1:
        raise RuntimeError(f"orientation-locked motion planning failed for {label}")
    return result


def _arm_entry(planner, grip_state):
    return np.hstack([planner._arm_qpos(planner.agent), float(grip_state)])


def _record_segment(planners, gripper_states, mover_idx, fn):
    mover = planners[mover_idx]
    other_idx = 1 - mover_idx
    other = planners[other_idx]
    static_entry = _arm_entry(other, gripper_states[other_idx]).copy()
    mover.start_recording()
    try:
        fn()
        mover_tape = list(mover.stop_recording())
    except Exception:
        mover.stop_recording()
        raise
    other_tape = [static_entry.copy() for _ in range(len(mover_tape))]
    if mover_idx == 0:
        return mover_tape, other_tape
    return other_tape, mover_tape


def _coplay(env, left_tape, right_tape, frame_cb=None, labels=None):
    n = max(len(left_tape), len(right_tape))
    last = None
    for i in range(n):
        la = left_tape[min(i, len(left_tape) - 1)]
        ra = right_tape[min(i, len(right_tape) - 1)]
        last = env.step(np.hstack([la, ra]))
        if frame_cb is not None and labels is not None:
            frame_cb(i, n, *labels(i))
    return last


def _find_record_wrapper(env):
    e = env
    while e is not None:
        if e.__class__.__name__ == "RecordEpisode":
            return e
        e = getattr(e, "env", None)
    return None


def _pick_mug(planner, cfg, mug, label_prefix, clean_wrist=False):
    if "exchange_bread" in str(getattr(mug, "name", "")):
        # Prepare Snack intentionally uses one fixed top-down jaw frame for its
        # nearly symmetric square bread. Searching the cup's yaw-equivalent IK
        # variants can select a needlessly difficult 180-degree wrist branch.
        grasp = _make_grasp_pose(
            planner.agent,
            approaching=np.array([0.0, 0.0, -1.0]),
            closing=_tcp_closing(planner.agent),
            center=_bread_grasp_center(mug, cfg),
        )
        pregrasp = sapien.Pose(
            grasp.p + np.array([0.0, 0.0, cfg.mug_pregrasp]), grasp.q
        )
        planner.open_gripper(steps=8)
        _move_or_fail(
            planner, pregrasp, use_rrt=True, label=f"{label_prefix} pregrasp"
        )
        grasp = sapien.Pose(_bread_grasp_center(mug, cfg), grasp.q)
        _move_or_fail(
            planner, grasp, label=f"{label_prefix} grasp", refine_steps=10
        )
        planner.close_gripper(steps=cfg.close_steps)
        _wait(planner, steps=12)
        lift_pose = sapien.Pose(
            grasp.p + np.array([0.0, 0.0, cfg.mug_lift]), grasp.q
        )
        _move_orientation_locked(
            planner, lift_pose, refine_steps=8, label=f"{label_prefix} lift"
        )
        return grasp, _np_pos(mug) - planner.agent.tcp.pose.sp.p

    pregrasp_body_center_local_z = (
        cfg.left_second_pregrasp_body_center_local_z
        if clean_wrist
        else cfg.mug_body_center_local_z
    )
    grasp = _top_grasp_pose(
        planner,
        _upright_mug_body_grasp_center(mug, cfg, pregrasp_body_center_local_z),
        closing=_upright_mug_body_closing_axis(mug),
        minimize_wrist=clean_wrist,
    )
    pregrasp = sapien.Pose(grasp.p + np.array([0.0, 0.0, cfg.mug_pregrasp]), grasp.q)
    planner.open_gripper(steps=8)
    _move_or_fail(
        planner,
        pregrasp,
        use_rrt=True,
        label=f"{label_prefix} pregrasp",
    )

    if clean_wrist:
        # Clear the handle at pregrasp, then descend toward the body centre for
        # a balanced pinch that will not slip during the carry.
        # Preserve the selected wrist orientation through the straight descent.
        # Do not run variant selection again after reaching the pregrasp.
        grasp = sapien.Pose(
            _upright_mug_body_grasp_center(
                mug, cfg, cfg.left_second_grasp_body_center_local_z
            ),
            grasp.q,
        )
    else:
        grasp = _top_grasp_pose(
            planner,
            _upright_mug_body_grasp_center(mug, cfg),
            closing=_upright_mug_body_closing_axis(mug),
        )
    _move_or_fail(
        planner,
        grasp,
        label=f"{label_prefix} grasp",
        refine_steps=10,
    )
    planner.close_gripper(steps=cfg.close_steps)
    _wait(planner, steps=8)
    lift_pose = sapien.Pose(grasp.p + np.array([0.0, 0.0, cfg.mug_lift]), grasp.q)
    _move_orientation_locked(
        planner,
        lift_pose,
        refine_steps=8,
        label=f"{label_prefix} lift",
    )
    mug_in_tcp = _np_pos(mug) - planner.agent.tcp.pose.sp.p
    return grasp, mug_in_tcp


def _retreat_to_side(planner, cfg, label_prefix):
    side = -1.0 if planner.agent_idx == 0 else 1.0
    cur = planner.agent.tcp.pose.sp
    _move_optional(
        planner,
        sapien.Pose(cur.p + np.array([0.0, 0.0, cfg.retreat_up]), cur.q),
        label=f"{label_prefix} retreat up",
    )
    cur = planner.agent.tcp.pose.sp
    _move_optional(
        planner,
        sapien.Pose(cur.p + np.array([0.0, side * cfg.retreat_side, 0.0]), cur.q),
        label=f"{label_prefix} retreat side",
    )
    _wait(planner, steps=6)


def _place_mug_on_middle(
    planner, cfg, mug, middle_world, grasp_q, label_prefix, retreat=True
):
    mug_in_tcp = _np_pos(mug) - planner.agent.tcp.pose.sp.p
    transit_tcp = (
        middle_world
        + np.array([0.0, 0.0, cfg.middle_transit], dtype=np.float64)
        - mug_in_tcp
    )
    _move_orientation_locked(
        planner,
        sapien.Pose(transit_tcp, grasp_q),
        label=f"{label_prefix} middle transit",
    )
    _wait(planner, steps=6)
    mug_in_tcp = _np_pos(mug) - planner.agent.tcp.pose.sp.p
    place_tcp = middle_world - mug_in_tcp
    _move_orientation_locked(
        planner,
        sapien.Pose(place_tcp + np.array([0.0, 0.0, cfg.middle_hover]), grasp_q),
        label=f"{label_prefix} middle hover",
    )
    mug_in_tcp = _np_pos(mug) - planner.agent.tcp.pose.sp.p
    place_tcp = middle_world - mug_in_tcp
    _move_orientation_locked(
        planner,
        sapien.Pose(place_tcp, grasp_q),
        refine_steps=10,
        label=f"{label_prefix} middle place",
    )
    _wait(planner, steps=12)
    planner.open_gripper(steps=cfg.open_steps)
    _wait(planner, steps=4)
    cur = planner.agent.tcp.pose.sp
    _move_optional(
        planner,
        sapien.Pose(cur.p + np.array([0.0, 0.0, 0.06]), cur.q),
        label=f"{label_prefix} middle release clear up",
    )
    if retreat:
        _retreat_after_middle(planner, cfg, label_prefix)


def _retreat_after_middle(planner, cfg, label_prefix):
    _wait(planner, steps=18)
    _retreat_to_side(planner, cfg, label_prefix)


def _place_mug_in_box(
    planner, cfg, base_env, mug, box_bottom_world, grasp_q, label_prefix
):
    release_xy = (
        cfg.left_box_release_xy if planner.agent_idx == 0 else cfg.right_box_release_xy
    )
    place_world = np.array(
        [
            box_bottom_world[0] + release_xy[0],
            box_bottom_world[1] + release_xy[1],
            box_bottom_world[2] + base_env.box_wall_thickness / 2 + cfg.box_place_clear,
        ],
        dtype=np.float64,
    )
    mug_in_tcp = _np_pos(mug) - planner.agent.tcp.pose.sp.p
    transit_tcp = (
        place_world + np.array([0.0, 0.0, cfg.box_transit], dtype=np.float64) - mug_in_tcp
    )
    _move_orientation_locked(
        planner,
        sapien.Pose(transit_tcp, grasp_q),
        label=f"{label_prefix} box transit",
    )
    _wait(planner, steps=6)

    if not cfg.box_release_from_above:
        release_pose = None
        mug_in_tcp = _np_pos(mug) - planner.agent.tcp.pose.sp.p
        for h in cfg.box_release_heights:
            candidate_tcp = (
                place_world
                + np.array([0.0, 0.0, h], dtype=np.float64)
                - mug_in_tcp
            )
            candidate_pose = sapien.Pose(candidate_tcp, grasp_q)
            if planner.move_to_pose_with_screw(candidate_pose, dry_run=True) != -1:
                release_pose = candidate_pose
                break
            if planner.move_to_pose(candidate_pose, dry_run=True, attempts=1) != -1:
                release_pose = candidate_pose
                break
        if release_pose is not None and np.linalg.norm(
            release_pose.p - planner.agent.tcp.pose.sp.p
        ) > 1e-4:
            # Cups are lowered until supported by the floor to avoid tipping.
            _move_orientation_locked(
                planner,
                release_pose,
                refine_steps=14,
                label=f"{label_prefix} box release pose",
            )

    _wait(planner, steps=cfg.box_release_wait)
    planner.open_gripper(steps=cfg.open_steps)
    _wait(planner, steps=cfg.box_post_release_wait)

    if not cfg.box_retreat_after_release:
        return

    cur = planner.agent.tcp.pose.sp
    _move_optional(
        planner,
        sapien.Pose(cur.p + np.array([0.0, 0.0, cfg.box_release_clear_up]), cur.q),
        label=f"{label_prefix} release clear above box",
    )
    _wait(planner, steps=8)
    _wait(planner, steps=10)

    side = -1.0 if planner.agent_idx == 0 else 1.0
    cur = planner.agent.tcp.pose.sp
    _move_optional(
        planner,
        sapien.Pose(cur.p + np.array([0.0, side * cfg.box_retreat_side, 0.0]), cur.q),
        label=f"{label_prefix} box retreat side",
    )
    cur = planner.agent.tcp.pose.sp
    _move_optional(
        planner,
        sapien.Pose(cur.p + np.array([0.0, 0.0, cfg.retreat_up]), cur.q),
        label=f"{label_prefix} box retreat up",
    )
    _wait(planner, steps=8)


def _stage_mug(planner, cfg, mug, middle_world, label_prefix):
    grasp, _ = _pick_mug(planner, cfg, mug, label_prefix)
    _place_mug_on_middle(planner, cfg, mug, middle_world, grasp.q, label_prefix)


def _stage_mug_no_retreat(planner, cfg, mug, middle_world, label_prefix):
    grasp, _ = _pick_mug(planner, cfg, mug, label_prefix)
    _place_mug_on_middle(
        planner, cfg, mug, middle_world, grasp.q, label_prefix, retreat=False
    )


def _receive_mug_and_box(
    planner, cfg, base_env, mug, box_bottom_world, label_prefix, clean_wrist=False
):
    grasp, _ = _pick_mug(
        planner, cfg, mug, label_prefix, clean_wrist=clean_wrist
    )
    _place_mug_in_box(
        planner, cfg, base_env, mug, box_bottom_world, grasp.q, label_prefix
    )


def object_tilt_deg(actor):
    """Tilt of an upright mug from vertical, in degrees (0 = perfectly upright)."""
    R = actor.pose.sp.to_transformation_matrix()[:3, :3]
    best = max(abs(R[2, ax]) / (np.linalg.norm(R[:, ax]) + 1e-9) for ax in range(3))
    return float(np.degrees(np.arccos(np.clip(best, 0.0, 1.0))))


def mugs_upright(base_env, max_deg=8.0):
    return (
        object_tilt_deg(base_env.left_mug) <= max_deg
        and object_tilt_deg(base_env.right_mug) <= max_deg
    )


def bottles_upright(base_env, max_deg=8.0):
    # Backward-compatible alias used by the dataset / demo helpers.
    return mugs_upright(base_env, max_deg=max_deg)


def solve(env, seed=None, debug=False, vis=False, frame_cb=None, **kwargs):
    rec = _find_record_wrapper(env)
    config = kwargs.pop("config", None)
    labels = kwargs.pop("labels", None)
    try:
        return _solve(
            env,
            rec,
            seed=seed,
            debug=debug,
            vis=vis,
            frame_cb=frame_cb,
            config=config,
            labels=labels,
        )
    except RuntimeError as exc:
        if debug:
            print(f"  motion planning failed: {exc}")
        if rec is not None:
            rec.save_trajectory = True
        env.reset(seed=seed)
        return -1


def _solve(
    env,
    rec,
    seed=None,
    debug=False,
    vis=False,
    frame_cb=None,
    config=None,
    labels=None,
):
    base_env = env.unwrapped
    cfg = config or BottleExchangeConfig()
    label_left_1, label_right_1, label_left_2, label_right_2 = labels or (
        LABEL_LEFT_1,
        LABEL_RIGHT_1,
        LABEL_LEFT_2,
        LABEL_RIGHT_2,
    )
    gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
    left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis)
    right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis)
    planners = [left, right]
    _configure_motion_smoothing(planners, cfg)

    middle_left_to_right = np.asarray(base_env.workspace_offset, dtype=np.float64) + np.array(
        [
            cfg.left_to_right_middle_rel_xy[0],
            cfg.left_to_right_middle_rel_xy[1],
            cfg.middle_stage_z,
        ]
    )
    middle_right_to_left = np.asarray(base_env.workspace_offset, dtype=np.float64) + np.array(
        [
            cfg.right_to_left_middle_rel_xy[0],
            cfg.right_to_left_middle_rel_xy[1],
            cfg.middle_stage_z,
        ]
    )

    saved_flags = None
    if rec is not None:
        saved_flags = (rec.save_trajectory, rec._save_video)
        rec.save_trajectory = False
        rec._save_video = False

    try:
        env.reset(seed=seed)
        _wait(left, steps=SETTLE)
        home_l = _arm_entry(left, MultiPandaArmPlanner.OPEN).copy()
        home_r = _arm_entry(right, MultiPandaArmPlanner.OPEN).copy()

        right_box = _np_pos(base_env.right_box_parts[0])
        left_box = _np_pos(base_env.left_box_parts[0])

        l1a, r1a = _record_segment(
            planners,
            gripper_states,
            0,
            lambda: _stage_mug_no_retreat(
                left,
                cfg,
                base_env.left_mug,
                middle_left_to_right,
                "left mug to middle",
            ),
        )
        l1_retreat, r1_retreat_static = _record_segment(
            planners,
            gripper_states,
            0,
            lambda: _retreat_after_middle(left, cfg, "left mug to middle"),
        )
        l1b, r1b = _record_segment(
            planners,
            gripper_states,
            1,
            lambda: _receive_mug_and_box(
                right,
                cfg,
                base_env,
                base_env.left_mug,
                right_box,
                "right staged mug to right box",
            ),
        )
        # Continue directly from the arms' phase-1 retreat poses.  Sending both
        # arms back to their initial joint targets here caused an unnecessary,
        # visibly fast reset before the second cup.
        l2a, r2a = _record_segment(
            planners,
            gripper_states,
            1,
            lambda: _stage_mug_no_retreat(
                right,
                cfg,
                base_env.right_mug,
                middle_right_to_left,
                "right mug to middle",
            ),
        )
        l2_retreat_static, r2_retreat = _record_segment(
            planners,
            gripper_states,
            1,
            lambda: _retreat_after_middle(right, cfg, "right mug to middle"),
        )
        l2b, r2b = _record_segment(
            planners,
            gripper_states,
            0,
            lambda: _receive_mug_and_box(
                left,
                cfg,
                base_env,
                base_env.right_mug,
                left_box,
                "left staged mug to left box",
                clean_wrist=True,
            ),
        )
    finally:
        if rec is not None:
            rec.save_trajectory, rec._save_video = saved_flags

    env.reset(seed=seed)
    left_full = (
        [home_l.copy() for _ in range(SETTLE)]
        + l1a
        + l1_retreat
        + l1b
        + l2a
        + l2_retreat_static
        + l2b
    )
    right_full = (
        [home_r.copy() for _ in range(SETTLE)]
        + r1a
        + r1_retreat_static
        + r1b
        + r2a
        + r2_retreat
        + r2b
    )
    phase1_end = SETTLE + len(l1a) + len(l1_retreat) + len(l1b)
    left_second_pick_start = phase1_end + len(l2a) + len(l2_retreat_static)
    left_full += [left_full[-1].copy() for _ in range(FINAL_HOLD)]
    right_full += [right_full[-1].copy() for _ in range(FINAL_HOLD)]
    total_steps = len(left_full)

    base_env._subgoal_segments = {
        "phase1_end": int(phase1_end),
        "left_second_pick_start": int(left_second_pick_start),
        "end": int(total_steps),
        "left": [
            {"label": label_left_1, "start": 0, "end": int(phase1_end)},
            {"label": label_left_2, "start": int(phase1_end), "end": int(total_steps)},
        ],
        "right": [
            {"label": label_right_1, "start": 0, "end": int(phase1_end)},
            {"label": label_right_2, "start": int(phase1_end), "end": int(total_steps)},
        ],
    }
    base_env._coplay_tapes = (left_full, right_full, phase1_end, total_steps)
    base_env._action_tape = np.asarray(
        [np.hstack([la, ra]) for la, ra in zip(left_full, right_full)]
    )

    def _labels(i):
        if i < phase1_end:
            return 1, label_left_1, label_right_1
        return 2, label_left_2, label_right_2

    last = _coplay(env, left_full, right_full, frame_cb=frame_cb, labels=_labels)
    if debug:
        ev = base_env.evaluate()
        print(
            f"  [done] success={bool(ev['success'][0].item())} "
            f"L_stage={bool(ev['left_mug_staged_once'][0].item())} "
            f"R_stage={bool(ev['right_mug_staged_once'][0].item())} "
            f"L_box={bool(ev['left_mug_in_right_box'][0].item())} "
            f"R_box={bool(ev['right_mug_in_left_box'][0].item())}"
        )
    return last


def left_second_pick_wrist_motion(base_env):
    """Return joint-7 path/net rotation before the left arm's second grasp."""
    actions = np.asarray(base_env._action_tape)
    start = int(base_env._subgoal_segments["left_second_pick_start"])
    close = np.flatnonzero(actions[start:, 7] < 0)
    if len(close) == 0:
        return float("inf"), float("inf")
    end = start + int(close[0])
    wrist = np.unwrap(actions[start : end + 1, 6])
    path_deg = float(np.degrees(np.abs(np.diff(wrist)).sum()))
    net_deg = float(np.degrees(abs(wrist[-1] - wrist[0])))
    return path_deg, net_deg
