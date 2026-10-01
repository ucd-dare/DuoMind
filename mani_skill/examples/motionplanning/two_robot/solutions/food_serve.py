"""Keypoint solver for the revised ``TwoRobotFoodServeReplicaCAD-v1``.

The fixed tray/support sits toward the global camera and left robot. The right
robot relays round bread to the shared middle while the left robot places a
light-colored mug on the tray. The left robot then collects the relayed bread
and places it beside the mug:

* Phase 1: LEFT ``pick up mug and place on tray`` in parallel with RIGHT
  ``pick up bread and place in the middle``.
* Phase 2: LEFT ``pick up bread and place beside mug on tray`` while RIGHT
  remains clear.

As in the other two-robot solvers, each arm is first run solo to record joint
target tapes. The tapes are then replayed together into the recorded episode.
"""

from dataclasses import dataclass

import numpy as np
import sapien

from mani_skill.examples.motionplanning.two_robot.solutions.planner import (
    MultiPandaArmPlanner,
    make_grasp_pose as _make_grasp_pose,
    move_optional as _move_optional,
    move_or_fail as _move_or_fail,
    np_pos as _np_pos,
    try_pose_variants as _try_pose_variants,
    wait_steps as _wait,
)

LABEL_L1 = "L1: pick up mug and place on tray"
LABEL_L2 = "L2: pick up bread and place beside mug on tray"
LABEL_R1 = "R1: pick up bread and place in the middle"

SETTLE = 12
FINAL_HOLD = 35


@dataclass
class FoodServeConfig:
    # RoboTwin bread's actor origin is at its bottom; grasp at body centre.
    bread_grasp_z: float = 0.012
    bread_pregrasp: float = 0.09
    bread_lift: float = 0.20
    bread_transit_z: float = 0.23
    middle_rel_xy: tuple = (-0.10, 0.0)
    bread_stage_clearance: float = 0.055

    # Top-down side pinch of the upright mug body. The closing axis is derived
    # from the randomized mug pose so it always stays perpendicular to the handle.
    mug_grasp_z: float = 0.035
    mug_pregrasp: float = 0.10
    mug_lift: float = 0.18

    # Put the two objects side-by-side around the fixed tray centre.
    mug_tray_offset: tuple = (-0.070, 0.0)
    bread_tray_offset: tuple = (0.075, 0.055)
    # RoboCasa tray pose is above its support-contact reference; RoboTwin actor
    # origins are at the object bottoms. A small negative relative offset places
    # the bottom directly at the rendered tray surface instead of dropping it.
    mug_on_tray_z: float = -0.004
    bread_on_tray_z: float = 0.010


def _top_grasp_pose(planner, center, closing):
    pose = _make_grasp_pose(
        planner.agent,
        approaching=np.array([0, 0, -1]),
        closing=np.asarray(closing, dtype=np.float64),
        center=np.asarray(center, dtype=np.float64),
    )
    return _try_pose_variants(planner, pose)


def _upright_mug_body_closing_axis(mug):
    """World closing axis across the mug body, perpendicular to its handle.

    The RoboTwin mug's handle extends along mesh-local x. Its upright local z
    axis is horizontal after the +90-degree x rotation, so pinching along local
    z avoids the handle for every randomized world-yaw pose.
    """
    q = np.asarray(mug.pose.sp.q, dtype=np.float64)
    w, x, y, z = q
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    closing = rotation[:, 2]
    closing[2] = 0.0
    return closing / np.linalg.norm(closing)


def _move_orientation_locked(planner, pose, refine_steps=0, label="fixed-pose move"):
    """Use a Cartesian move without an RRT wrist-orientation detour."""
    print(f"Planning {label}: p={np.array2string(np.asarray(pose.p), precision=4)}")
    result = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
    if result == -1:
        raise RuntimeError(f"orientation-locked motion planning failed for {label}")
    return result


def _clone_state(state):
    if isinstance(state, dict):
        return {k: _clone_state(v) for k, v in state.items()}
    return state.clone() if hasattr(state, "clone") else state


def _pad(tape, n):
    if len(tape) >= n:
        return list(tape[:n])
    if not tape:
        raise RuntimeError("cannot pad an empty tape")
    return list(tape) + [tape[-1]] * (n - len(tape))


def _right_bread_to_middle(right, cfg, base_env, bread_rest_z, middle_world):
    bread_pos = _np_pos(base_env.bread)
    grasp = _top_grasp_pose(
        right,
        bread_pos + np.array([0, 0, cfg.bread_grasp_z]),
        closing=[0, 1, 0],
    )
    pregrasp = sapien.Pose(grasp.p + np.array([0, 0, cfg.bread_pregrasp]), grasp.q)
    right.open_gripper(steps=8)
    _move_orientation_locked(right, pregrasp, label="right bread pregrasp")
    _move_or_fail(right, grasp, label="right bread grasp", refine_steps=10)
    right.close_gripper(steps=50)
    _move_orientation_locked(
        right,
        sapien.Pose(grasp.p + np.array([0, 0, cfg.bread_lift]), grasp.q),
        label="right bread lift",
    )

    bread_in_tcp = _np_pos(base_env.bread) - right.agent.tcp.pose.sp.p
    stage_world = np.array([middle_world[0], middle_world[1], bread_rest_z + 0.002])
    stage_tcp = stage_world - bread_in_tcp
    _move_orientation_locked(
        right,
        sapien.Pose(stage_tcp + np.array([0, 0, cfg.bread_transit_z]), grasp.q),
        label="bread transit above middle",
    )
    _move_orientation_locked(
        right,
        sapien.Pose(stage_tcp + np.array([0, 0, cfg.bread_stage_clearance]), grasp.q),
        label="bread stage hover",
        refine_steps=8,
    )
    _move_orientation_locked(right, sapien.Pose(stage_tcp, grasp.q), label="bread stage")
    right.open_gripper(steps=14)
    _move_optional(
        right,
        sapien.Pose(
            right.agent.tcp.pose.sp.p + np.array([0, 0.30, 0.20]),
            grasp.q,
        ),
        use_rrt=True,
        label="right clear middle",
    )
    _wait(right, steps=18)


def _left_mug_to_tray(left, cfg, base_env):
    mug_pos = _np_pos(base_env.mug)
    grasp = _top_grasp_pose(
        left,
        mug_pos + np.array([0, 0, cfg.mug_grasp_z]),
        closing=_upright_mug_body_closing_axis(base_env.mug),
    )
    pregrasp = sapien.Pose(grasp.p + np.array([0, 0, cfg.mug_pregrasp]), grasp.q)
    left.open_gripper(steps=8)
    _move_orientation_locked(left, pregrasp, label="left mug pregrasp")
    _move_or_fail(left, grasp, label="left mug side grasp", refine_steps=12)
    left.close_gripper(steps=42)
    _move_orientation_locked(
        left,
        sapien.Pose(grasp.p + np.array([0, 0, cfg.mug_lift]), grasp.q),
        label="left mug lift",
        refine_steps=10,
    )

    mug_in_tcp = _np_pos(base_env.mug) - left.agent.tcp.pose.sp.p
    tray_pos = _np_pos(base_env.tray)
    desired_mug_pos = tray_pos + np.array(
        [cfg.mug_tray_offset[0], cfg.mug_tray_offset[1], cfg.mug_on_tray_z]
    )
    place_tcp = desired_mug_pos - mug_in_tcp
    _move_orientation_locked(
        left,
        sapien.Pose(place_tcp + np.array([0, 0, 0.12]), grasp.q),
        label="mug above tray",
    )
    _move_orientation_locked(
        left,
        sapien.Pose(place_tcp, grasp.q),
        label="mug place on tray",
        refine_steps=14,
    )
    _wait(left, steps=8)
    left.open_gripper(steps=30)
    _wait(left, steps=10)
    _move_optional(
        left,
        sapien.Pose(left.agent.tcp.pose.sp.p + np.array([0, 0, 0.16]), grasp.q),
        label="left clear mug vertically",
    )
    _wait(left, steps=20)


def _left_bread_to_tray(left, cfg, base_env):
    left.open_gripper(steps=8)
    bread_pos = _np_pos(base_env.bread)
    expected_middle = np.asarray(base_env.workspace_offset, dtype=np.float64)[:2] + np.asarray(
        cfg.middle_rel_xy, dtype=np.float64
    )
    if np.linalg.norm(bread_pos[:2] - expected_middle) > 0.30:
        raise RuntimeError(
            "staged bread is outside the shared handoff region: "
            f"xy={np.array2string(bread_pos[:2], precision=4)}"
        )
    grasp = _top_grasp_pose(
        left,
        bread_pos + np.array([0, 0, cfg.bread_grasp_z]),
        closing=[1, 0, 0],
    )
    pregrasp = sapien.Pose(grasp.p + np.array([0, 0, cfg.bread_pregrasp]), grasp.q)
    _move_orientation_locked(left, pregrasp, label="left bread pregrasp")
    _wait(left, steps=8)

    # Re-measure after approaching because the staged bread may still be settling.
    bread_pos = _np_pos(base_env.bread)
    grasp = sapien.Pose(
        np.array([bread_pos[0], bread_pos[1], bread_pos[2] + cfg.bread_grasp_z]),
        grasp.q,
    )
    _move_or_fail(left, grasp, label="left bread grasp", refine_steps=12)
    left.close_gripper(steps=50)
    _move_orientation_locked(
        left,
        sapien.Pose(grasp.p + np.array([0, 0, cfg.bread_lift]), grasp.q),
        label="left bread lift",
    )

    bread_in_tcp = _np_pos(base_env.bread) - left.agent.tcp.pose.sp.p
    tray_pos = _np_pos(base_env.tray)
    desired_bread_pos = tray_pos + np.array(
        [cfg.bread_tray_offset[0], cfg.bread_tray_offset[1], cfg.bread_on_tray_z]
    )
    place_tcp = desired_bread_pos - bread_in_tcp
    _move_orientation_locked(
        left,
        sapien.Pose(place_tcp + np.array([0, 0, 0.12]), grasp.q),
        label="bread above tray",
    )
    _move_orientation_locked(
        left,
        sapien.Pose(place_tcp, grasp.q),
        label="bread place beside mug",
        refine_steps=10,
    )
    left.open_gripper(steps=20)
    _move_optional(
        left,
        sapien.Pose(left.agent.tcp.pose.sp.p + np.array([0, -0.14, 0.18]), grasp.q),
        use_rrt=True,
        label="left retreat from tray",
    )
    _wait(left, steps=35)


def _find_record_wrapper(env):
    e = env
    while e is not None:
        if e.__class__.__name__ == "RecordEpisode":
            return e
        e = getattr(e, "env", None)
    return None


def solve(env, seed=None, debug=False, vis=False, frame_cb=None):
    base_env = env.unwrapped
    cfg = FoodServeConfig()
    gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
    left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis)
    right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis)

    def restore(state):
        base_env.set_state_dict(state)
        gripper_states[0] = MultiPandaArmPlanner.OPEN
        gripper_states[1] = MultiPandaArmPlanner.OPEN

    rec = _find_record_wrapper(env)
    saved_flags = None
    if rec is not None:
        saved_flags = (rec.save_trajectory, rec._save_video)
        rec.save_trajectory = False
        rec._save_video = False
    try:
        env.reset(seed=seed)
        _wait(right, steps=SETTLE)
        bread_rest_z = float(_np_pos(base_env.bread)[2])
        middle_world = np.array(base_env.workspace_offset, dtype=np.float64) + np.array(
            [cfg.middle_rel_xy[0], cfg.middle_rel_xy[1], 0.0]
        )
        state0 = _clone_state(base_env.get_state_dict())

        restore(state0)
        right.start_recording()
        _right_bread_to_middle(right, cfg, base_env, bread_rest_z, middle_world)
        right_phase1 = right.stop_recording()

        restore(state0)
        left.start_recording()
        _left_mug_to_tray(left, cfg, base_env)
        left_phase1 = left.stop_recording()

        restore(state0)
        _coplay(env, left_phase1, right_phase1)
        state1 = _clone_state(base_env.get_state_dict())
        if debug:
            print(
                "  [pass1] mug_on_tray="
                f"{bool(base_env._mug_on_tray().item())} "
                f"bread={_np_pos(base_env.bread)} mug={_np_pos(base_env.mug)}"
            )

        restore(state1)
        left.start_recording()
        _left_bread_to_tray(left, cfg, base_env)
        left_phase2 = left.stop_recording()
    finally:
        if rec is not None:
            rec.save_trajectory, rec._save_video = saved_flags

    env.reset(seed=seed)
    base_env._coplay_start = _clone_state(base_env.get_state_dict())
    home_left = np.hstack([left._arm_qpos(left.agent), MultiPandaArmPlanner.OPEN])
    home_right = np.hstack([right._arm_qpos(right.agent), MultiPandaArmPlanner.OPEN])

    phase1_len = max(len(left_phase1), len(right_phase1))
    left1 = _pad(left_phase1, phase1_len)
    right1 = _pad(right_phase1, phase1_len)
    left2 = list(left_phase2)
    right2 = [right1[-1]] * len(left2)

    left_full = [home_left] * SETTLE + left1 + left2
    right_full = [home_right] * SETTLE + right1 + right2
    phase1_end = SETTLE + phase1_len
    left_full += [left_full[-1]] * FINAL_HOLD
    right_full += [right_full[-1]] * FINAL_HOLD
    total = len(left_full)

    def labels(i):
        if i < phase1_end:
            return 1, LABEL_L1, LABEL_R1
        return 2, LABEL_L2, LABEL_R1

    base_env._coplay_tapes = (
        [np.asarray(a) for a in left_full],
        [np.asarray(a) for a in right_full],
        int(phase1_end),
        int(total),
    )
    base_env._subgoal_segments = {
        "phase1_end": int(phase1_end),
        "end": int(total),
        "left": [
            {"label": LABEL_L1, "start": 0, "end": int(phase1_end)},
            {"label": LABEL_L2, "start": int(phase1_end), "end": int(total)},
        ],
        "right": [
            {"label": LABEL_R1, "start": 0, "end": int(total)},
        ],
    }

    last = _coplay(env, left_full, right_full, frame_cb=frame_cb, labels=labels)
    if debug:
        info = last[-1]
        print(
            "  [pass2] success="
            f"{bool(info['success'].item())} mug_on_tray="
            f"{bool(info['mug_on_tray'].item())} bread_on_tray="
            f"{bool(info['bread_on_tray'].item())}"
        )
    return last


def _coplay(env, left_tape, right_tape, frame_cb=None, labels=None):
    """Step both arms' equal-format [qpos(7), gripper] target tapes."""
    n = max(len(left_tape), len(right_tape))
    last = None
    for i in range(n):
        la = left_tape[min(i, len(left_tape) - 1)]
        ra = right_tape[min(i, len(right_tape) - 1)]
        last = env.step(np.hstack([la, ra]))
        if frame_cb is not None and labels is not None:
            phase, left_label, right_label = labels(i)
            frame_cb(i, n, phase, left_label, right_label)
    return last
