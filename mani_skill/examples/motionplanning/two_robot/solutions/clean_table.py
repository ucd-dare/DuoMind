"""Keypoint motion-planning solver for the ``TwoRobotCleanTable`` tasks.

Object layout (workspace_offset = (1.4, -1.2, 0.92); arms mounted at table
height; -y is the LEFT arm / agents[0], +y is the RIGHT arm / agents[1]):

* LEFT cube (red) ~ world y in [-1.72, -1.52] -> in front of the LEFT arm
* RIGHT cube (green) ~ world y in [-0.88, -0.68] -> in front of the RIGHT arm
* basket ~ world (0.94, -0.58)         -> on the RIGHT, reachable only by the right arm
* table middle ~ world (1.35, -1.20)   -> reachable by *both* arms (relay zone)

Cooperative plan (matches the requested instructions), scheduled as TWO phases:

* Phase 1  (R1 ∥ L1, both arms move at once):
    - RIGHT: "pick <color> cube and place in box" -- clears its own cubes into the basket
    - LEFT : "pick <color> cube and place at middle" -- relays its
      cube(s) to the shared middle so the right arm can reach them
* Phase 2  (R2, left idle):
    - RIGHT: "pick <color> cube and place in box" -- picks the relayed cube(s) from
      the middle and drops them in the basket

The task has one cube per side. The solver discovers the per-side cubes from the
environment and schedules the relay and direct placement from that declaration.

Simultaneous execution mirrors ``food_serve.py``: each arm is run SOLO (the
other frozen) to record a per-step "tape" of joint targets, then the two tapes
are replayed together so both robots move at once. The solo runs keep all the
physics-based re-measurement that makes the contact-rich grasps reliable.
"""

from dataclasses import dataclass

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.two_robot.solutions.planner import (
    MultiPandaArmPlanner,
    make_grasp_pose as _make_grasp_pose,
    move_optional as _move_optional,
    move_or_fail as _move_or_fail,
    np_pos as _np_pos,
    tcp_closing as _tcp_closing,
    try_pose_variants as _try_pose_variants,
    wait_steps as _wait,
)

# Subgoal instruction templates (natural-language, for sub-goal-conditioned
# training). Labels name the cube being manipulated and switch at the exact end
# of each per-cube pick/place primitive. There is no "idle" label: during padding
# or a final hold, an arm keeps its most recent instruction.
LABEL_RIGHT = "pick {color} cube and place in box"
LABEL_LEFT = "pick {color} cube and place at middle"

# Steps spent letting the scene settle at the start of the recorded episode.
SETTLE = 12
# Steps held still at the very end so the success metric reads a static scene.
FINAL_HOLD = 30


@dataclass
class CleanTableConfig:
    # Cube top-grasp. The task cubes are 0.0512 m wide after the requested 0.8
    # scale. The gripper still opens to its widest command before every grasp,
    # and the jaws align to a cube face before closing.
    cube_grasp_z: float = -0.01
    cube_pregrasp: float = 0.07
    # small, reliably reachable settle-lift right after the grasp (a tall
    # straight-up lift can exceed the arm's reach for far-forward cubes; the
    # transit/hover move re-plans the real climb toward the basket / middle).
    cube_lift: float = 0.06
    # altitude the cube is carried at while in transit (above everything)
    transit_z: float = 0.20
    # --- basket drop ---
    # distinct in-basket slots (relative to the basket centre xy) so cubes do
    # not stack on top of each other and bounce out. Spread to the four corners.
    box_slots: tuple = (
        (-0.065, -0.055),
        (0.065, 0.055),
        (0.065, -0.055),
        (-0.065, 0.055),
    )
    # Cube-centre height above the basket bottom at release. The cube bottom is
    # 13 mm above the 130 mm rim at this height, so neither the held cube nor the
    # gripper enters/hits the box. The cube settles into its slot under gravity.
    box_drop_z: float = 0.16
    box_hover_z: float = 0.22
    # --- middle relay ---
    # where on the table cubes are relayed (relative to workspace centre). The
    # relay point sits at the workspace centre, directly in front of both arm
    # bases (x = 1.4) -- this is the only strip both arms can reach, and x = 0
    # minimises the (edge-of-envelope) reach for each.
    middle_rel_xy: tuple = (0.0, 0.0)
    # spacing between relay slots when there is more than one relayed cube
    # (offset along x so both sit in the shared reachable strip).
    middle_slot_dx: float = 0.16
    middle_place_clear: float = 0.06
    middle_hover_z: float = 0.18


def _cube_face_closing(cube):
    """Horizontal closing axis aligned with one of the cube's vertical faces.

    Cubes spawn with a random yaw, so the gripper must align its jaws with a face
    (not the diagonal, whose width exceeds the gripper opening). Returns the
    cube's local x-axis projected onto the table plane.
    """
    rot = cube.pose.sp.to_transformation_matrix()[:3, :3]
    closing = np.asarray(rot[:, 0], dtype=np.float64)
    closing[2] = 0.0
    if np.linalg.norm(closing) < 1e-6:
        closing = np.array([1.0, 0.0, 0.0])
    return closing / np.linalg.norm(closing)


def _top_grasp_pose(planner, center, closing=None):
    if closing is None:
        closing = _tcp_closing(planner.agent)
    base = _make_grasp_pose(
        planner.agent, approaching=np.array([0, 0, -1]), closing=closing, center=center
    )
    # A cube top-grasp is 4-fold ambiguous in wrist yaw (0, ±90, 180 all align the
    # jaws with a face). Pick the reachable variant whose closing (jaw) axis is
    # CLOSEST to the gripper's current closing axis, so the gripper does the
    # smallest possible reorientation -- no big meaningless spin on the way from
    # the basket to the next cube. (Comparing the cartesian jaw axes is robust;
    # the jaws are 180-degree symmetric, hence the abs(dot).)
    cur_close = _tcp_closing(planner.agent)
    cur_close = cur_close / (np.linalg.norm(cur_close) + 1e-9)
    cands = []
    for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        cand = base * sapien.Pose(q=euler2quat(0, 0, yaw))
        cand_close = cand.to_transformation_matrix()[:3, 1]
        cand_close = cand_close / (np.linalg.norm(cand_close) + 1e-9)
        ang = np.arccos(np.clip(abs(float(np.dot(cur_close, cand_close))), 0.0, 1.0))
        cands.append((ang, cand))
    cands.sort(key=lambda c: c[0])
    for _, cand in cands:
        if planner.move_to_pose_with_screw(cand, dry_run=True) != -1:
            return cand
    return cands[0][1]


def _move_clean_pregrasp(arm, pose, candidates=3):
    """Execute the cleanest of several collision-aware RRT pregrasp plans.

    MPLib's RRT branch is randomized. The first successful branch can wind a
    Panda joint far around even when a direct-looking branch also exists. Sample
    a few valid plans without changing simulator state, score their excess joint
    travel plus wrist travel, and execute only the best one.
    """
    print(f"Planning cube pregrasp: p={np.array2string(np.asarray(pose.p), precision=4)}")
    plans = []
    for _ in range(candidates):
        result = arm.move_to_pose(pose, dry_run=True, attempts=2)
        if result == -1:
            continue
        q = np.unwrap(np.asarray(result["position"], dtype=np.float64)[:, :7], axis=0)
        path = np.abs(np.diff(q, axis=0)).sum(axis=0)
        net = np.abs(q[-1] - q[0])
        winding = float(np.max(path - net))
        wrist_path = float(path[6])
        plans.append((winding + 0.2 * wrist_path, result))
    if not plans:
        raise RuntimeError("motion planning failed for cube pregrasp")
    plans.sort(key=lambda item: item[0])
    return arm.follow_path(plans[0][1])


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


def _cube_color(cube):
    """Return the color token from an actor named ``cleanup_cube_<color>``."""
    prefix = "cleanup_cube_"
    if not cube.name.startswith(prefix):
        raise ValueError(f"unexpected clean-table cube name: {cube.name!r}")
    return cube.name[len(prefix):].replace("_", " ")


def _segments(cubes, ends, label_template, start=0, final_end=None):
    """Build contiguous color-specific instruction segments.

    ``ends`` contains cumulative tape offsets measured immediately after each
    cube's full pick/place primitive. ``final_end`` extends the last instruction
    across synchronization padding/final settling so every frame stays labelled.
    """
    if len(cubes) != len(ends):
        raise ValueError(f"segment mismatch: {len(cubes)} cubes, {len(ends)} ends")
    segments = []
    cursor = int(start)
    for cube, end in zip(cubes, ends):
        segments.append({
            "label": label_template.format(color=_cube_color(cube)),
            "start": cursor,
            "end": int(start + end),
        })
        cursor = int(start + end)
    if segments and final_end is not None:
        segments[-1]["end"] = int(final_end)
    return segments


# ---------------------------------------------------------------------------
# Per-arm pick/place primitives (each drives ONLY its own arm; recorded as tape)
# ---------------------------------------------------------------------------

def _pick_cube(arm, cfg, cube):
    """Top-grasp ``cube`` at its current pose: open, pregrasp, descend, close, lift.

    Returns the grasp pose (so the caller keeps the wrist orientation for carry).
    """
    cube_pos = _np_pos(cube)
    grasp = _top_grasp_pose(
        arm, cube_pos + np.array([0, 0, cfg.cube_grasp_z]), closing=_cube_face_closing(cube)
    )
    pregrasp = sapien.Pose(grasp.p + np.array([0, 0, cfg.cube_pregrasp]), grasp.q)
    arm.open_gripper(steps=12)
    _move_clean_pregrasp(arm, pregrasp)
    # straight-down screw descent from directly above keeps the jaws centred on
    # the cube (xy precision matters at this tight clearance); refine to settle.
    _move_or_fail(arm, grasp, label="cube grasp", refine_steps=14)
    arm.close_gripper(steps=45)
    # A modest, non-fatal lift to clear the table; for far-forward cubes a tall
    # straight-up lift can exceed the arm's reach, so don't fail on it -- the
    # following transit/hover move re-plans the climb toward the (much closer)
    # basket or middle anyway.
    _move_optional(
        arm,
        sapien.Pose(grasp.p + np.array([0, 0, cfg.cube_lift]), grasp.q),
        use_rrt=False,
        label="cube lift",
    )
    return grasp


def _place_in_box(arm, cfg, cube, grasp_q, box_center, slot):
    """Carry the grasped cube above a basket slot, descend, release, retreat."""
    cube_in_tcp = _np_pos(cube) - arm.agent.tcp.pose.sp.p
    slot_xy = box_center[:2] + np.array(slot)

    def cube_to_tcp(z):
        return np.array([slot_xy[0], slot_xy[1], box_center[2] + z]) - cube_in_tcp

    # Carry to the basket with a SCREW (straight Cartesian) move, not RRT: a
    # straight line keeps the base joint turning the short way toward the basket
    # instead of winding the long way around (RRT picks an arbitrary branch).
    _move_or_fail(
        arm,
        sapien.Pose(cube_to_tcp(cfg.box_hover_z), grasp_q),
        use_rrt=False,
        label="box hover",
    )
    _move_or_fail(
        arm,
        sapien.Pose(cube_to_tcp(cfg.box_drop_z), grasp_q),
        use_rrt=False,
        label="box drop",
    )
    arm.open_gripper(steps=16)
    # retreat straight up so the gripper clears the basket walls.
    _move_optional(
        arm,
        sapien.Pose(arm.agent.tcp.pose.sp.p + np.array([0, 0, 0.16]), grasp_q),
        use_rrt=False,
        label="box retreat",
    )
    _wait(arm, steps=8)


def _place_at_middle(arm, cfg, cube, grasp_q, rest_z, middle_world):
    """Relay the grasped cube to a middle slot, set it down, release, retreat."""
    cube_in_tcp = _np_pos(cube) - arm.agent.tcp.pose.sp.p
    target = np.array([middle_world[0], middle_world[1], rest_z + 0.002])

    def cube_to_tcp(z):
        return np.array([target[0], target[1], z]) - cube_in_tcp

    _move_or_fail(
        arm,
        sapien.Pose(cube_to_tcp(target[2] + cfg.transit_z), grasp_q),
        use_rrt=False,
        label="middle transit",
    )
    _move_or_fail(
        arm,
        sapien.Pose(cube_to_tcp(target[2] + cfg.middle_place_clear), grasp_q),
        use_rrt=False,
        label="middle above",
    )
    _move_or_fail(
        arm, sapien.Pose(cube_to_tcp(target[2]), grasp_q), use_rrt=False, label="middle place"
    )
    arm.open_gripper(steps=14)
    # retreat up and back toward this arm's own side to clear the shared middle.
    side_sign = -1.0 if arm.agent_idx == 0 else 1.0
    _move_optional(
        arm,
        sapien.Pose(
            arm.agent.tcp.pose.sp.p + np.array([0, side_sign * 0.20, 0.18]), grasp_q
        ),
        use_rrt=False,
        label="middle retreat",
    )
    _wait(arm, steps=12)


def _right_clear_side(right, cfg, right_cubes, box_center):
    """R1: right arm drops each of its own cubes into the basket, one by one."""
    ends = []
    for i, cube in enumerate(right_cubes):
        grasp = _pick_cube(right, cfg, cube)
        _place_in_box(right, cfg, cube, grasp.q, box_center, cfg.box_slots[i])
        ends.append(len(right.tape))
    return ends


def _left_relay_to_middle(left, cfg, left_cubes, rest_z, middle_slots):
    """L1: left arm relays each of its cubes to a middle slot, one by one."""
    ends = []
    for cube, middle in zip(left_cubes, middle_slots):
        grasp = _pick_cube(left, cfg, cube)
        _place_at_middle(left, cfg, cube, grasp.q, rest_z, middle)
        ends.append(len(left.tape))
    return ends


def _right_collect_middle(right, cfg, relayed_cubes, box_center, start_slot):
    """R2: right arm picks the relayed cube(s) from the middle into the basket."""
    ends = []
    for i, cube in enumerate(relayed_cubes):
        grasp = _pick_cube(right, cfg, cube)
        _place_in_box(right, cfg, cube, grasp.q, box_center, cfg.box_slots[start_slot + i])
        ends.append(len(right.tape))
    return ends


# ---------------------------------------------------------------------------
# Solver: record per-arm tapes, then co-play both arms simultaneously
# ---------------------------------------------------------------------------

def _find_record_wrapper(env):
    e = env
    while e is not None:
        if e.__class__.__name__ == "RecordEpisode":
            return e
        e = getattr(e, "env", None)
    return None


def _side_cubes(base_env):
    left, right = [], []
    sides = getattr(base_env, "item_sides", ["left"] * len(base_env.items))
    for cube, side in zip(base_env.items, sides):
        (left if side == "left" else right).append(cube)
    return left, right


def _shuffle_right_cubes(right_cubes, seed):
    """Return a seed-deterministic random pickup order for right-side cubes.

    This remains generic for any future multi-cube declaration. The current task
    has only green on the right, so this is a no-op.
    """
    cubes = list(right_cubes)
    if len(cubes) > 1:
        order = np.random.default_rng(seed).permutation(len(cubes))
        cubes = [cubes[int(i)] for i in order]
    return cubes


def solve(env, seed=None, debug=False, vis=False, **kwargs):
    """Solve one episode. Returns the final ``env.step`` tuple on success, or
    ``-1`` if motion planning fails (after leaving a clean recorded reset so the
    caller can flush an empty trajectory)."""
    rec = _find_record_wrapper(env)
    try:
        return _solve(env, rec, seed=seed, debug=debug, vis=vis, **kwargs)
    except RuntimeError as exc:
        if debug:
            print(f"  motion planning failed: {exc}")
        # Guarantee the record buffer exists so run.py's flush(save=False) on a
        # failed seed does not crash on a None trajectory buffer.
        if rec is not None:
            rec.save_trajectory = True
        env.reset(seed=seed)
        return -1


def _solve(env, rec, seed=None, debug=False, vis=False, **kwargs):
    base_env = env.unwrapped
    cfg = CleanTableConfig()
    gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
    left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis)
    right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis)

    def restore(state):
        base_env.set_state_dict(state)
        gripper_states[0] = MultiPandaArmPlanner.OPEN
        gripper_states[1] = MultiPandaArmPlanner.OPEN

    # NOTE: ``save_video`` is a read-only property backed by ``_save_video``;
    # toggle the backing field (assigning ``save_video`` raises AttributeError).
    saved_flags = None
    if rec is not None:
        saved_flags = (rec.save_trajectory, rec._save_video)
        rec.save_trajectory = False
        rec._save_video = False
    try:
        env.reset(seed=seed)
        _wait(right, steps=SETTLE)

        left_cubes, right_cubes = _side_cubes(base_env)
        right_cubes = _shuffle_right_cubes(right_cubes, seed)
        box_center = base_env.basket_center[0].cpu().numpy().astype(np.float64)
        rest_z = float(_np_pos(left_cubes[0])[2])
        middle0 = np.array(base_env.workspace_offset, dtype=np.float64) + np.array(
            [cfg.middle_rel_xy[0], cfg.middle_rel_xy[1], 0.0]
        )
        # one middle slot per left cube, spread along x around the relay centre.
        n_left = len(left_cubes)
        middle_slots = [
            middle0 + np.array([(j - (n_left - 1) / 2.0) * cfg.middle_slot_dx, 0.0, 0.0])
            for j in range(n_left)
        ]
        S0 = _clone_state(base_env.get_state_dict())

        # --- Phase 1 solo tapes (independent: right's own cubes vs left's cubes) ---
        restore(S0)
        right.start_recording()
        right_phase1_ends = _right_clear_side(right, cfg, right_cubes, box_center)
        rA = right.stop_recording()

        restore(S0)
        left.start_recording()
        left_phase1_ends = _left_relay_to_middle(
            left, cfg, left_cubes, rest_z, middle_slots
        )
        lA = left.stop_recording()

        # --- reach the real post-phase-1 state so phase 2 can be planned ---
        restore(S0)
        _coplay(env, lA, rA)
        if debug:
            print(
                "  [pass1] after co-play A: items_in_basket="
                f"{int(base_env._items_in_basket()[0].sum().item())}/{len(base_env.items)}"
            )
    finally:
        if rec is not None:
            rec.save_trajectory, rec._save_video = saved_flags

    # ==================================================================
    # Clean recorded pass:
    #   * Phase 1: co-play the two solo tapes so BOTH arms move at once.
    #   * Phase 2: plan R2 LIVE against the real post-phase-1 scene. Pre-baking
    #     R2 as a tape (relative to a *predicted* state) is fragile -- the global
    #     PhysX solve diverges slightly when the second arm is also moving, so the
    #     relayed cube ends up a couple cm off and the baked grasp misses. Live
    #     planning reads the cube's true pose, so the right arm always finds it.
    # ==================================================================
    env.reset(seed=seed)
    background_props = []
    seen_background = set()
    for actor in base_env.replicacad_scene.movable_objects.values():
        if id(actor) in seen_background:
            continue
        if any(
            token in actor.name
            for token in base_env.hidden_replicacad_name_substrings
        ):
            continue
        seen_background.add(id(actor))
        background_props.append(actor)
    background_initial = {id(actor): _np_pos(actor).copy() for actor in background_props}
    max_background_displacement = 0.0
    max_background_speed = 0.0
    max_background_displacement_prop = ""
    max_background_speed_prop = ""
    home_l = np.hstack([left._arm_qpos(left.agent), MultiPandaArmPlanner.OPEN])
    home_r = np.hstack([right._arm_qpos(right.agent), MultiPandaArmPlanner.OPEN])

    # Phase 1: both arms move simultaneously; pad the shorter tape so both finish.
    La_co = max(len(lA), len(rA))
    lA_p, rA_p = _pad(lA, La_co), _pad(rA, La_co)

    left_full = [home_l] * SETTLE + lA_p
    right_full = [home_r] * SETTLE + rA_p
    phase1_end = SETTLE + La_co
    max_robot_basket_force = 0.0

    def track_rollout_quality():
        nonlocal max_robot_basket_force
        nonlocal max_background_displacement, max_background_displacement_prop
        nonlocal max_background_speed, max_background_speed_prop
        max_robot_basket_force = max(
            max_robot_basket_force, _robot_basket_contact_force(base_env)
        )
        for actor in background_props:
            displacement = float(
                np.linalg.norm(_np_pos(actor) - background_initial[id(actor)])
            )
            if all(body.kinematic for body in actor._bodies):
                velocity = np.zeros(3, dtype=np.float64)
            else:
                velocity = actor.linear_velocity
                if hasattr(velocity, "cpu"):
                    velocity = velocity.cpu().numpy()
                velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)[0]
            speed = float(np.linalg.norm(velocity))
            if displacement > max_background_displacement:
                max_background_displacement = displacement
                max_background_displacement_prop = actor.name
            if speed > max_background_speed:
                max_background_speed = speed
                max_background_speed_prop = actor.name

    _coplay(env, left_full, right_full, frame_cb=track_rollout_quality)
    if debug:
        print(
            "  [pass2:phase1] items_in_basket="
            f"{int(base_env._items_in_basket()[0].sum().item())}/{len(base_env.items)}"
        )

    # Phase 2 (live): left arm frozen at its current pose, right collects the
    # relayed cube(s) from the middle. Sync gripper_states so the frozen left arm
    # keeps holding its (open) pose while the right arm is driven.
    gripper_states[0] = MultiPandaArmPlanner.OPEN
    gripper_states[1] = MultiPandaArmPlanner.OPEN
    right.frame_cb = track_rollout_quality
    right.start_recording()
    right_phase2_ends = _right_collect_middle(
        right, cfg, left_cubes, box_center, start_slot=len(right_cubes)
    )
    rB = right.stop_recording()
    phase2_end = phase1_end + len(rB)

    # Final settle so the success metric sees a static scene.
    last = _wait(right, steps=FINAL_HOLD)
    right.frame_cb = None
    T_total = phase2_end + FINAL_HOLD

    # Expose the full flat action tape (reconstructed from the recorded pass) so a
    # caller can screen the demo for clean (non-winding) wrist motion. During R2
    # the left arm is frozen at its phase-1-end pose (gripper open).
    left_frozen = np.hstack([left._arm_qpos(left.agent), MultiPandaArmPlanner.OPEN])
    phase1_actions = [np.hstack([l, r]) for l, r in zip(left_full, right_full)]
    r2_actions = [np.hstack([left_frozen, rb]) for rb in rB]
    hold_src = r2_actions[-1] if r2_actions else phase1_actions[-1]
    tape = phase1_actions + r2_actions + [hold_src] * FINAL_HOLD
    base_env._action_tape = np.asarray(tape, dtype=np.float64)
    base_env._max_robot_basket_contact_force = float(max_robot_basket_force)
    base_env._background_prop_count = len(background_props)
    base_env._max_background_prop_displacement = float(max_background_displacement)
    base_env._max_background_prop_speed = float(max_background_speed)
    base_env._max_background_prop_displacement_name = max_background_displacement_prop
    base_env._max_background_prop_speed_name = max_background_speed_prop

    if debug:
        print(
            "  [pass2:done] items_in_basket="
            f"{int(base_env._items_in_basket()[0].sum().item())}/{len(base_env.items)} "
            f"success={bool(last[-1]['success'].item())}"
        )

    # Per-arm, per-cube subgoal labels. Tape offsets are exact boundaries between
    # complete cube operations. The first label includes the initial settle, and
    # each arm's last label extends through synchronization/final holds (no idle).
    left_segments = _segments(
        left_cubes,
        left_phase1_ends,
        LABEL_LEFT,
        start=SETTLE,
        final_end=T_total,
    )
    left_segments[0]["start"] = 0
    right_segments = _segments(
        right_cubes,
        right_phase1_ends,
        LABEL_RIGHT,
        start=SETTLE,
        final_end=phase1_end,
    )
    right_segments[0]["start"] = 0
    right_segments.extend(
        _segments(
            left_cubes,
            right_phase2_ends,
            LABEL_RIGHT,
            start=phase1_end,
            final_end=T_total,
        )
    )
    base_env._subgoal_segments = {
        "phase1_end": int(phase1_end),
        "end": int(T_total),
        "right_cube_order": [_cube_color(cube) for cube in right_cubes + left_cubes],
        "left": left_segments,
        "right": right_segments,
    }
    return last


def wrist_grab_max_path(action_tape):
    """Max wrist-joint (joint-7) *winding*, in degrees, over every grasp-approach
    window of BOTH arms. Winding = (total angular path) - |net rotation|, i.e. the
    back-and-forth EXCESS beyond the direct turn.

    A clean demo turns the wrist monotonically to align with the cube (winding ~0),
    even if the net turn is large (e.g. unfolding from home, or aligning to a
    differently-oriented cube). RRT occasionally winds the wrist the long way on
    the approach -- that excess is exactly what makes the gripper look like it is
    "spinning", and is what this screens out. Action layout: left = [arm_qpos(7),
    grip], right = [arm_qpos(7), grip]; wrist joint is index 6 (left) / 14 (right);
    gripper command is index 7 / 15.
    """
    a = np.asarray(action_tape, dtype=np.float64)
    worst = 0.0
    for wrist_col, grip_col in ((6, 7), (14, 15)):
        g = np.sign(a[:, grip_col])
        events = [k for k in range(1, len(a)) if g[k] != g[k - 1]]
        prev = 0
        for k in events:
            if g[k] < 0:  # gripper closing = a grasp; the approach is [prev, k]
                seg = np.degrees(a[prev : k + 1, wrist_col])
                path = float(np.abs(np.diff(seg)).sum())
                net = abs(float(seg[-1] - seg[0]))
                worst = max(worst, path - net)
            prev = k
    return worst


def grasp_approach_max_joint_winding(action_tape, subgoal_segments):
    """Maximum single-joint backtracking during any color-specific cube approach.

    For each per-cube instruction segment, measure from the exact segment start
    through its first gripper-close event. For every arm joint, winding is total
    angular path minus absolute net rotation. A direct monotonic approach has
    winding near zero; a conspicuous RRT loop produces a large value.
    """
    a = np.asarray(action_tape, dtype=np.float64)
    worst = 0.0
    for side, joint_start, grip_col in (("left", 0, 7), ("right", 8, 15)):
        for segment in subgoal_segments.get(side, []):
            start = max(0, int(segment["start"]))
            end = min(len(a), int(segment["end"]))
            if end - start < 2:
                continue
            grip = np.sign(a[start:end, grip_col])
            close_offsets = [
                k
                for k in range(1, len(grip))
                if grip[k] < 0 and grip[k] != grip[k - 1]
            ]
            if not close_offsets:
                continue
            close = start + close_offsets[0]
            q = np.degrees(a[start : close + 1, joint_start : joint_start + 7])
            path = np.abs(np.diff(q, axis=0)).sum(axis=0)
            net = np.abs(q[-1] - q[0])
            worst = max(worst, float(np.max(path - net)))
    return worst


def cube_operation_max_wrist_net_rotation(action_tape, subgoal_segments):
    """Maximum absolute wrist-joint net rotation over a per-cube operation.

    The approach winding metric intentionally ignores a large monotonic turn. That
    is appropriate for small face alignment, but it allowed a near-full revolution
    while carrying a cube when RRT selected the opposite joint-7 branch. Bounding
    per-operation net rotation prevents that visually unnecessary long turn.
    """
    a = np.asarray(action_tape, dtype=np.float64)
    worst = 0.0
    for side, wrist_col in (("left", 6), ("right", 14)):
        for segment in subgoal_segments.get(side, []):
            start = max(0, int(segment["start"]))
            end = min(len(a), int(segment["end"]))
            if end - start < 2:
                continue
            wrist = np.degrees(a[start:end, wrist_col])
            worst = max(worst, abs(float(wrist[-1] - wrist[0])))
    return worst


def cube_operation_max_wrist_path(action_tape, subgoal_segments):
    """Maximum total joint-7 travel during one color-specific cube operation."""
    a = np.asarray(action_tape, dtype=np.float64)
    worst = 0.0
    for side, wrist_col in (("left", 6), ("right", 14)):
        for segment in subgoal_segments.get(side, []):
            start = max(0, int(segment["start"]))
            end = min(len(a), int(segment["end"]))
            if end - start < 2:
                continue
            wrist = np.unwrap(a[start:end, wrist_col])
            path = float(np.degrees(np.abs(np.diff(wrist)).sum()))
            worst = max(worst, path)
    return worst


def left_place_counterclockwise_wrist_path(action_tape):
    """Maximum positive joint-7 travel while LEFT carries a cube to its drop.

    Panda joint coordinates use the positive joint-7 direction as
    counter-clockwise.  Measure only from a left-gripper close event through the
    following open event, so normal pre-grasp alignment and the later retreat do
    not affect this placement-direction quality gate.  Summing positive
    increments (instead of checking only the net change) also catches a visually
    meaningless counter-clockwise detour that is later undone.
    """
    a = np.asarray(action_tape, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    grip = np.sign(a[:, 7])
    close_events = np.flatnonzero((grip[1:] < 0) & (grip[1:] != grip[:-1])) + 1
    open_events = np.flatnonzero((grip[1:] >= 0) & (grip[1:] != grip[:-1])) + 1
    worst = 0.0
    for close in close_events:
        later_opens = open_events[open_events > close]
        if not len(later_opens):
            continue
        release = int(later_opens[0])
        wrist = np.unwrap(a[int(close) : release + 1, 6])
        positive_path = np.maximum(np.diff(wrist), 0.0).sum()
        worst = max(worst, float(np.degrees(positive_path)))
    return worst


def _robot_basket_contact_force(base_env):
    """Return the largest robot/basket contact force in the current CPU step."""
    robot_entities = {
        link._bodies[0].entity
        for agent in base_env.agent.agents
        for link in agent.robot.links
    }
    basket_entities = {part._bodies[0].entity for part in base_env.basket_parts}
    worst_impulse = 0.0
    for contact in base_env.scene.get_contacts():
        a = contact.bodies[0].entity
        b = contact.bodies[1].entity
        if not (
            (a in robot_entities and b in basket_entities)
            or (b in robot_entities and a in basket_entities)
        ):
            continue
        impulse = np.sum([point.impulse for point in contact.points], axis=0)
        worst_impulse = max(worst_impulse, float(np.linalg.norm(impulse)))
    return worst_impulse / float(base_env.scene.timestep)


def _coplay(env, left_tape, right_tape, frame_cb=None, labels=None):
    n = max(len(left_tape), len(right_tape))
    last = None
    for i in range(n):
        la = left_tape[min(i, len(left_tape) - 1)]
        ra = right_tape[min(i, len(right_tape) - 1)]
        last = env.step(np.hstack([la, ra]))
        if frame_cb is not None:
            if labels is not None:
                phase, ll, rl = labels(i)
                frame_cb(i, n, phase, ll, rl)
            else:
                frame_cb()
    return last
