"""Keypoint motion-planning solver for ``TwoRobotHangBagReplicaCAD-v1``.

Collaborative bag hanging.  The LEFT arm can only reach the bag; the RIGHT arm can
only reach the hook -- so the two arms hand the bag over via a middle drop-off:

* Phase 1 (LEFT): top-down grasp the bag's rigid carry handle, lift, carry it to the
  MIDDLE of the table (reachable by both arms) and set it down standing, release, and
  retreat clear.
* Phase 2 (RIGHT): top-down grasp the same handle at the middle, lift only enough to
  clear the table, carry it to the hook, thread the handle loop onto the tilted peg
  (a straight slide ALONG the peg axis), release so the bag hangs, and retreat.

The arms mostly act sequentially, but at the handoff the RIGHT arm begins its approach
while the LEFT arm withdraws along a simple joint-space retreat tape. Every carry is
done as short, mostly-straight screw (Cartesian) moves so the paths are direct and the
bag stays upright.

Geometry (workspace_offset = (1.4, -1.2, 0.92); LEFT = agents[0], base world y=-1.88;
RIGHT = agents[1], base world y=-0.52):

* The handle is a thin arch whose top BAR runs along world x and is ~0.03 m thick in y.
  Both arms grip it TOP-DOWN with the fingers closing across y (straddling the bar).
* The bag never rotates (fixed yaw), so every move is a pure translation: a target bag
  pose maps to a tcp target by the same offset the tcp had at grasp time.
* To hang, the RIGHT arm keeps a centre grip and uses measured correction steps near
  the hook so the physical loop, not just the nominal wrist target, is centred before
  the gripper opens.
"""

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from robopoly.examples.motionplanning.two_robot.solutions.planner import (
    MultiPandaArmPlanner,
    make_grasp_pose as _make_grasp_pose,
    move_optional as _move_optional,
    move_or_fail as _move_or_fail,
    wait_steps as _wait,
)

LABEL_LEFT = "pick up bag and place in middle"
LABEL_RIGHT = "pick up bag and put on hook"

SETTLE = 12
MAX_FINAL_SETTLE = 180
FINAL_SUCCESS_STREAK = 10
CARRY_VEL = 0.32   # joint vel/acc fraction for the arms (slow -> the handle grip holds)
RIGHT_HOOK_STEP_REPEATS = 1
PRE_RELEASE_SETTLE = 30
RELEASE_OPEN_STEPS = 80
POST_RELEASE_SETTLE = 40

# --- tunables ---------------------------------------------------------------
GRIP_BELOW_TOP = 0.015     # grip this far below the handle bar's top
LIFT = 0.14                # straight-up lift after a grasp
# Bag's bar-centre target (rel workspace) at the drop-off. The handoff sits ~0.68 m
# from each arm base (near both reach limits), so it is kept only modestly forward and
# roughly centred in y so BOTH arms can reach it with a low vertical approach.
MIDDLE_REL = (0.09, 0.0)
# The right arm grips the bar at CENTRE so the bag hangs from the same centered
# handle point after release. Keep the pickup lift low so the bag does not collide
# with the hook or become a big pendulum before threading.
RIGHT_GRIP_X_OFF = 0.0
RIGHT_GRIP_BELOW_TOP = GRIP_BELOW_TOP
RIGHT_LIFT = 0.035
# Approximate centre of the handle loop's open hole in the handle-link local frame.
# Local axes from the asset: x runs along the arch, y is vertical, z is the thin
# through-axis. Putting this point on the hook peg means the peg is inside the loop
# before the gripper opens.
LOOP_OPEN_LOCAL = np.array([-0.15, 0.105, 0.0])
LOOP_ABOVE_PEG = 0.01
HANDLE_ABOVE_PEG = 0.035
THREAD_DIST = 0.10         # straight slide along the peg to thread the loop on
THREAD_FINAL_DIST = 0.0
THREAD_BAR_ABOVE_PEG = 0.095
THREAD_STAGE_BAR_ABOVE_PEG = 0.13
HOOK_Y_BACKOFF = 0.055
# Once threaded, lower in place. A positive x bias made the arm visibly pull the
# loop back outward before setting it down.
RELEASE_BAR_X_BIAS = 0.0
RELEASE_LOOP_Y_BIAS = HOOK_Y_BACKOFF
SEAT_BAR_ABOVE_PEG = 0.075
RELEASE_TILT_PITCH_DEG = 0.0
LEFT_RETREAT_TAPE_STEPS = 90
RIGHT_APPROACH_DELAY_SECONDS = 0.5
# Height above a target the arm first moves to before descending. Kept small: the bag
# is tall, so at the far handoff a big hover clearance pushes the tcp past the arm's
# reach.
APPROACH_UP = 0.06
HANG_STAGE_UP = 0.035      # staging height above the peg before the thread-in


def _try_topdown_grasp(planner, center, closings):
    for closing in closings:
        pose = _make_grasp_pose(
            planner.agent,
            approaching=np.array([0.0, 0.0, -1.0]),
            closing=np.asarray(closing, dtype=np.float64),
            center=center,
        )
        if planner.move_to_pose_with_screw(pose, dry_run=True) != -1:
            return pose
    return _make_grasp_pose(
        planner.agent,
        approaching=np.array([0.0, 0.0, -1.0]),
        closing=np.asarray(closings[0], dtype=np.float64),
        center=center,
    )


def _handle_bar_top_center(base_env):
    """World (x, y, z) of the centre of the top of the bag's handle, from the handle
    link's live collision geometry (independent of the link's pivot origin).

    The environment now uses the original bag handle directly. Handles both
    convex-mesh and box collision shapes."""
    import sapien.physx as physx

    link = base_env.bag_handle_link._objs[0]
    Tw = link.entity.pose.to_transformation_matrix()
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for cs in link.get_collision_shapes():
        T = Tw @ cs.get_local_pose().to_transformation_matrix()
        if isinstance(cs, physx.PhysxCollisionShapeBox):
            hs = np.asarray(cs.half_size)
            pts = np.array(
                [[sx * hs[0], sy * hs[1], sz * hs[2]]
                 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            )
        else:
            pts = np.asarray(cs.vertices) * np.asarray(cs.scale)
        w = (T[:3, :3] @ pts.T).T + T[:3, 3]
        mins = np.minimum(mins, w.min(0))
        maxs = np.maximum(maxs, w.max(0))
    return np.array([(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, maxs[2]])


def _handle_origin(base_env):
    """World position of the handle link origin, which is what ``evaluate`` checks."""
    return base_env.bag_handle_link.pose.p[0].cpu().numpy()


def _handle_loop_opening(base_env):
    """World position of the approximate centre of the handle loop opening."""
    ent = base_env.bag_handle_link._objs[0].entity
    T = ent.pose.to_transformation_matrix()
    p = np.asarray(LOOP_OPEN_LOCAL, dtype=np.float64)
    return T[:3, :3] @ p + T[:3, 3]


def _handle_on_hook(base_env):
    """Conservative pre-release check: the hook is actually through the handle loop
    before the gripper opens, while the handle origin also remains in the task's
    accepted hang band."""
    peg = base_env._peg_tip_world()[0].cpu().numpy()
    handle = _handle_origin(base_env)
    loop = _handle_loop_opening(base_env)
    body = base_env.bag_body_link.pose.p[0].cpu().numpy()
    dz = handle[2] - peg[2]
    return (
        np.linalg.norm(handle[:2] - peg[:2]) < float(base_env.hang_xy_radius) + 0.025
        and np.linalg.norm(loop[:2] - peg[:2]) < 0.13
        and abs(loop[2] - peg[2]) < 0.18
        and -float(base_env.hang_handle_z_below) < dz < float(base_env.hang_handle_z_above)
        and body[2] < handle[2]
    )


def _through_axis(base_env):
    """World unit vector of the handle loop's through-axis (= peg thread axis).

    Set by the bag's fixed yaw."""
    yaw = float(base_env.bag_yaw)
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


def _topdown_bar_grasp(planner, base_env, center):
    """Top-down grasp of the original handle bar at ``center``.

    Pinch across the handle loop's through-axis first; fall back to world-y variants
    if that wrist orientation is unreachable.
    """
    thru = _through_axis(base_env)
    return _try_topdown_grasp(
        planner,
        center,
        (
            thru,
            -thru,
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ),
    )


def _fixed_wrist_bar_grasp(planner, base_env, center):
    """Use the arm's current wrist orientation for a no-spin approach to the handle."""
    return sapien.Pose(center, planner.agent.tcp.pose.sp.q)


def _grasp_handle(
    planner,
    base_env,
    x_off,
    z_below_top=GRIP_BELOW_TOP,
    close_steps=45,
    grasp_fn=None,
    above_use_rrt=True,
    joint_pregrasp=False,
):
    """Top-down grasp of the handle bar (optionally offset +x). Returns
    (grasp_pose, grasp_center) so the caller can compute translation targets."""
    bar = _handle_bar_top_center(base_env)
    center = bar + np.array([x_off, 0.0, -z_below_top])
    if grasp_fn is None:
        grasp_fn = _topdown_bar_grasp
    grasp = grasp_fn(planner, base_env, center)
    above = sapien.Pose(center + np.array([0, 0, APPROACH_UP]), grasp.q)
    planner.open_gripper(steps=10)
    if joint_pregrasp:
        _move_joint_pregrasp(planner, above, label="grasp above")
    else:
        _move_or_fail(planner, above, use_rrt=above_use_rrt, label="grasp above")
    _move_or_fail(planner, grasp, label="grasp descend", refine_steps=6)
    planner.close_gripper(steps=close_steps)
    return grasp, center


def _move_joint_pregrasp(planner, pose, label="joint pregrasp"):
    """Go to a pregrasp with a monotonic joint interpolation.

    The pose target still comes from MPLib IK, but bypassing the full Cartesian
    trajectory removes visible redundant-wrist swivels on the right-arm pickup.
    """
    print(f"Planning {label}: p={np.array2string(np.asarray(pose.p), precision=4)}")
    result = planner.move_to_pose_with_screw(pose, dry_run=True)
    if result == -1:
        result = planner.move_to_pose(pose, dry_run=True)
    if result == -1:
        raise RuntimeError(f"motion planning failed for {label}")
    start = planner._arm_qpos(planner.agent)
    target = np.asarray(result["position"][-1], dtype=np.float64)
    dist = float(np.max(np.abs(target - start)))
    steps = max(45, int(np.ceil(dist / 0.012)))
    for i in range(1, steps + 1):
        alpha = i / steps
        # Smoothstep gives a gentle start/stop without changing the target posture.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        qpos = start * (1.0 - alpha) + target * alpha
        planner.env.step(planner._flat_action(qpos))
        planner._record(qpos)
        planner.elapsed_steps += 1
        if planner.vis:
            planner.base_env.render_human()
        if planner.frame_cb is not None:
            planner.frame_cb()


def _lift(planner, grasp, dz=LIFT):
    cur = planner.agent.tcp.pose.sp
    _move_or_fail(planner, sapien.Pose(cur.p + np.array([0, 0, dz]), grasp.q),
                  use_rrt=False, label="lift")


def _gentle_screw(planner, target, q, label, segments=4):
    """Move in short Cartesian screw segments without RRT perturbations.

    RRT is useful for free-space routing, but while pinching the bag handle it can
    shake the object out of the gripper.  These short straight segments preserve the
    live grasp or fail loudly instead of silently choosing a rough detour.
    """
    start = np.asarray(planner.agent.tcp.pose.sp.p, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for i in range(1, segments + 1):
        p = start + (target - start) * (i / segments)
        pose = sapien.Pose(p, q)
        print(f"Planning {label} {i}/{segments}: p={np.array2string(p, precision=4)}")
        if planner.move_to_pose_with_screw(pose) == -1:
            raise RuntimeError(f"motion planning failed for {label}")


def _left_place_middle(planner, base_env, grasp, grasp_center):
    """LEFT: carry the held bag to the middle and set it down STANDING, then release.

    The bag hangs a fixed offset below the tcp; the grasp put the bar top at its
    standing height (bag bottom on the table), so returning the bar to that same z at
    the middle sets the bag down standing again -- a pure horizontal carry + release."""
    mid = np.array(base_env.workspace_offset, dtype=np.float64) + np.array(
        [MIDDLE_REL[0], MIDDLE_REL[1], 0.0]
    )
    # tcp target = grasp offset (tcp-relative-to-bar) applied at the middle bar centre.
    # grasp_center already = bar_top + (x_off, 0, -GRIP_BELOW_TOP); the bar's standing
    # top z is grasp_center.z + GRIP_BELOW_TOP, so keep grasp_center.z to stand it.
    place = np.array([mid[0], mid[1], grasp_center[2]])
    hover = place + np.array([0, 0, APPROACH_UP])
    _move_or_fail(planner, sapien.Pose(hover, grasp.q), use_rrt=True, label="mid hover")
    _move_or_fail(planner, sapien.Pose(place, grasp.q), use_rrt=False,
                  label="mid place", refine_steps=8)
    _wait(planner, steps=16)
    # Measure-and-correct: the long lateral carry lets the bag roll/slide in the thin
    # grip, so it lands off-target. Nudge the tcp to bring the bag's actual bar back to
    # the target xy (clamped to the arm's reach), so it ends where the right arm can
    # reach it. One or two small corrections converge.
    target_xy = mid[:2]
    for _ in range(2):
        bar = _handle_bar_top_center(base_env)
        err = target_xy - bar[:2]
        if np.linalg.norm(err) < 0.02:
            break
        cur = np.asarray(planner.agent.tcp.pose.sp.p, dtype=np.float64)
        new_tcp = _clamp_reach(planner, cur + np.array([err[0], err[1], 0.0]))
        _move_optional(planner, sapien.Pose(new_tcp, grasp.q), use_rrt=False,
                       label="mid correct", refine_steps=6)
        _wait(planner, steps=14)
    _wait(planner, steps=8)
    planner.open_gripper(steps=18)
    _wait(planner, steps=10)
    return place


def _clamp_reach(planner, tcp_xyz, max_reach=0.78):
    """Clamp a target tcp so it stays within ``max_reach`` (m) horizontally of the arm
    base -- keeps a correction move from commanding an unreachable pose."""
    base_xy = np.asarray(planner.robot.pose.sp.p[:2], dtype=np.float64)
    d = np.asarray(tcp_xyz[:2], dtype=np.float64) - base_xy
    r = np.linalg.norm(d)
    if r > max_reach:
        d = d / r * max_reach
    return np.array([base_xy[0] + d[0], base_xy[1] + d[1], tcp_xyz[2]])


def _left_retreat(planner, grasp_q):
    """LEFT: lift straight up off the bag, then pull back toward the -y home side so
    the arm is clear of the middle before the right arm moves in."""
    cur = planner.agent.tcp.pose.sp
    # Stay at reachable heights: a tall vertical lift from the middle handoff pose is
    # near the left Panda's envelope and can fail IK, so clear the bag with a small
    # lift and then park over the left-side workspace.
    up = sapien.Pose(cur.p + np.array([0.0, 0.0, 0.035]), grasp_q)
    _move_optional(planner, up, use_rrt=True, label="left retreat up")
    park = np.array(planner.base_env.workspace_offset, dtype=np.float64) + np.array(
        [0.0, -0.46, 0.38]
    )
    park_pose = _topdown_bar_grasp(planner, planner.base_env, park)
    _move_optional(planner, park_pose, use_rrt=True, label="left retreat park")


def _left_retreat_tape(planner):
    """Joint-space tape that moves the LEFT arm away while the RIGHT arm starts
    reaching for the handoff. This avoids a separate visible wait for left retreat."""
    start = planner.robot.get_qpos()[0].cpu().numpy()
    home = np.asarray(
        list(planner.base_env.home_arm_qpos_left) + [0.04, 0.04],
        dtype=np.float64,
    )
    tape = []
    for i in range(1, LEFT_RETREAT_TAPE_STEPS + 1):
        alpha = i / LEFT_RETREAT_TAPE_STEPS
        q = start * (1 - alpha) + home * alpha
        tape.append(np.hstack([q[: planner.arm_dof], MultiPandaArmPlanner.OPEN]))
    return tape


def _right_hang(planner, base_env, grasp, grasp_center, debug=False):
    """RIGHT: carry the held bag to the hook and thread the handle loop onto the peg.

    A pinch on the thin handle bar is NOT a rigid grasp -- the bag settles a little
    lower and off-centre and can swing -- so rather than assume the grip transform we
    (1) let the lifted bag settle and MEASURE where its handle bar actually sits
    relative to the tcp, then place the bar with that measured offset, and (2) move
    slowly with settles so the bag stays hanging quietly (no swing) through the thread.

    The loop is brought just beyond the peg tip, slid straight ALONG the peg axis toward
    the post to thread it on with the bar held a little above the peg, then released so
    the bag drops the last bit and the bar catches on the peg."""
    _wait(planner, steps=20)   # let the lifted bag settle to a quiet hang
    if debug:
        print(f"    [R after settle] handle={np.round(_handle_origin(base_env),3)} "
              f"bar={np.round(_handle_bar_top_center(base_env),3)} "
              f"tcp={np.round(planner.agent.tcp.pose.sp.p,3)}")
    # Peg geometry (world). Prefer the environment's tilted-peg helper so the solver
    # tracks task-specific hook edits instead of assuming a flat horizontal peg.
    axis_fn = getattr(base_env, "_peg_axis_world", None)
    axis = np.asarray(axis_fn() if axis_fn is not None else base_env.hook_peg_axis,
                      dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    peg = base_env._peg_tip_world()[0].cpu().numpy()

    def _handle_target(dx):
        # Success is checked on the live handle-link origin.  Keep that origin inside
        # the hook band while sliding from beyond the peg tip back onto the peg.
        return np.array(
            [peg[0] + axis[0] * dx, peg[1] + HOOK_Y_BACKOFF + axis[1] * dx,
             peg[2] + axis[2] * dx + HANDLE_ABOVE_PEG],
            dtype=np.float64,
        )

    def _bar_target(dx, z_above, y_bias=HOOK_Y_BACKOFF):
        # Use the physical top bar for gross hook approach. The handle-link origin is
        # much lower than the bar, so origin-based targets make the bag fly too high.
        return np.array(
            [peg[0] + axis[0] * dx, peg[1] + y_bias + axis[1] * dx,
             peg[2] + axis[2] * dx + z_above],
            dtype=np.float64,
        )

    tcp0 = np.asarray(planner.agent.tcp.pose.sp.p, dtype=np.float64)
    bar_rel = _handle_bar_top_center(base_env) - tcp0
    stage = _bar_target(THREAD_DIST, THREAD_STAGE_BAR_ABOVE_PEG) - bar_rel

    def _dbg(tag):
        if debug:
            print(f"    [{tag}] tcp={np.round(planner.agent.tcp.pose.sp.p,3)} "
                  f"handle={np.round(_handle_origin(base_env),3)} "
                  f"loop={np.round(_handle_loop_opening(base_env),3)} "
                  f"bar={np.round(_handle_bar_top_center(base_env),3)}")
    _move_or_fail(planner, sapien.Pose(stage, grasp.q), use_rrt=False,
                  label="hang stage", refine_steps=16)
    _wait(planner, steps=12); _dbg("stage")
    # Add the hook-post obstacle only after the gentle carry.  Adding it earlier makes
    # MPLib perturb the already-grasping start state, which shakes the bag out.
    hp = base_env.hook.pose.p[0].cpu().numpy()
    ph = float(base_env.hook_post_half[2])
    planner.add_box_collision(
        extents=[0.055, 0.055, 2 * ph + 0.08],
        pose=sapien.Pose([float(hp[0]), float(hp[1]), float(hp[2]) + ph + 0.05]),
    )
    # Re-measure after the staging carry, then thread while still holding the bag:
    # approach from beyond the peg tip (+axis), slide back opposite the peg direction
    # until the upper bar is just above the peg, then release.  This keeps the hook
    # through the handle before the gripper opens, without asking for an unreachable
    # wrist height from the low loop-centre keypoint.
    # With the bag pinched by the original handle, the useful invariant is the live handle
    # origin's offset from the wrist.  Command the handle from just beyond the peg tip
    # to the peg check point while it is still inside the hook's accepted z band.
    rel = _handle_origin(base_env) - np.asarray(planner.agent.tcp.pose.sp.p, dtype=np.float64)
    bar_rel = _handle_bar_top_center(base_env) - np.asarray(
        planner.agent.tcp.pose.sp.p, dtype=np.float64
    )
    pre_thread_handle = _handle_target(THREAD_DIST)
    threaded_handle = _handle_target(THREAD_FINAL_DIST)
    pre_thread_bar = _bar_target(THREAD_DIST, THREAD_BAR_ABOVE_PEG)
    # Finish the thread-through already on the final catch line.  That way once the
    # hook is through the handle, the next motion is a vertical-ish seat onto the hook
    # instead of a visible sideways shift.
    threaded_bar = _bar_target(
        THREAD_FINAL_DIST, THREAD_BAR_ABOVE_PEG, y_bias=RELEASE_LOOP_Y_BIAS
    )
    pre_thread = pre_thread_bar - bar_rel
    threaded = threaded_bar - bar_rel
    _move_or_fail(planner, sapien.Pose(pre_thread, grasp.q), use_rrt=True,
                  label="hang pre-thread", refine_steps=6)
    _wait(planner, steps=12); _dbg("pre-thread")
    _move_or_fail(planner, sapien.Pose(threaded, grasp.q), use_rrt=False,
                  label="hang thread-through", refine_steps=16)
    _wait(planner, steps=20); _dbg("threaded")
    # Once the hook is through the loop, go directly to the final seated pose instead
    # of inching the bag rightward through repeated correction moves.
    seat_bar = peg + np.array(
        [RELEASE_BAR_X_BIAS, RELEASE_LOOP_Y_BIAS, SEAT_BAR_ABOVE_PEG]
    )
    bar_rel = _handle_bar_top_center(base_env) - np.asarray(
        planner.agent.tcp.pose.sp.p, dtype=np.float64
    )
    release_q = (
        sapien.Pose(q=grasp.q)
        * sapien.Pose(q=euler2quat(0.0, np.deg2rad(RELEASE_TILT_PITCH_DEG), 0.0))
    ).q
    _move_or_fail(
        planner,
        sapien.Pose(seat_bar - bar_rel, release_q),
        use_rrt=False,
        label="hang seat on hook",
        refine_steps=14,
    )
    # Let the hook/contact carry the handle before releasing.  A short pause plus a
    # slow gripper opening damps the bag's residual swing without adding a long static
    # tail to the visualization.
    _wait(planner, steps=PRE_RELEASE_SETTLE); _dbg("seat")
    planner.clear_collisions()
    if debug:
        handle_now = _handle_origin(base_env)
        loop_now = _handle_loop_opening(base_env)
        bar_now = _handle_bar_top_center(base_env)
        print(f"  [hang] rel={np.round(rel,3)} handle={np.round(handle_now,3)} "
              f"loop={np.round(loop_now,3)} bar_top={np.round(bar_now,3)} "
              f"handle_target={np.round(threaded_handle,3)} "
              f"tcp={np.round(planner.agent.tcp.pose.sp.p,3)}")
    if not _handle_on_hook(base_env):
        raise RuntimeError("right gripper refused to release before handle is on hook")
    planner.open_gripper(steps=RELEASE_OPEN_STEPS)
    _wait(planner, steps=POST_RELEASE_SETTLE)
    # retreat: lift straight up off the handle, then pull back toward the arm's base
    # (horizontally away from the hung bag) so the gripper clears cleanly.
    base_xy = np.asarray(planner.robot.pose.sp.p[:2], dtype=np.float64)
    cur = planner.agent.tcp.pose.sp
    _move_optional(planner, sapien.Pose(cur.p + np.array([0, 0, 0.16]), release_q),
                   use_rrt=False, label="hang retreat up")
    cur = planner.agent.tcp.pose.sp
    back = (base_xy - cur.p[:2])
    back = back / (np.linalg.norm(back) + 1e-9) * 0.22
    _move_optional(planner, sapien.Pose(cur.p + np.array([back[0], back[1], 0.04]), release_q),
                   use_rrt=True, label="hang retreat back")


def _find_record_wrapper(env):
    e = env
    while e is not None:
        if e.__class__.__name__ == "RecordEpisode":
            return e
        e = getattr(e, "env", None)
    return None


def solve(env, seed=None, debug=False, vis=False, frame_cb=None, **kwargs):
    """Solve one hang-bag episode (sequential left place-in-middle, right hang).

    ``frame_cb`` (optional, no-arg) is invoked after every stepped frame so a caller
    can stream a video of the run. Returns the final ``env.step`` tuple, or ``-1`` on
    a motion-planning failure."""
    base_env = env.unwrapped
    rec = _find_record_wrapper(env)
    try:
        env.reset(seed=seed)
        gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
        # Move slowly: the bag is gripped by a thin handle and a fast carry lets its
        # inertia slide/swing it out of the grasp (esp. the long lateral handoff).
        left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis,
                                    joint_vel_limits=CARRY_VEL, joint_acc_limits=CARRY_VEL)
        right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis,
                                     joint_vel_limits=CARRY_VEL, joint_acc_limits=CARRY_VEL)
        left.frame_cb = right.frame_cb = frame_cb

        _wait(left, steps=SETTLE)

        # ---- Phase 1: LEFT picks the bag and places it in the middle ----
        grasp, center = _grasp_handle(left, base_env, x_off=0.0)
        _lift(left, grasp)
        if debug: print(f"    [L after lift] bar={np.round(_handle_bar_top_center(base_env),3)}")
        _left_place_middle(left, base_env, grasp, center)
        if debug: print(f"    [L after place] bar={np.round(_handle_bar_top_center(base_env),3)} "
                        f"tcp={np.round(left.agent.tcp.pose.sp.p,3)}")
        left_retreat = _left_retreat_tape(left)

        # Give the bag half a second to settle and let the LEFT arm begin clearing
        # the handoff before the RIGHT arm starts its approach.
        delay_steps = max(1, int(round(
            RIGHT_APPROACH_DELAY_SECONDS * base_env.control_freq
        )))
        delay_steps = min(delay_steps, len(left_retreat))
        for q in left_retreat[:delay_steps]:
            last = env.step(left._flat_action(q))
            if frame_cb is not None:
                frame_cb()
        phase1_end = int(base_env.elapsed_steps[0].item())

        # ---- Phase 2: RIGHT picks the bag from the middle and hangs it ----
        right.partner_tape = left_retreat[delay_steps:]
        right.partner_idx = 0
        right._partner_cursor = 0
        grasp_r, center_r = _grasp_handle(
            right,
            base_env,
            x_off=RIGHT_GRIP_X_OFF,
            z_below_top=RIGHT_GRIP_BELOW_TOP,
            close_steps=55,
            grasp_fn=_fixed_wrist_bar_grasp,
            above_use_rrt=False,
            joint_pregrasp=True,
        )
        _lift(right, grasp_r, dz=RIGHT_LIFT)
        if debug: print(f"    [handoff overlap] left retreat tape "
                        f"{min(right._partner_cursor, len(left_retreat))}/{len(left_retreat)}")
        if debug: print(f"    [R after lift] handle={np.round(_handle_origin(base_env),3)} "
                        f"bar={np.round(_handle_bar_top_center(base_env),3)} "
                        f"tcp={np.round(right.agent.tcp.pose.sp.p,3)}")
        right.path_step_repeats = RIGHT_HOOK_STEP_REPEATS
        try:
            _right_hang(right, base_env, grasp_r, center_r, debug=debug)
        finally:
            right.path_step_repeats = 1
            right.partner_tape = None
            right.partner_idx = None

        # Stop as soon as the arms are away and the task has stayed successful for
        # several consecutive frames; avoid a long, meaningless frozen tail.
        success_streak = 0
        for _ in range(MAX_FINAL_SETTLE):
            last = _wait(right, steps=1)
            if bool(base_env.evaluate()["success"][0].item()):
                success_streak += 1
                if success_streak >= FINAL_SUCCESS_STREAK:
                    break
            else:
                success_streak = 0
        end = int(base_env.elapsed_steps[0].item())
        base_env._subgoal_segments = {
            "phase1_end": phase1_end,
            "end": end,
            "left": [
                {"label": LABEL_LEFT, "start": 0, "end": end},
            ],
            "right": [
                {"label": LABEL_RIGHT, "start": 0, "end": end},
            ],
        }
    except RuntimeError as exc:
        if debug:
            print(f"  motion planning failed: {exc}")
        if rec is not None:
            rec.save_trajectory = True
        env.reset(seed=seed)
        return -1

    if debug:
        ev = base_env.evaluate()
        print(f"  [done] success={bool(ev['success'][0].item())} "
              f"bag_hung={bool(ev['bag_hung'][0].item())}")
    return last
