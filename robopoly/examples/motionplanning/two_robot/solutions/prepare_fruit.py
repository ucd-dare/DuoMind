"""Keypoint motion-planning solver for ``TwoRobotPrepareFruitReplicaCAD-v1``.

Object layout (workspace_offset = (1.4, -1.2, 0.92); arms mounted at table
height; -y is the LEFT arm / agents[0], +y is the RIGHT arm / agents[1]):

* two BANANAS  ~ world y in [-1.65, -1.48] -> in front of the LEFT arm
* two ORANGES  ~ world y in [-0.95, -0.75] -> in front of the RIGHT arm
* two PLATES   ~ world y ~ -1.2 (between the arms), separated in x by >= 0.28:
    - ``plates[0]`` (RoboCasa ``plate_8``) is a warm wood/orange-toned plate
      -> the "BROWN plate"
    - ``plates[1]`` (RoboCasa ``plate_9``) is a pale bluish-white plate
      -> the "WHITE plate"

Goal: every plate ends up holding exactly one banana AND one orange. The LEFT
arm owns the bananas, the RIGHT arm owns the oranges. Each arm serves the fruit
CLOSEST to it first. The cross-serving pattern requested by the user:

* first  banana -> BROWN plate    |  first  orange -> WHITE  plate
* second banana -> WHITE  plate   |  second orange -> BROWN  plate   (vice versa)

so each plate receives its banana and its orange from the two arms in the two
phases. The sub-goals, scheduled as TWO SIMULTANEOUS phases (both arms move at
the same time):

* Phase 1 (L1 ∥ R1):
    - L1  left : "place the first banana on the brown plate"
    - R1  right: "place the first orange on the white plate"
* Phase 2 (L2 ∥ R2):
    - L2  left : "place the second banana on the white plate"
    - R2  right: "place the second orange on the brown plate"

Execution is LIVE and single-pass (NOT the record-then-replay co-play of
``food_serve``, whose tape replay diverges on these contact-rich grasps and makes
an arm re-grab an already-placed fruit). Within each phase the two arms PICK their
fruit at the SAME TIME (safe -- the banana and orange pick zones are far apart),
then PLACE one arm at a time. The central plate strip is narrow and mplib plans
one arm in isolation (it models neither the other arm nor already-placed fruit),
so sequential validated placement is much more reliable for this contact-rich
task than optimistic co-placement. Everything is planned against the real current
state, so each arm reliably proceeds to the correct REMAINING fruit; every grasp
retries on a drop, re-measuring the fruit.

Two environment tweaks make the task solvable (analogous to the cook-pot lid
change): the plates are built KINEMATIC (a placed fruit brushing the rim can't
shove the plate away) and their x-spread was tightened so both plates sit ~0.70 m
from the arms rather than at the reach limit. See the env file.
"""

from dataclasses import dataclass

import numpy as np
import sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

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

# Per-arm subgoal instruction strings (natural language, for sub-goal-conditioned
# training). Each arm carries a distinct instruction per phase.
LABEL_L1 = "put bread on brown plate"
LABEL_L2 = "put bread on white plate"
LABEL_R1 = "put orange on white plate"
LABEL_R2 = "put orange on brown plate"

# Steps spent letting the scene settle at the start of the recorded episode.
SETTLE = 12
# Require the completed arrangement to remain successful for consecutive frames;
# a one-frame success while an object is still sliding is not a valid demo.
FINAL_SUCCESS_STREAK = 10
MAX_FINAL_SETTLE = 120


@dataclass
class PrepareFruitConfig:
    # --- banana grasp ---
    # bananas settle into varied curved shapes (a thin ~0.03 m body, sometimes with
    # a curled tip that inflates the bounding box, or a flat C-curve). Grasping the
    # AABB centre misses the sparse tip and knocks the fruit away, so instead grasp
    # the DENSE BASE SLAB: take the ring of collision vertices within ``banana_slab``
    # of the resting bottom, close the jaws across that slab's minor (thickness)
    # axis, at its centroid, ``banana_grasp_above`` above the bottom. This grips the
    # thickest, most stable part and lifts reliably across both banana assets.
    banana_slab: float = 0.035
    banana_grasp_above: float = 0.018
    # grasp depths (above the resting bottom) tried across retry attempts. Different
    # banana shapes grip best at different depths, so cycling these turns the retry
    # loop into an effective search instead of repeating one deterministic grasp.
    # Hold the upper half of the face-up bread. A centre-height TCP lets a finger
    # corner touch the irregular braided rim during the last descent and kick the
    # bread before closure begins.
    banana_grasp_depths: tuple = (0.006,)
    # scale applied to the arm's vel/acc during the banana grasp descent + close, so
    # the jaws close gently around the thin fruit instead of batting it away / making
    # it skitter across the (low-friction, rigid) table.
    gentle_descent_scale: float = 0.2
    # --- orange grasp ---
    # oranges are ~round (~0.065-0.07 m) -- nearly the gripper's 0.08 m opening, so
    # the grasp is tight. Retries try grasp heights a touch above centre (a narrower
    # cross-section = more jaw clearance) as well as centre, searching for a hold.
    orange_grasp_z: float = 0.0
    # Grasp the narrower upper hemisphere; a centre-height descent lets a finger
    # corner hit the widest point first and shove the orange away.
    orange_grasp_heights: tuple = (0.015,)
    pregrasp: float = 0.08
    close_steps: int = 45
    bread_close_steps: int = 90
    # frames held still after the gripper closes, before lifting, so the grip fully
    # engages the fruit (otherwise it slips out as the lift accelerates).
    post_close_settle: int = 12
    # vel/acc scale for moves made WHILE HOLDING a fruit (lift, retract, carry). A
    # fast lift/carry accelerates a marginally-gripped fruit out of the jaws; moving
    # gently keeps it seated.
    carry_scale: float = 0.5
    # high transit altitude (above the workspace surface) that every horizontal move
    # is done at, so the carried gripper always clears the plates and any placed
    # fruit. Modest enough that the far plate stays inside the arm's reach.
    transit_z: float = 0.28
    # --- plate placement ---
    # banana and orange sit on OPPOSITE sides of the plate centre (offset along the
    # perpendicular to the banana's long axis) so the SECOND fruit placed on a plate
    # doesn't knock the FIRST off: their centres are 0.03 + 0.055 = 0.085 m apart,
    # clearing the banana half-width (~0.015) + orange radius (~0.033). Both offsets
    # keep the fruit on the flat area (banana end ~0.085, orange edge ~0.088, both
    # < the ~0.105 flat radius of the smaller plate) and inside 0.8*radius.
    banana_slot_offset: float = 0.03
    orange_slot_offset: float = 0.055
    # fruit-CENTRE release height above the plate surface. Must exceed the fruit's
    # half-height so the body rests ON the plate rather than being driven INTO it
    # (a low release interpenetrates the now-kinematic plate and the fruit pops off).
    # But it must NOT be so high that the fruit free-falls and bounces on the rigid
    # plate -- a bounce is chaotic and won't reproduce between the solo recording and
    # the co-play replay. So the fruit is set down JUST above the surface (banana
    # half-height ~0.04, orange radius ~0.033) and the descent itself is done slowly
    # (see ``place_descent_scale``) so it's a gentle placement, not a drop.
    banana_release_z: float = 0.010
    orange_release_z: float = 0.045
    # slow the final descent onto the plate so the fruit is set down softly (no
    # bounce). At the bottom the fruit's LOWEST point is left this far above the plate
    # surface -- essentially touching -- so opening the gripper is a set-down, not a
    # drop onto the rigid (kinematic) plate. The descent height is derived from the
    # HELD fruit's own geometry, so it works for a fat or a thin banana alike.
    place_descent_scale: float = 0.45
    place_contact_gap: float = 0.006
    # open the gripper SLOWLY on release: a thin banana pinched between the jaws
    # stores elastic energy and springs out ("bounces") if the jaws snap open.
    place_open_steps: int = 45
    # frames to hold still after opening the gripper, letting the fruit settle before
    # the arm retreats (longer for the banana, which rocks as it lands).
    banana_settle: int = 25
    orange_settle: int = 8
    # number of placement attempts after the initial pick. A missed release usually
    # leaves the fruit close to the plate, so re-picking and placing from the live
    # pose recovers without changing the environment.
    place_retries: int = 3
    # BOTH arms place at the same time. When their reaches cross the central strip the
    # RIGHT (orange) arm carries this much HIGHER so the two arms pass at different
    # altitudes instead of colliding, then each descends straight down to its plate.
    coplace_cross_lift: float = 0.16
    # gap the gripper hovers above the release point before descending, and how far
    # it retreats straight up afterwards. Kept modest so the TCP target stays inside
    # the arm's reach even for the far plate (a tall hover balloons out of reach).
    plate_hover_gap: float = 0.13
    open_steps: int = 16


# ---------------------------------------------------------------------------
# Fruit geometry (world-frame AABB from the live collision shapes)
# ---------------------------------------------------------------------------

def _fruit_vertices(actor):
    """World-frame collision-mesh vertices of a (merged) fruit actor.

    The merged actors block ``get_first_collision_mesh``; read the convex-mesh
    vertices off the single scene-0 PhysX shape instead.
    """
    ent = actor._objs[0]
    T_ent = ent.get_pose().to_transformation_matrix()
    pts = []
    for comp in ent.get_components():
        for cs in getattr(comp, "collision_shapes", []) or []:
            v = np.asarray(cs.get_vertices()) * np.asarray(cs.get_scale())
            T_local = cs.get_local_pose().to_transformation_matrix()
            v = (T_local[:3, :3] @ v.T).T + T_local[:3, 3]
            v = (T_ent[:3, :3] @ v.T).T + T_ent[:3, 3]
            pts.append(v)
    return np.vstack(pts)


def _banana_grasp_params(actor, cfg, grasp_above=None):
    """Return (grasp_centre_xyz, closing_axis, long_axis) for a banana.

    Computed from the DENSE BASE SLAB (collision vertices within ``banana_slab`` of
    the resting bottom): ``closing_axis`` is the horizontal perpendicular to the
    slab's long axis (the thickness the jaws pinch across); ``long_axis`` its major
    axis (used to offset the orange clear of the banana on the shared plate); the
    grasp centre sits at the middle cross-slice, ``grasp_above`` above the bottom
    (defaults to ``cfg.banana_grasp_above``; retries vary it to search depths).
    """
    if grasp_above is None:
        grasp_above = cfg.banana_grasp_above
    # Prepare Food now uses the same RoboTwin bread as Food Serve. Use its live
    # collision bounds because the bread can settle at a small roll angle; grasp the
    # actual body centre and close across its current horizontal minor axis.
    if str(getattr(actor, "name", "")).startswith("food_bread"):
        vertices = _fruit_vertices(actor)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        center[2] += grasp_above
        xy = vertices[:, :2] - vertices[:, :2].mean(axis=0)
        _, axes = np.linalg.eigh(xy.T @ xy)
        major = axes[:, 1]
        long_axis = np.array([major[0], major[1], 0.0])
        long_axis /= np.linalg.norm(long_axis)
        closing = np.array([-long_axis[1], long_axis[0], 0.0])
        return center, closing, long_axis
    v = _fruit_vertices(actor)
    bottom = float(v[:, 2].min())
    # Long axis from the FULL horizontal point cloud (more samples -> a far more
    # stable jaw alignment than the small base slab, whose PCA jitters for a thin
    # banana and lets a jaw corner clip the fruit).
    all_xy = v[:, :2]
    acxy = all_xy.mean(axis=0)
    _, avecs = np.linalg.eigh((all_xy - acxy).T @ (all_xy - acxy))
    major = avecs[:, 1]
    long_axis = np.array([major[0], major[1], 0.0])
    long_axis /= np.linalg.norm(long_axis)
    # Grasp the MIDDLE CROSS-SLICE of the base slab, not its centroid: for a strongly
    # C-curved banana the centroid can fall in the concave gap (off the body), so the
    # jaws would close on nothing. Take the thin band of base-slab vertices around the
    # median position along the long axis -- that slice is always on the banana.
    slab = v[v[:, 2] <= bottom + cfg.banana_slab]
    t = (slab[:, :2] - acxy) @ major
    band = np.abs(t - np.median(t)) <= 0.015
    scxy = slab[band][:, :2].mean(axis=0)
    # close the jaws across the banana's THICKNESS -- the horizontal perpendicular
    # to the (stable) long axis. Deriving it from long_axis is far more robust than
    # a PCA of the small cross-slice, whose minor axis is noisy for a thin banana.
    closing = np.array([-long_axis[1], long_axis[0], 0.0])
    closing /= np.linalg.norm(closing)
    center = np.array([scxy[0], scxy[1], bottom + grasp_above])
    return center, closing, long_axis


def _reachable_place_yaw(planner, place_center, hover_center):
    """Return top-down (place_pose, hover_pose) at a wrist yaw the arm can reach.

    A held fruit is carried/placed with the wrist yaw FREE (the fruit just rotates
    with the wrist), so search the yaws for one the arm can reach. The binding
    constraint is the lower PLACE pose (the hover above it is easier), so a yaw is
    accepted only if BOTH plate poses plan -- screw first across all yaws, then
    RRT. This makes the edge-of-envelope far-plate placements plan where the fixed
    grasp orientation would fail. Falls back to the straight (yaw-0) pair if
    nothing plans (the caller's ``move_or_fail`` then reports the failure).
    """
    def pair(yaw):
        q = euler2quat(0, 0, yaw)
        place = _make_grasp_pose(planner.agent, np.array([0, 0, -1]),
                                 _tcp_closing(planner.agent), place_center)
        place = place * sapien.Pose(q=q)
        hover = sapien.Pose(hover_center, place.q)
        return place, hover

    yaws = (0.0, np.pi, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4)
    for use_rrt in (False, True):
        for yaw in yaws:
            place, hover = pair(yaw)
            ok_h = (planner.move_to_pose(hover, dry_run=True) if use_rrt
                    else planner.move_to_pose_with_screw(hover, dry_run=True)) != -1
            ok_p = (planner.move_to_pose(place, dry_run=True) if use_rrt
                    else planner.move_to_pose_with_screw(place, dry_run=True)) != -1
            if ok_h and ok_p:
                return place, hover
    return pair(0.0)


def _common_pick_place_q(planner, pick_center, place_center, hover_center, pregrasp):
    """Choose one top-down yaw reachable at both the pick and destination.

    The returned orientation is used unchanged for the entire orange operation,
    eliminating wrist spins while avoiding an orientation that only one endpoint
    can reach.
    """
    pick_base = _make_grasp_pose(
        planner.agent, np.array([0, 0, -1]),
        _tcp_closing(planner.agent), pick_center
    )
    place_base = _make_grasp_pose(
        planner.agent, np.array([0, 0, -1]),
        _tcp_closing(planner.agent), place_center
    )
    yaws = (0.0, np.pi, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4)
    for use_rrt in (False, True):
        for yaw in yaws:
            delta = sapien.Pose(q=euler2quat(0, 0, yaw))
            pick = pick_base * delta
            q = pick.q
            poses = (
                sapien.Pose(np.asarray(pick_center) + [0, 0, pregrasp], q),
                pick,
                sapien.Pose(hover_center, (place_base * delta).q),
                place_base * delta,
            )
            plan = planner.move_to_pose if use_rrt else planner.move_to_pose_with_screw
            if all(plan(pose, dry_run=True) != -1 for pose in poses):
                return q
    raise RuntimeError("no single fixed wrist yaw reaches orange pick and plate")


def _top_grasp_pose(planner, center, closing=None, yaw_angles=None):
    if closing is None:
        closing = _tcp_closing(planner.agent)
    pose = _make_grasp_pose(
        planner.agent, approaching=np.array([0, 0, -1]), closing=closing, center=center
    )
    return _try_pose_variants(planner, pose, yaw_angles=yaw_angles)


def _fixed_top_grasp_pose(planner, center, closing=None, q=None):
    """Top-down grasp with one fixed wrist orientation.

    Prepare Food objects do not require an orientation search. Keeping the same
    pose removes gratuitous pre-grasp wrist spins and preserves the bread's roll
    and pitch so it can be set down face up.
    """
    if q is not None:
        return sapien.Pose(np.asarray(center), np.asarray(q))
    if closing is None:
        closing = _tcp_closing(planner.agent)
    return _make_grasp_pose(
        planner.agent,
        approaching=np.array([0, 0, -1]),
        closing=closing,
        center=np.asarray(center),
    )


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


# ---------------------------------------------------------------------------
# Per-arm pick/place primitives.
#
# Every horizontal move happens at a HIGH transit altitude and every approach /
# retreat is a straight vertical screw, so the arm only descends when it is
# already directly over its target. mplib plans one arm in isolation -- it does
# not model the plates, the placed fruit, or the other arm -- so a low RRT path
# would sweep the gripper straight through them and knock them off (or blow up the
# sim). Staying high between targets is what keeps that from happening.
# ---------------------------------------------------------------------------

def _transit_z(arm, cfg):
    """World z of the high transit altitude (clears the plates and any placed fruit)."""
    return float(arm.base_env.workspace_offset[2]) + cfg.transit_z


class _gentle:
    """Context manager: temporarily scale an arm's planner vel/acc down, so a
    contact-rich descent (e.g. onto a thin banana) approaches slowly and nudges
    the fruit into alignment instead of knocking it away."""

    def __init__(self, arm, scale):
        self.arm, self.scale = arm, scale

    def __enter__(self):
        p = self.arm.planner
        self._vel = np.array(p.joint_vel_limits, dtype=np.float64)
        self._acc = np.array(p.joint_acc_limits, dtype=np.float64)
        p.joint_vel_limits = self._vel * self.scale
        p.joint_acc_limits = self._acc * self.scale
        return self

    def __exit__(self, *exc):
        p = self.arm.planner
        p.joint_vel_limits = self._vel
        p.joint_acc_limits = self._acc


def _above(pose_or_xyz, z):
    """Sapien pose at the given (x, y) with height ``z`` (orientation from arg)."""
    p = pose_or_xyz.p if isinstance(pose_or_xyz, sapien.Pose) else np.asarray(pose_or_xyz)
    q = pose_or_xyz.q if isinstance(pose_or_xyz, sapien.Pose) else None
    return sapien.Pose([p[0], p[1], z], q)


def _pick_banana(arm, cfg, banana, attempt=0):
    """Top-grasp a banana across the thickness axis of its dense base slab: approach
    high, descend, RE-MEASURE, grasp, close, lift high. Returns the grasp pose.

    Re-measuring at the pregrasp (after settling) recomputes the slab grasp from
    the fruit's true pose. ``attempt`` selects the grasp depth so successive retries
    search different depths (a thin banana grips at a different height than a flat
    C-curved one).
    """
    tz = _transit_z(arm, cfg)
    depth = cfg.banana_grasp_depths[attempt % len(cfg.banana_grasp_depths)]
    center, closing, _ = _banana_grasp_params(banana, cfg, grasp_above=depth)
    yaw_angles = None if str(getattr(banana, "name", "")).startswith("food_bread") \
        else [0, np.pi]
    # only the 180-degree wrist flip preserves the jaw alignment to the thickness
    # axis (a free yaw sweep would rotate the jaws onto the long axis and miss).
    approx = _top_grasp_pose(arm, center, closing=closing, yaw_angles=yaw_angles)
    arm.open_gripper(steps=10)
    # Approach the pregrasp DIRECTLY (the fruit sits in the arm's own clear zone, so
    # no need to route up to the plate-clearing transit altitude -- and that high
    # waypoint is unreachable for a far fruit). The high transit is used only when
    # CARRYING over the plates.
    _move_or_fail(arm, sapien.Pose(approx.p + np.array([0, 0, cfg.pregrasp]), approx.q),
                  use_rrt=True, label="banana pregrasp")
    _wait(arm, steps=10)
    center, closing, _ = _banana_grasp_params(banana, cfg, grasp_above=depth)  # re-measure
    grasp = _top_grasp_pose(arm, center, closing=closing, yaw_angles=yaw_angles)
    # recentre directly above the re-measured grasp so the final descent is purely
    # vertical (a diagonal descent onto the thin banana clips and knocks it away).
    _move_or_fail(arm, sapien.Pose(grasp.p + np.array([0, 0, cfg.pregrasp]), grasp.q),
                  label="banana recentre")
    # descend and close SLOWLY so the jaws settle around the thin banana instead of
    # batting it aside on contact.
    with _gentle(arm, cfg.gentle_descent_scale):
        _move_or_fail(arm, grasp, label="banana grasp", refine_steps=18)
        arm.close_gripper(steps=cfg.close_steps)
    _move_or_fail(arm, _above(grasp, tz), label="banana lift")
    return grasp


def _pick_orange(arm, cfg, orange, attempt=0):
    """Top-grasp a round orange near its centre: approach high, descend, RE-MEASURE,
    grasp, close, lift high. Re-measuring corrects any nudge to the round fruit;
    ``attempt`` selects the grasp height (retries search for a firm hold)."""
    tz = _transit_z(arm, cfg)
    gz = cfg.orange_grasp_heights[attempt % len(cfg.orange_grasp_heights)]
    approx = _top_grasp_pose(arm, _np_pos(orange) + np.array([0, 0, gz]))
    arm.open_gripper(steps=10)
    # Approach the pregrasp DIRECTLY (fruit sits in the arm's own clear zone; the
    # plate-clearing transit altitude is unnecessary here and unreachable for a far
    # fruit -- it is used only when carrying over the plates).
    _move_or_fail(arm, sapien.Pose(approx.p + np.array([0, 0, cfg.pregrasp]), approx.q),
                  use_rrt=True, label="orange pregrasp")
    _wait(arm, steps=10)
    orange_pos = _np_pos(orange)  # re-measure after settle
    grasp = sapien.Pose(
        np.array([orange_pos[0], orange_pos[1], orange_pos[2] + gz]),
        approx.q,
    )
    # recentre directly above the re-measured centre so the final descent is purely
    # vertical (the orange nearly fills the jaws, so a diagonal descent clips it).
    _move_or_fail(arm, sapien.Pose(grasp.p + np.array([0, 0, cfg.pregrasp]), grasp.q),
                  label="orange recentre")
    _move_or_fail(arm, grasp, label="orange grasp", refine_steps=10)
    arm.close_gripper(steps=cfg.close_steps)
    _move_or_fail(arm, _above(grasp, tz), label="orange lift")
    return grasp


def _place_on_plate(arm, cfg, fruit, grasp_q, plate, slot_xy, release_z, settle=6):
    """Carry the grasped fruit (already lifted to transit height) over ``plate`` +
    ``slot_xy`` at altitude, descend straight down, release, lift back to transit.

    Placement picks one reachable top-down yaw, transforms the held food's TCP
    offset into that yaw, and follows one high target plus a vertical descent. The
    food's roll/pitch never changes, so bread remains face up."""
    del grasp_q
    tz = _transit_z(arm, cfg)
    plate_z = _np_pos(plate)[2]
    # Keep the exact grasp orientation for the whole carry. No wrist rotation is
    # introduced between picking and releasing the food.
    target_q = np.asarray(arm.agent.tcp.pose.sp.q, dtype=np.float64)
    fruit_offset_world = _np_pos(fruit) - arm.agent.tcp.pose.sp.p
    if np.linalg.norm(fruit_offset_world) > 0.12:
        raise RuntimeError("grasp lost before place")
    fruit_offset_local = quat2mat(target_q).T @ fruit_offset_world
    target_offset_world = quat2mat(target_q) @ fruit_offset_local
    # Set-down height from the HELD fruit's real geometry: lower it until its LOWEST
    # point is just above the plate surface (``place_contact_gap``), so releasing is a
    # gentle set-down rather than a drop onto the rigid plate. (Falls back to the
    # nominal ``release_z`` if the fruit's verts can't be read.)
    setdown_z = plate_z + release_z
    if not str(getattr(fruit, "name", "")).startswith("food_bread"):
        try:
            verts = _fruit_vertices(fruit)
            bottom_offset = float(_np_pos(fruit)[2] - verts[:, 2].min())
            setdown_z = plate_z + bottom_offset + cfg.place_contact_gap
        except Exception:
            pass
    place_tcp = np.array([slot_xy[0], slot_xy[1], setdown_z]) - target_offset_world
    hover_tcp = place_tcp + np.array([0, 0, cfg.plate_hover_gap])
    transit_tcp = hover_tcp.copy()
    transit_tcp[2] = tz
    with _gentle(arm, cfg.carry_scale):
        _move_or_fail(
            arm, sapien.Pose(transit_tcp, target_q), use_rrt=True,
            label="carry to plate"
        )
        _move_or_fail(arm, sapien.Pose(hover_tcp, target_q), label="plate hover")
    # descend SLOWLY onto the plate (fall back to full speed if the slowed-down path
    # trips mplib's time-parameterization)...
    try:
        with _gentle(arm, cfg.place_descent_scale):
            _move_or_fail(arm, sapien.Pose(place_tcp, target_q), label="plate place")
    except Exception:
        _move_or_fail(arm, sapien.Pose(place_tcp, target_q), label="plate place (full speed)")
    # ...and open the jaws SLOWLY so a pinched banana doesn't spring out.
    arm.open_gripper(steps=cfg.place_open_steps)
    _wait(arm, steps=settle)  # let the fruit settle onto the plate before retreating
    # lift straight back up to transit altitude so the gripper clears everything.
    _move_or_fail(arm, _above(sapien.Pose(place_tcp, target_q), tz), label="plate lift")
    _wait(arm, steps=6)


# ---------------------------------------------------------------------------
# Assignment: order each arm's fruit nearest-first; pick plate slots.
# ---------------------------------------------------------------------------

def _nearest_first(fruits, base_xy):
    """Return ``fruits`` ordered by increasing distance from ``base_xy``."""
    return sorted(
        fruits, key=lambda f: float(np.linalg.norm(_np_pos(f)[:2] - base_xy))
    )


def _plate_slots(base_env, cfg, banana_first, banana_second):
    """World-frame (banana_xy, orange_xy) placement slot for each plate.

    On every plate the banana is centred and the orange is offset perpendicular to
    THAT plate's banana's long axis so the two fruits sit side-by-side (never
    stack). The perpendicular has two signs; we pick the one leaning toward the
    RIGHT-arm base -- the orange is always placed by the right arm, so this keeps
    the (edge-of-envelope, far-plate) orange slot as reachable as possible while
    still clearing the banana. Returns ``{plate_idx: (banana_slot_xy, orange_slot_xy)}``.
    ``plates[0]`` (brown plate) receives ``banana_first``; ``plates[1]`` (white
    plate) receives ``banana_second``.
    """
    right_base_xy = np.asarray(base_env.right_agent.robot.pose.sp.p, dtype=np.float64)[:2]
    slots = {}
    for plate_idx, banana in ((0, banana_first), (1, banana_second)):
        plate_xy = _np_pos(base_env.plates[plate_idx])[:2]
        if str(getattr(banana, "name", "")).startswith("snack_square_bread"):
            # Identical square breads have no stable PCA major axis. Split their
            # slots along the line between the arms so each arm uses its own half.
            perp = right_base_xy - plate_xy
            perp /= np.linalg.norm(perp)
        else:
            _, _, long_axis = _banana_grasp_params(banana, cfg)
            perp = np.array([-long_axis[1], long_axis[0]])
            if np.dot(perp, right_base_xy - plate_xy) < 0:
                perp = -perp
        # banana and orange on opposite sides of centre so neither rolls off and
        # the two never collide when the second one is set down.
        banana_slot = plate_xy - perp * cfg.banana_slot_offset
        orange_slot = plate_xy + perp * cfg.orange_slot_offset
        slots[plate_idx] = (banana_slot, orange_slot)
    return slots


# ---------------------------------------------------------------------------
# Retreat: clear the acting arm out of the shared central plate zone.
# ---------------------------------------------------------------------------

def _retreat_to_side(arm, cfg, base_env):
    """Lift and pull the arm back over its OWN fruit zone, high and clear of the
    central plates, so the other arm has the plate region to itself.

    The two arms share the narrow central strip where the plates sit (y ~ -1.2),
    so they take turns there; between turns each arm parks over its own side
    (bananas at y < -1.2 for the LEFT arm, oranges at y > -1.2 for the RIGHT arm),
    well away from the plates and from each other.
    """
    base_y = float(np.asarray(arm.agent.robot.pose.sp.p, dtype=np.float64)[1])
    plate_y = float(base_env.workspace_offset[1])  # central plate strip (~-1.2)
    # park over the arm's OWN fruit zone: a step from the base TOWARD the plate
    # strip (never past it), so the arm sits between its base and the plates --
    # clear of the plates and of the other arm, but still on-table (a step the
    # other way would drive the arm off the table behind its base and blow up).
    toward_center = 0.30 if plate_y > base_y else -0.30
    own_y = base_y + toward_center
    center = np.array([base_env.workspace_offset[0], own_y, _transit_z(arm, cfg)])
    park = _reachable_topdown_single(arm, center)
    _move_optional(arm, park, use_rrt=True, label="retreat to side")
    _wait(arm, steps=4)


def _reachable_topdown_single(planner, center):
    """A top-down pose at ``center`` at the first reachable wrist yaw (screw then
    RRT), for parking moves where the exact yaw does not matter."""
    base = _make_grasp_pose(planner.agent, np.array([0, 0, -1]),
                            _tcp_closing(planner.agent), center)
    cands = [base * sapien.Pose(q=euler2quat(0, 0, y))
             for y in (0.0, np.pi, np.pi / 2, -np.pi / 2)]
    for cand in cands:
        if planner.move_to_pose_with_screw(cand, dry_run=True) != -1:
            return cand
    for cand in cands:
        if planner.move_to_pose(cand, dry_run=True) != -1:
            return cand
    return base


# ---------------------------------------------------------------------------
# Live-parallel solver: both arms pick AT ONCE, then place one at a time, planned
# LIVE in a SINGLE pass.
#
# Each arm's pick-and-place is a GENERATOR that yields one motion command at a
# time (move / gripper / wait) and does its own live re-measurement + retries
# BETWEEN yields. A driver advances the two generators in lockstep: each tick it
# plans BOTH arms' next command against the real current state and steps them
# together. Because it is a single live pass (never record-then-replay), the
# contact-rich grasps always act on the true fruit pose -- so each arm reliably
# proceeds to the correct REMAINING fruit and never re-grabs one already placed
# (the failure mode of tape co-play). The pick zones are far apart, while the
# plate strip is shared, so picks are co-executed and placements are validated
# sequentially.
# Order (honouring the assignment):
#   Phase 1 (L1 ∥ R1):  left  first  banana -> BROWN plate        (plates[0])
#                       right first  orange -> WHITE          plate (plates[1])
#   Phase 2 (L2 ∥ R2):  left  second banana -> WHITE          plate (plates[1])
#                       right second orange -> BROWN          plate (plates[0])
# ---------------------------------------------------------------------------

# Motion commands a pick-place generator yields. ``move`` reports back (via the
# generator's ``.send``) whether it planned+executed, so the generator can retry.
#   ('move', pose, use_rrt, scale)   scale=None for full speed, else a _gentle factor
#   ('grip', value, steps)           hold qpos, drive the gripper to ``value``
#   ('wait', steps)                  hold still


def _is_gripped(arm, fruit):
    return float(np.linalg.norm(_np_pos(fruit) - arm.agent.tcp.pose.sp.p)) < 0.11


def _is_on_plate(base_env, fruit, plate_idx):
    return bool(base_env._fruit_on_plate(
        fruit, base_env.plates[plate_idx], base_env.plate_radii[plate_idx])[0].item())


def _plan_dry(arm, pose, use_rrt, scale):
    """Plan (without executing) a trajectory for one arm to ``pose`` from its CURRENT
    qpos. Returns the list of arm-qpos waypoints, or ``None`` if planning fails.

    mplib's time-parameterization can occasionally raise (e.g. "Fail to parameterize
    path", more likely under the slowed-down ``scale``); on any such error, retry once
    at FULL speed, then give up (``None``) so the caller can retry rather than abort.
    """
    def _do():
        if use_rrt:
            return arm.move_to_pose(pose, dry_run=True)
        res = arm.move_to_pose_with_screw(pose, dry_run=True)
        return arm.move_to_pose(pose, dry_run=True) if res == -1 else res
    try:
        if scale:
            with _gentle(arm, scale):
                res = _do()
        else:
            res = _do()
    except Exception:
        try:
            res = _do()  # retry at full speed (parameterization is easier)
        except Exception:
            return None
    if res == -1:
        return None
    return [np.asarray(q, dtype=np.float64) for q in res["position"]]


def _park_gen(arm, cfg, base_env):
    """Yield the commands to lift and park the arm over its own fruit zone, clear of
    the plates and the other arm."""
    base_y = float(np.asarray(arm.agent.robot.pose.sp.p, dtype=np.float64)[1])
    plate_y = float(base_env.workspace_offset[1])
    own_y = base_y + (0.30 if plate_y > base_y else -0.30)
    center = np.array([base_env.workspace_offset[0], own_y, _transit_z(arm, cfg)])
    yield ('move', _reachable_topdown_single(arm, center), True, None)
    yield ('wait', 4)


def _retract_over_base_gen(arm, cfg, base_env):
    """Yield the command to pull the (fruit-holding) arm back over its OWN base, high
    and folded, so it is clear of the central plates while the OTHER arm reaches in to
    place -- otherwise the reaching arm can strike this one and blow up the sim."""
    base_y = float(np.asarray(arm.agent.robot.pose.sp.p, dtype=np.float64)[1])
    center = np.array([base_env.workspace_offset[0], base_y, _transit_z(arm, cfg)])
    # A Prepare Tea cup must keep its requested spawn yaw throughout the entire
    # carry. Preserve the grasp quaternion instead of selecting a new yaw here.
    if arm.agent_idx == 1 and hasattr(cfg, "mug_grasp_q"):
        retract = sapien.Pose(center, cfg.mug_grasp_q)
    else:
        retract = _reachable_topdown_single(arm, center)
    yield ('move', retract, True, cfg.carry_scale)


def _pick_banana_gen(arm, cfg, base_env, banana, tries=1):
    """Generator: LIVE pick of ``banana``, retrying (re-measuring each attempt) until
    the fruit is gripped and lifted to transit height. Raises if every attempt fails.
    Yields only PICK commands so it can be co-executed with the other arm's pick --
    the picks happen in the two arms' separate zones, so they never collide."""
    tz = _transit_z(arm, cfg)
    OPEN, CLOSED = MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.CLOSED
    for attempt in range(tries):
        depth = cfg.banana_grasp_depths[attempt % len(cfg.banana_grasp_depths)]
        is_square_bread = str(getattr(banana, "name", "")).startswith(
            "snack_square_bread"
        )
        if is_square_bread:
            vertices = _fruit_vertices(banana)
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            center[2] += cfg.square_bread_grasp_above
        else:
            center, _, _ = _banana_grasp_params(banana, cfg, grasp_above=depth)
        # Face-up bread is nearly round in the horizontal plane, making its PCA
        # jaw axis unstable. Use the arm's fixed closing axis for every bread.
        grasp = _fixed_top_grasp_pose(arm, center)
        yield ('grip', OPEN, 10)
        # One direct pregrasp followed by a vertical descent. The old re-measure /
        # recenter pair produced a visible, unnecessary lateral move in mid-air.
        if (yield ('move', sapien.Pose(grasp.p + np.array([0, 0, cfg.pregrasp]), grasp.q),
                   True, None)) is False:
            yield from _park_gen(arm, cfg, base_env); continue
        base_env._monitor_bread_grasp = True
        if (yield ('move', grasp, False, cfg.gentle_descent_scale)) is False:
            base_env._monitor_bread_grasp = False
            yield from _park_gen(arm, cfg, base_env); continue
        yield ('grip', CLOSED, cfg.bread_close_steps)
        yield ('wait', cfg.post_close_settle)  # let the grip engage before lifting
        base_env._monitor_bread_grasp = False
        yield ('move', _above(grasp, tz), False, cfg.carry_scale)  # lift high (gentle)
        if _is_gripped(arm, banana):
            return
        base_env._had_object_retry = True
        yield ('grip', OPEN, 6)
        yield from _park_gen(arm, cfg, base_env)
    raise RuntimeError("banana pick failed")


def _pick_orange_gen(arm, cfg, base_env, orange, tries=1):
    """Generator: LIVE pick of ``orange``, retrying until gripped and lifted."""
    tz = _transit_z(arm, cfg)
    OPEN, CLOSED = MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.CLOSED
    for attempt in range(tries):
        is_mug = str(getattr(orange, "name", "")).startswith("tea_mug")
        is_square_bread = str(getattr(orange, "name", "")).startswith(
            "snack_square_bread"
        )
        gz = (
            cfg.mug_grasp_z
            if is_mug
            else (
                cfg.square_bread_grasp_above
                if is_square_bread
                else cfg.orange_grasp_heights[attempt % len(cfg.orange_grasp_heights)]
            )
        )
        # Pick one reachable top-down orientation, execute it once, and retain it
        # unchanged through carry/release. Dry-run variants are never executed.
        if is_square_bread:
            vertices = _fruit_vertices(orange)
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            center[2] += gz
        else:
            center = _np_pos(orange) + np.array([0, 0, gz])
        # Use the fixed quaternion whose jaw axis is perpendicular to the handle.
        grasp = (
            _fixed_top_grasp_pose(arm, center, q=cfg.mug_grasp_q)
            if is_mug
            else (
                _fixed_top_grasp_pose(arm, center)
                if is_square_bread
                else _top_grasp_pose(arm, center)
            )
        )
        yield ('grip', OPEN, 10)
        if (yield ('move', sapien.Pose(grasp.p + np.array([0, 0, cfg.pregrasp]), grasp.q),
                   True, None)) is False:
            yield from _park_gen(arm, cfg, base_env); continue
        if (yield ('move', grasp, False, cfg.gentle_descent_scale)) is False:
            yield from _park_gen(arm, cfg, base_env); continue
        close_steps = (
            cfg.mug_close_steps
            if is_mug
            else (cfg.bread_close_steps if is_square_bread else cfg.close_steps)
        )
        yield ('grip', CLOSED, close_steps)
        yield ('wait', cfg.post_close_settle)  # let the grip engage before lifting
        yield ('move', _above(grasp, tz), False, cfg.carry_scale)  # lift high (gentle)
        if _is_gripped(arm, orange):
            return
        base_env._had_object_retry = True
        yield ('grip', OPEN, 6)
        yield from _park_gen(arm, cfg, base_env)
    raise RuntimeError("orange pick failed")


def _single_arm_place(arm, other, cfg, base_env, fruit, plate_idx, slot_xy, release_z, settle):
    """Place the fruit the arm is holding onto ``plate_idx`` while ``other`` stays
    frozen (holding its own fruit clear of the plates). Live, single-arm -- so the
    two arms never crowd the central plates at once and cannot collide."""
    plate = base_env.plates[plate_idx]
    _place_on_plate(arm, cfg, fruit, None, plate, slot_xy, release_z, settle=settle)
    _retreat_to_side(arm, cfg, base_env)
    return _is_on_plate(base_env, fruit, plate_idx)


def _validated_single_arm_place(
    arm,
    other,
    cfg,
    base_env,
    fruit,
    plate_idx,
    slot_xy,
    release_z,
    settle,
    repick,
    debug_label,
):
    """Place one held fruit, verify it landed on the requested plate, and retry.

    The first attempt assumes the fruit is already in the gripper from the
    simultaneous pick. If release misses or the grasp was lost, the same arm
    re-picks that live fruit pose and tries again.
    """
    del other
    last_error = None
    for attempt in range(cfg.place_retries):
        try:
            if attempt > 0 and not _is_gripped(arm, fruit):
                repick(arm, cfg, fruit, attempt=attempt)
            ok = _single_arm_place(
                arm, None, cfg, base_env, fruit, plate_idx, slot_xy, release_z, settle
            )
            if ok:
                return True
            last_error = RuntimeError(f"{debug_label} released off target plate")
        except RuntimeError as exc:
            last_error = exc
            _retreat_to_side(arm, cfg, base_env)
    raise RuntimeError(f"{debug_label} failed after retries: {last_error}")


def _place_gen(arm, cfg, base_env, fruit, plate_idx, slot_xy, release_z, settle,
               transit_offset=0.0):
    """Generator: LIVE place of the held ``fruit`` onto ``plate_idx`` -- carry high,
    hover, gentle low set-down (no bounce), slow release, lift, park. Yields commands
    so it can be co-executed with the OTHER arm's place for a SIMULTANEOUS placement.
    ``transit_offset`` raises this arm's high carry so two crossing arms pass at
    different altitudes. Returns True if released, False if a move failed / grasp lost."""
    plate = base_env.plates[plate_idx]
    tz = _transit_z(arm, cfg)
    plate_z = _np_pos(plate)[2]
    # Pick one reachable top-down placement yaw. This may turn once during the
    # high carry but never rolls/pitches the food, so bread remains face up.
    place_center = np.array([slot_xy[0], slot_xy[1], plate_z + release_z])
    hover_center = place_center + np.array([0, 0, cfg.plate_hover_gap])
    current_q = np.asarray(arm.agent.tcp.pose.sp.q, dtype=np.float64)
    is_mug = str(getattr(fruit, "name", "")).startswith("tea_mug")
    if is_mug:
        # Never yaw the held cup: preserve its requested spawn/grasp orientation.
        target_q = np.asarray(cfg.mug_grasp_q, dtype=np.float64)
        place = sapien.Pose(place_center, target_q)
    else:
        place, _ = _reachable_place_yaw(arm, place_center, hover_center)
        target_q = np.asarray(place.q, dtype=np.float64)
    fruit_offset_world = _np_pos(fruit) - arm.agent.tcp.pose.sp.p
    if np.linalg.norm(fruit_offset_world) > 0.12:
        return False  # grasp lost in transit
    fruit_offset_local = quat2mat(current_q).T @ fruit_offset_world
    target_offset_world = quat2mat(target_q) @ fruit_offset_local
    setdown_z = plate_z + release_z
    try:
        verts = _fruit_vertices(fruit)
        setdown_z = plate_z + float(_np_pos(fruit)[2] - verts[:, 2].min()) + cfg.place_contact_gap
    except Exception:
        pass
    place_tcp = np.array([slot_xy[0], slot_xy[1], setdown_z]) - target_offset_world
    hover_tcp = place_tcp + np.array([0, 0, cfg.plate_hover_gap])
    transit_tcp = hover_tcp.copy()
    transit_tcp[2] = tz + transit_offset
    # Exactly one high carry target, then a straight hover/place descent. This
    # avoids the former intermediate Cartesian target that made the arm wander in
    # the air after it was already above the destination.
    if (yield ('move', sapien.Pose(transit_tcp, target_q), True, cfg.carry_scale)) is False:
        return False
    if (yield ('move', sapien.Pose(hover_tcp, target_q), False, cfg.carry_scale)) is False:
        return False
    if (yield ('move', sapien.Pose(place_tcp, target_q), False, cfg.place_descent_scale)) is False:
        if (yield ('move', sapien.Pose(place_tcp, target_q), False, None)) is False:  # full-speed retry
            return False
    yield ('grip', MultiPandaArmPlanner.OPEN, cfg.place_open_steps)
    yield ('wait', settle)
    yield ('move', _above(sapien.Pose(place_tcp, target_q), tz), False, None)  # lift
    yield from _park_gen(arm, cfg, base_env)
    return True


def _prepare(arm, cmd, gripper_states):
    """Turn one generator command into (arm-qpos-trajectory, gripper-cmd, ok)."""
    idx = arm.agent_idx
    cur = arm._arm_qpos(arm.agent)
    if cmd[0] == 'move':
        _, pose, use_rrt, scale = cmd
        traj = _plan_dry(arm, pose, use_rrt, scale)
        if traj is None:
            return [cur], gripper_states[idx], False   # plan failed -> hold, retry
        if len(traj) == 0:
            return [cur], gripper_states[idx], True     # already at target
        return traj, gripper_states[idx], True
    if cmd[0] == 'grip':
        _, val, steps = cmd
        start = float(gripper_states[idx])
        gripper_states[idx] = val
        # Ramp the normalized finger target instead of jumping from fully open to
        # fully closed in one control frame. A step target makes one jaw contact
        # first and visibly kicks a round bread sideways before the other jaw
        # catches it.
        ramp = np.linspace(start, float(val), int(steps) + 1, dtype=np.float64)[1:]
        return [cur] * steps, ramp, True
    # 'wait'
    return [cur] * cmd[1], gripper_states[idx], True


# Optional no-arg hook invoked after every env.step of the co-executed motions, so a
# viz script can grab an overlaid frame per step (see the subgoal-overlay renderer).
# The planners' own ``frame_cb`` covers the single-arm placement path.
_frame_hook = None


def _costep(env, ltraj, lgrip, rtraj, rgrip):
    """Step the env driving BOTH arms along their (possibly different-length)
    trajectories at once; the shorter one holds its last qpos."""
    n = max(len(ltraj), len(rtraj))
    last = None
    lgrip_seq = np.asarray(lgrip).reshape(-1) if np.ndim(lgrip) else None
    rgrip_seq = np.asarray(rgrip).reshape(-1) if np.ndim(rgrip) else None
    for i in range(n):
        lq = ltraj[min(i, len(ltraj) - 1)]
        rq = rtraj[min(i, len(rtraj) - 1)]
        lg = lgrip_seq[min(i, len(lgrip_seq) - 1)] if lgrip_seq is not None else lgrip
        rg = rgrip_seq[min(i, len(rgrip_seq) - 1)] if rgrip_seq is not None else rgrip
        bread_before = None
        if getattr(env.unwrapped, "_monitor_bread_grasp", False):
            bread_before = [_np_pos(bread).copy() for bread in env.unwrapped.breads]
        last = env.step(np.hstack([lq, lg, rq, rg]))
        if bread_before is not None:
            max_step = max(
                np.linalg.norm(_np_pos(bread) - before)
                for bread, before in zip(env.unwrapped.breads, bread_before)
            )
            env.unwrapped._max_bread_grasp_step_displacement = max(
                env.unwrapped._max_bread_grasp_step_displacement,
                float(max_step),
            )
        if _frame_hook is not None:
            _frame_hook()
    return last


def _coexec(env, left, right, lgen, rgen, gripper_states):
    """Advance two pick-place generators in lockstep, stepping both arms together.
    Returns the last ``env.step`` result."""
    def _start(gen):
        try:
            return gen.send(None), False
        except StopIteration:
            return None, True
    lcmd, ldone = _start(lgen)
    rcmd, rdone = _start(rgen)
    last = None
    while not (ldone and rdone):
        if not ldone:
            ltraj, lgrip, lok = _prepare(left, lcmd, gripper_states)
        else:
            ltraj, lgrip, lok = [left._arm_qpos(left.agent)], gripper_states[0], True
        if not rdone:
            rtraj, rgrip, rok = _prepare(right, rcmd, gripper_states)
        else:
            rtraj, rgrip, rok = [right._arm_qpos(right.agent)], gripper_states[1], True
        last = _costep(env, ltraj, lgrip, rtraj, rgrip)
        if not ldone:
            try:
                lcmd = lgen.send(lok)
            except StopIteration:
                ldone = True
        if not rdone:
            try:
                rcmd = rgen.send(rok)
            except StopIteration:
                rdone = True
    return last


def solve(env, seed=None, debug=False, vis=False, frame_cb=None, config=None,
          labels=None, **kwargs):
    """Solve one episode. Returns the final ``env.step`` tuple on success, or
    ``-1`` if motion planning fails (after a clean reset so the caller can flush an
    empty trajectory). ``frame_cb`` (no-arg) is invoked after every stepped frame so a
    caller can capture an overlaid video; ``base_env._viz_phase`` (1 or 2) tells it
    which phase is running."""
    global _frame_hook
    base_env = env.unwrapped
    try:
        return _solve(
            env, base_env, seed=seed, debug=debug, vis=vis, frame_cb=frame_cb,
            config=config, labels=labels,
        )
    except RuntimeError as exc:
        if debug:
            print(f"  motion planning failed: {exc}")
        env.reset(seed=seed)
        return -1
    finally:
        _frame_hook = None


def _solve(env, base_env, seed=None, debug=False, vis=False, frame_cb=None,
           config=None, labels=None):
    global _frame_hook
    cfg = config or PrepareFruitConfig()
    labels = labels or (LABEL_L1, LABEL_L2, LABEL_R1, LABEL_R2)
    gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
    # Move gently: the plates sit near the arms' reach limit, where a fast Cartesian
    # descent can whip the arm through a near-singular config and fling the fruit.
    speed = dict(joint_vel_limits=0.85, joint_acc_limits=0.85)
    left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis, **speed)
    right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis, **speed)
    # frames from the co-executed motions come through the module hook; frames from
    # the single-arm placement path come through each planner's own callback.
    _frame_hook = frame_cb
    left.frame_cb = right.frame_cb = frame_cb
    base_env._viz_phase = 1
    base_env._had_object_retry = False
    base_env._max_bread_grasp_step_displacement = 0.0
    base_env._monitor_bread_grasp = False

    env.reset(seed=seed)
    # settle both arms at home
    hl, hr = left._arm_qpos(left.agent), right._arm_qpos(right.agent)
    _costep(env, [hl] * SETTLE, MultiPandaArmPlanner.OPEN, [hr] * SETTLE, MultiPandaArmPlanner.OPEN)
    if hasattr(cfg, "mug_grasp_z"):
        # Prepare Tea's clockwise-rotated handle points along world X. Close along
        # world Y so both fingers contact the two sides exactly 90 degrees away
        # from the handle, then preserve this quaternion through the whole carry.
        cfg.mug_grasp_q = np.asarray(
            _fixed_top_grasp_pose(
                right,
                np.asarray(right.agent.tcp.pose.sp.p, dtype=np.float64),
                closing=np.array([0.0, 1.0, 0.0]),
            ).q,
            dtype=np.float64,
        )

    # nearest-first ordering per arm (each serves the closest fruit first).
    left_base_xy = np.asarray(base_env.left_agent.robot.pose.sp.p, dtype=np.float64)[:2]
    right_base_xy = np.asarray(base_env.right_agent.robot.pose.sp.p, dtype=np.float64)[:2]
    banana_1st, banana_2nd = _nearest_first(base_env.bananas, left_base_xy)
    orange_1st, orange_2nd = _nearest_first(base_env.oranges, right_base_xy)
    slots = _plate_slots(base_env, cfg, banana_1st, banana_2nd)

    def steps_now():
        return int(base_env.elapsed_steps[0].item())

    bounds = [steps_now()]

    def phase(banana, orange, banana_plate, orange_plate, label):
        """One phase: the two arms PICK from their separate zones AND PLACE onto the
        two plates SIMULTANEOUSLY. The left arm owns bananas, the right owns oranges.

        The plate strip is shared, so when the two reaches cross it (banana's plate to
        the right of the orange's) the right arm carries HIGHER (``coplace_cross_lift``)
        and both descend straight down into their own x-columns -- they pass at
        different altitudes rather than colliding. Any fruit that misses is recovered
        by a validated single-arm re-place (the other arm parked clear)."""
        # --- simultaneous picks (safe: the two food zones are far apart) ---
        _coexec(env, left, right,
                _pick_banana_gen(left, cfg, base_env, banana),
                _pick_orange_gen(right, cfg, base_env, orange),
                gripper_states)
        # --- pull both arms back over their bases (moving apart -> safe) ---
        _coexec(env, left, right,
                _retract_over_base_gen(left, cfg, base_env),
                _retract_over_base_gen(right, cfg, base_env),
                gripper_states)
        # --- simultaneous carry, descent, release, lift, and park ---
        # The right arm uses a higher transit altitude while the reaches cross;
        # both arms then descend into separate plate columns in lockstep.
        _coexec(
            env, left, right,
            _place_gen(
                left, cfg, base_env, banana, banana_plate,
                slots[banana_plate][0], cfg.banana_release_z, cfg.banana_settle,
            ),
            _place_gen(
                right, cfg, base_env, orange, orange_plate,
                slots[orange_plate][1], cfg.orange_release_z, cfg.orange_settle,
                transit_offset=cfg.coplace_cross_lift,
            ),
            gripper_states,
        )
        # A missed release/drop is not repaired inside a retained demonstration.
        # Abort the episode so the dataset generator discards it completely.
        if not _is_on_plate(base_env, banana, banana_plate):
            raise RuntimeError(f"{label} bread released off target plate")
        if not _is_on_plate(base_env, orange, orange_plate):
            raise RuntimeError(f"{label} orange released off target plate")
        bounds.append(steps_now())
        if debug:
            print(f"  [{label}] simultaneous placement + recovery")
            print(f"  [{label}] banana_on_plate={_is_on_plate(base_env, banana, banana_plate)} "
                  f"orange_on_plate={_is_on_plate(base_env, orange, orange_plate)} (step {steps_now()})")

    # Phase 1: first banana -> brown plate (0), first orange -> white plate (1).
    base_env._viz_phase = 1
    phase(banana_1st, orange_1st, 0, 1, "phase1")
    # Phase 2: second banana -> white plate (1), second orange -> brown plate (0).
    base_env._viz_phase = 2
    phase(banana_2nd, orange_2nd, 1, 0, "phase2")

    # Final settle: retain the episode only if the complete arrangement stays
    # successful (including face-up bread and static food) for a sustained window.
    hl, hr = left._arm_qpos(left.agent), right._arm_qpos(right.agent)
    success_streak = 0
    last = None
    for _ in range(MAX_FINAL_SETTLE):
        last = _costep(env, [hl], gripper_states[0], [hr], gripper_states[1])
        if bool(base_env.evaluate()["success"][0].item()):
            success_streak += 1
            if success_streak >= FINAL_SUCCESS_STREAK:
                break
        else:
            success_streak = 0
    if success_streak < FINAL_SUCCESS_STREAK:
        raise RuntimeError("food arrangement did not remain stably successful")
    end = steps_now()

    base_env._subgoal_segments = {
        "phase1_end": int(bounds[1]),
        "end": int(end),
        "left": [
            {"label": labels[0], "start": 0, "end": int(bounds[1])},
            {"label": labels[1], "start": int(bounds[1]), "end": int(end)},
        ],
        "right": [
            {"label": labels[2], "start": 0, "end": int(bounds[1])},
            {"label": labels[3], "start": int(bounds[1]), "end": int(end)},
        ],
    }
    if debug:
        info = last[-1]
        print(f"  [phase2] done ({end} steps): all_plates_ready="
              f"{bool(info.get('all_plates_ready').item())} "
              f"success={bool(info['success'].item())}")
    return last
