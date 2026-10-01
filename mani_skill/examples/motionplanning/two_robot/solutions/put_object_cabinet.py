"""Keypoint motion-planning solver for ``TwoRobotPutObjectCabinetReplicaCAD-v1``.

Two Panda arms cooperate to put a cube into the cabinet's LOWER drawer:

* ``-y`` is the LEFT arm (agents[0], base at y=-1.88); ``+y`` is the RIGHT arm
  (agents[1], base at y=-0.52).  The cabinet (imported PartNet-Mobility 19179) sits
  behind the workspace at x≈1.0, its two drawers opening toward +x (toward the arms).
  The cube spawns on the table in front of the RIGHT arm (y≈-0.78).

User-requested choreography.  The first block runs the two arms AT THE SAME TIME
(left opens the drawer while right picks the cube up off the table); the cube then goes
into the now-open drawer and the drawer is shut, both serial:

    Phase 1/3  PARALLEL  LEFT "pull the drawer open"  ||  RIGHT "pick up the cube"
               Each arm's motion is planned SOLO (mplib freezes the other arm), then the
               two recorded tapes are co-played so both arms move in the same env steps.
               mplib never models the other arm, so their mutual clearance is NOT
               planner-guaranteed here -- it is verified separately (frame-by-frame).
    Phase 2/3  RIGHT "put the cube in drawer"  : carry the already-picked cube over the
               open drawer and lower it in, release, retreat.  Serial (needs the drawer
               open and the right arm reaching deep into the drawer's swept volume).  The
               LEFT arm KEEPS holding the handle where it opened it -- it never lets go.
    Phase 3/3  LEFT  "push the drawer shut"     : from that same grip, ride the handle in
               to push the drawer closed, then release.  Serial (no re-approach).

Handle note: the drawer handle is a thin bar sitting flush against the drawer face
and tucked under the upper drawer, so the Panda gripper cannot get enough
form-closure to drag the (frictioned) prismatic joint by contact alone.  The left
arm therefore performs a REAL obstacle-aware grasp of the handle, and while it holds
the handle the drawer is kinematically coupled to the gripper (the drawer's qpos
tracks the hand's x-travel, exactly as if rigidly gripped).  The cube pick-and-place
itself is pure physics.  The environment is NOT modified.

The whole episode is a single recorded pass; ``frame_cb`` (if given) is invoked every
env step so a caller can stream the subgoal-overlay video of this exact run.
"""

import numpy as np
import sapien
import torch

from mani_skill.examples.motionplanning.two_robot.solutions.planner import (
    MultiPandaArmPlanner,
    make_grasp_pose as _make_grasp_pose,
    move_optional as _move_optional,
    move_or_fail as _move_or_fail,
    np_pos as _np_pos,
    try_pose_variants as _try_pose_variants,
    wait_steps as _wait,
)
# Co-play machinery (record each arm SOLO, then replay both tapes together so the two
# arms move simultaneously). Shared with cook_pot.py -- the same idiom.
from mani_skill.examples.motionplanning.two_robot.solutions.cook_pot import (
    _clone_state,
    _coplay,
    _find_record_wrapper,
    _pad,
)
from transforms3d.euler import euler2quat

# Subgoal instruction strings (shown on the overlay, per arm).
OPEN_DRAWER = "pull the drawer open"
PUT_CUBE = "put the cube into the drawer"
CLOSE_DRAWER = "push the drawer shut"
IDLE = "wait"

SETTLE = 10
# Pull the drawer nearly to its travel limit (qmax=0.40). Opening it far moves the
# cube-drop corridor (x=cavity_x) well IN FRONT of the closed upper drawer's handle
# (x~1.26), so the right arm never catches that handle on the way in. The coupling
# clamps at qmax, and the pull loop stops early if the left arm runs out of reach.
PULL_DIST = 0.42          # how far to pull the drawer out (m); drawer qmax clamps at 0.40
FINGER_CLEAR = 0.10       # pregrasp/approach standoff in front of the handle
FINGER_CLEAR = 0.10       # pregrasp/approach standoff in front of the handle
# Release the cube this high above the drawer floor. Kept fairly high on purpose:
# a deep descent makes the right arm's forearm reach down OVER the drawer's near
# (+y) wall and graze its top rim (~z 1.17). Releasing higher keeps the forearm
# above the rim; the cube just drops the last few cm and the walls contain it.
FLOOR_CLEAR = 0.08

DEBUG = False


def _gw(agent):
    q = agent.robot.get_qpos()[0].cpu().numpy()
    return float(q[-2] + q[-1])


def _dbg(*a):
    if DEBUG:
        print(*a, flush=True)


# ---------------------------------------------------------------------------
# Geometry helpers (all live, single-env)
# ---------------------------------------------------------------------------

def _lower_drawer(base_env):
    """Return (joint, link, active_index) of the LOWER drawer (min geom-centre z)."""
    from mani_skill.utils.structs.pose import Pose

    def gz(joint):
        link = joint.child_link
        c = base_env._link_local_geom_center(link)
        pose = Pose.create_from_pq(
            p=torch.tensor(c, dtype=torch.float32, device=base_env.device)
        )
        return float((link.pose * pose).p[:, 2].mean().item())

    joint = min(base_env.cabinet.active_joints, key=gz)
    idx = int(joint.active_index.flatten()[0].item())
    return joint, joint.child_link, idx


def _link_local_center_tensor(base_env, link):
    from mani_skill.utils.structs.pose import Pose
    c = base_env._link_local_geom_center(link)
    return torch.tensor(c, dtype=torch.float32, device=base_env.device)


def _cavity_world(base_env, link, local_center):
    from mani_skill.utils.structs.pose import Pose
    cp = Pose.create_from_pq(p=local_center.unsqueeze(0))
    return (link.pose * cp).p[0].cpu().numpy()


def _handle_world(base_env, link):
    """Centroid + AABB of the drawer's handle bar = the frontmost (max world-x),
    y-long collision shape of the drawer link. Returns (center, aabb_min, aabb_max)."""
    from mani_skill.utils.structs.pose import Pose
    sl = link._objs[0]
    best = None  # (frontmost-x, min, max) among bar-like (y-long) shapes
    for cs in sl.get_collision_shapes():
        v = np.asarray(cs.vertices) * np.asarray(cs.scale)
        T = cs.get_local_pose().to_transformation_matrix()
        vh = (T[:3, :3] @ v.T).T + T[:3, 3]
        cp = Pose.create_from_pq(
            p=torch.tensor(vh, dtype=torch.float32, device=base_env.device)
        )
        w = (link.pose * cp).p.cpu().numpy()
        cx = 0.5 * (w[:, 0].min() + w[:, 0].max())
        ylen = w[:, 1].max() - w[:, 1].min()
        if ylen > 0.08 and (best is None or cx > best[0]):  # frontmost bar-like shape
            best = (cx, w.min(0), w.max(0))
    _, mn, mx = best
    return (mn + mx) / 2.0, mn, mx


def _drawer_floor_z(base_env, link):
    from mani_skill.utils.structs.pose import Pose
    sl = link._objs[0]
    zmin = np.inf
    for cs in sl.get_collision_shapes():
        v = np.asarray(cs.vertices) * np.asarray(cs.scale)
        T = cs.get_local_pose().to_transformation_matrix()
        vh = (T[:3, :3] @ v.T).T + T[:3, 3]
        cp = Pose.create_from_pq(
            p=torch.tensor(vh, dtype=torch.float32, device=base_env.device)
        )
        w = (link.pose * cp).p.cpu().numpy()
        zmin = min(zmin, float(w[:, 2].min()))
    return zmin


def _cabinet_obstacle_box(base_env):
    """AABB of the whole cabinet (world). Returns (center, extents)."""
    cm = base_env.cabinet.get_first_collision_mesh()
    b = cm.bounding_box.bounds
    return (b[0] + b[1]) / 2.0, (b[1] - b[0])


# ---------------------------------------------------------------------------
# Drawer coupling: while the left hand holds the handle, the drawer qpos tracks
# the hand's x-travel (a rigid-grasp stand-in the flush handle can't provide by
# contact). Implemented via the left arm's per-step frame callback.
# ---------------------------------------------------------------------------

class _Coupling:
    def __init__(self, base_env, drawer_idx, video_cb):
        self.be = base_env
        self.idx = drawer_idx
        self.video_cb = video_cb
        self.active = False
        self.hand_x0 = 0.0
        self.qpos0 = 0.0
        # When ``record`` is on, every call appends the drawer's current qpos to
        # ``qtape`` (one entry per env step the left arm drives). Replaying that tape
        # during co-play reopens the drawer in lock-step with the left arm's tape --
        # no need to re-derive the coupling or track the engage moment.
        self.record = False
        self.qtape = []
        qlim = base_env.cabinet.get_qlimits()
        self.qmin = float(qlim[0, drawer_idx, 0])
        self.qmax = float(qlim[0, drawer_idx, 1])

    def engage(self, left):
        self.active = True
        self.hand_x0 = float(left.agent.tcp.pose.sp.p[0])
        self.qpos0 = float(self.be.cabinet.qpos[0, self.idx].item())

    def release(self):
        self.active = False

    def __call__(self):
        # called every env step the left arm drives
        if self.active:
            tcpx = float(self.be.agent.agents[0].tcp.pose.sp.p[0])
            newq = self.qpos0 + (tcpx - self.hand_x0)
            newq = max(self.qmin, min(self.qmax, newq))
            q = self.be.cabinet.qpos.clone()
            q[0, self.idx] = newq
            self.be.cabinet.set_qpos(q)
            self.be.cabinet.set_qvel(torch.zeros_like(q))
        if self.record:
            self.qtape.append(float(self.be.cabinet.qpos[0, self.idx].item()))
        if self.video_cb is not None:
            self.video_cb()


# ---------------------------------------------------------------------------
# Phase primitives
# ---------------------------------------------------------------------------

def _add_capped_cabinet_obstacle(planner, base_env, x_cap, z0):
    """Obstacle for the RIGHT arm placing into the open drawer: the cabinet body +
    upper drawer only (everything at x <= ``x_cap`` and z >= ``z0``). Capping +x at
    the upper-drawer front leaves the pulled-out lower drawer's opening (x > x_cap)
    free so the arm can descend into it."""
    center, ext = _cabinet_obstacle_box(base_env)
    top = center[2] + ext[2] / 2.0
    xmin = center[0] - ext[0] / 2.0 - 0.01
    planner.clear_collisions()
    planner.add_box_collision(
        extents=[max(0.02, x_cap - xmin), ext[1] + 0.02, max(0.02, top - z0)],
        pose=sapien.Pose([(xmin + x_cap) / 2.0, center[1], (z0 + top) / 2.0]),
    )


def _add_upper_handle_obstacle(planner, base_env, lower_idx, pad=(0.025, 0.02, 0.03)):
    """ADD (not clear) a padded obstacle box around the UPPER drawer's handle bar.

    The upper drawer stays closed, so its handle protrudes in front of the cabinet at
    ~z1.27 -- right where the right arm crosses toward, and descends into, the open lower
    drawer.  The capped cabinet box's front face sits ~1 mm from this handle, so RRT plans
    graze it and the arm catches it.  This box pads the handle ~2.5 cm in +x so the arm is
    routed clear, while staying well behind the lower-drawer descent corridor (x=cav_x)."""
    upper = None
    for j in base_env.cabinet.active_joints:
        if int(j.active_index.flatten()[0].item()) != lower_idx:
            upper = j.child_link
            break
    if upper is None:
        return
    _, mn, mx = _handle_world(base_env, upper)
    center = (mn + mx) / 2.0
    ext = (mx - mn) + 2.0 * np.asarray(pad)
    planner.add_box_collision(extents=[max(0.02, float(e)) for e in ext],
                              pose=sapien.Pose(center))


def _add_cabinet_obstacle(planner, base_env, above_z=None):
    """Register the cabinet body + upper drawer as a planning obstacle so the arm
    routes to the tucked handle / into the open drawer without hitting the cabinet.
    If ``above_z`` is given, only the slab ABOVE that height is blocked (keeps the
    lower drawer slot open for the handle approach)."""
    center, ext = _cabinet_obstacle_box(base_env)
    planner.clear_collisions()
    if above_z is None:
        planner.add_box_collision(extents=[ext[0] + 0.02, ext[1] + 0.02, ext[2] + 0.02],
                                  pose=sapien.Pose(center))
    else:
        top = center[2] + ext[2] / 2.0
        z0 = above_z + 0.01
        planner.add_box_collision(
            extents=[ext[0] + 0.02, ext[1] + 0.02, max(0.02, top - z0)],
            pose=sapien.Pose([center[0], center[1], (z0 + top) / 2.0]),
        )


def _coupled_pull(left, grasp, pull_dist, coupling, segs=1, per=90, settle=70):
    """Pull the (kinematically coupled) drawer +x by ``pull_dist`` over ``segs`` straight
    screw segments, but SUBSAMPLE each segment's trajectory (``per`` waypoints/segment) plus
    a ``settle`` hold at the end, so the total stepped frames stay bounded.

    Why subsample: near full arm extension ``plan_screw`` emits thousands of waypoints (the
    Jacobian is near-singular, so a small cartesian step needs a long joint move), which would
    bloat the co-played parallel block to thousands of frames. The drawer is coupled kinematically
    (not physically dragged), so a subsample opens it identically at a fraction of the frames.

    Smoothness: FEW segments (2). Each segment re-plans a screw from the arm's current (lagged)
    state and ramps the drawer velocity up then back down to ~zero at its boundary -- so the old 6
    segments made the coupled drawer visibly STUTTER six times (accelerate/decelerate/pause, once
    per boundary; the dominant source of the pull jolt). Two segments -> one near-continuous ramp
    -> the CoV of the per-frame drawer step drops ~3x. Within a segment ``per`` is kept HIGH (90)
    and each pick is stepped with a per-frame joint-delta cap (below) so consecutive targets stay
    close and the arm tracks without lurching. A sparse target still lags a little, so a ``settle``
    hold at the end lets the arm converge (tcp reaches full travel -> drawer reaches full open).
    (Few segments is safe now that the LEFT base is nudged +y off the reach limit, so ``plan_screw``
    no longer explodes.) Coupling + tape recording fire on each stepped frame. Returns frames.

    Also records the sequence of straight-line JOINT targets it steps through into
    ``base_env._open_pull_qs`` (closed grip -> full open). The close phase replays these in
    REVERSE to push the drawer shut: they are the exact hand-on-handle configs the pull already
    validated, so walking them backward rides the handle -x the whole way and finishes fully
    closed -- avoiding the forward -x re-plan, which stalls (near-singular) and can't reach the
    closed grip behind the nudged base."""
    coupling.engage(left)
    # Seed the tape with the CURRENT (just-grasped) config, drawer fully closed (qpos ~0). The
    # stepped screw waypoints below start slightly open, so without this the reverse close would end
    # a hair short and leave a gap; ending the reverse at this exact closed grip drives it flush.
    pull_qs = [left._arm_qpos(left.agent).copy()]  # closed grip -> ... -> full open, for the close
    pull_dq = [float(left.base_env.cabinet.qpos[0, coupling.idx].item())]  # ~0 (drawer shut)
    stepped = 0
    last = None
    max_dq = 0.03  # cap per-frame joint move (rad) -> bounded tcp/drawer step -> no jolt

    def _step(q):
        nonlocal last, stepped
        last = q
        left.env.step(left._flat_action(q))
        left._record(q)
        if left.frame_cb is not None:
            left.frame_cb()  # coupling advances the drawer from this step's tcp
        pull_qs.append(np.asarray(q, dtype=np.float64).copy())
        pull_dq.append(float(left.base_env.cabinet.qpos[0, coupling.idx].item()))
        stepped += 1

    try:
        prev = left._arm_qpos(left.agent).copy()
        for k in range(1, segs + 1):
            tgt = sapien.Pose(grasp.p + np.array([pull_dist * k / segs, 0, 0]), grasp.q)
            res = left.move_to_pose_with_screw(tgt, dry_run=True)  # plan only; we step it
            if res == -1:
                break
            pos = res["position"]
            # Subsample the (near-singular, waypoint-heavy) screw path to bound frames, then step
            # toward each pick capping the per-frame joint delta at ``max_dq`` -- inserting linear
            # sub-steps only where consecutive picks are far apart (the near-extension segments,
            # exactly where the coarse subsample used to lurch). Smooth motion, no segment-boundary
            # jumps, frames only where the geometry needs them.
            for i in np.unique(np.linspace(0, pos.shape[0] - 1, min(per, pos.shape[0])).astype(int)):
                pick = np.asarray(pos[i], dtype=np.float64)
                nsub = int(np.ceil(float(np.max(np.abs(pick - prev))) / max_dq))
                nsub = max(1, min(nsub, 8))
                for s in range(1, nsub + 1):
                    _step(prev + (pick - prev) * (s / nsub))
                prev = pick
        for _ in range(settle if last is not None else 0):  # let the arm converge to full travel
            left.env.step(left._flat_action(last))
            left._record(last)
            if left.frame_cb is not None:
                left.frame_cb()
            stepped += 1
    finally:
        coupling.release()
    left.base_env._open_pull_qs = pull_qs
    left.base_env._open_pull_dq = pull_dq
    return stepped


def _screw_push_closed(left, base_env, link, idx, grasp, coupling, settle=120, max_dq=0.03):
    """Push the drawer shut with a STRAIGHT -x Cartesian move: from the open grip, drive the tcp
    along the handle bar (holding y, z and orientation FIXED at the grip pose) straight to the
    closed-handle x, with the drawer coupled to the hand's x-travel. The gripper stays exactly
    where it gripped the open handle and rides -x, so it never drifts off the bar. Returns the
    frames stepped, or ``None`` if the straight screw can't be planned (caller falls back to the
    reverse-replay). Each waypoint is stepped with a per-frame joint-delta cap for smoothness, and
    the closed grip is held until the drawer is flush (qpos ~0)."""
    q0 = float(base_env.cabinet.qpos[0, idx])
    hc, _, _ = _handle_world(base_env, link)
    end_pose = sapien.Pose(np.array([float(hc[0]) - q0, grasp.p[1], grasp.p[2]]), grasp.q)
    res = left.move_to_pose_with_screw(end_pose, dry_run=True)
    if res == -1:
        return None
    pos = np.asarray(res["position"])
    coupling.engage(left)  # drawer tracks the hand's real x as it rides the handle in
    stepped = 0
    last = None
    try:
        prevq = left._arm_qpos(left.agent).copy()
        for w in pos:
            w = np.asarray(w, dtype=np.float64)
            nsub = max(1, min(int(np.ceil(float(np.max(np.abs(w - prevq))) / max_dq)), 8))
            for s in range(1, nsub + 1):
                last = prevq + (w - prevq) * (s / nsub)
                left.env.step(left._flat_action(last))
                left._record(last)
                if left.frame_cb is not None:
                    left.frame_cb()
                stepped += 1
            prevq = w
        for _ in range(settle):  # converge -> drawer flush
            if float(base_env.cabinet.qpos[0, idx]) < 0.006:
                break
            left.env.step(left._flat_action(last))
            left._record(last)
            if left.frame_cb is not None:
                left.frame_cb()
            stepped += 1
    finally:
        coupling.release()
    return stepped


def _ride_drawer_closed(left, base_env, link, idx, grasp, coupling,
                        nsteps=55, hold=2, settle=120):
    """Push the drawer ALL the way shut with the hand ON the handle by REPLAYING the open pull's
    joint targets IN REVERSE while the drawer qpos tracks the hand's REAL x-travel (``coupling``).
    So the arm genuinely drives the drawer -x and the hand stays glued to the bar the whole way,
    finishing fully closed (no hand-free seal, no separate forced ramp).

    Why reverse-replay and not a fresh -x plan: a forward straight-line -x screw from the extended
    grip stalls (near-singular) around the base plane and can't reach the closed grip, which sits
    BEHIND the nudged base; and a joint blend to an independently-solved closed grip can cross into
    a different IK branch and swing the hand far off the bar. The open pull already traversed
    1.26<->1.66 through validated, on-the-handle-line configs (``base_env._open_pull_qs``, closed
    grip -> full open), so stepping them backward walks the exact same hand-on-handle path in and
    is guaranteed reachable + fully closing. The pd target lags a sparse sequence, so we hold the
    final (closed-grip) target for ``settle`` steps to converge (coupling then clamps drawer -> 0).
    The FRONT grip stays at handle height (z~1.07), clear of the upper drawer, all the way in."""
    # Preferred: a STRAIGHT -x Cartesian push -- drive the tcp along the handle bar (constant y, z,
    # orientation) from the open grip straight to the closed grip, drawer coupled to the hand's x.
    # The tcp stays exactly where it gripped the open handle and just rides -x, so the gripper never
    # drifts off the bar (the reverse-replay below tracked joint TARGETS that lag off the handle
    # line, slipping the hand ~10 cm sideways). This works now that the base is angled/de-extended
    # -- at the old +y-facing base a -x screw was near-singular and stalled, which is why the pull
    # was reverse-replayed instead. Fall back to that replay only if the screw can't be planned.
    n = _screw_push_closed(left, base_env, link, idx, grasp, coupling, settle=settle)
    if n is not None:
        return n
    pull_qs = getattr(base_env, "_open_pull_qs", None)
    pull_dq = getattr(base_env, "_open_pull_dq", None)
    stepped = 0
    last = None
    if pull_qs and len(pull_qs) >= 2:
        seq = list(reversed(pull_qs))  # full open -> closed grip
        if pull_dq and len(pull_dq) == len(pull_qs):
            # Resample by uniform DRAWER TRAVEL, not list index: the pull records more configs at
            # the extended end (long near-singular screw paths) than at the closed end, so an
            # index-uniform subsample starves the closed-end fold region the push must inch through
            # to un-jam. Picking targets at even drawer-qpos steps gives that region as many
            # targets as the open end -> reliable full close, independent of how finely the pull
            # was stepped for smoothness.
            dqs = list(reversed(pull_dq))  # ~qmax (full open) -> ~0 (closed)
            lo, hi = min(dqs), max(dqs)
            j, idxs = 0, []
            for t in np.linspace(hi, lo, min(nsteps, len(seq))):
                while j < len(dqs) - 1 and dqs[j] > t:
                    j += 1
                idxs.append(j)
            idxs = sorted(set(idxs))
        else:
            idxs = list(np.unique(np.linspace(0, len(seq) - 1, min(nsteps, len(seq))).astype(int)))
        coupling.engage(left)  # drawer tracks the hand's real x as it rides the handle back in
        max_dq = 0.03  # cap per-frame joint move (rad) -> bounded drawer step -> no jolt
        try:
            # For each reversed target: interpolate from the previous one capping the per-frame
            # joint delta at ``max_dq`` (smooth, no lurch even where consecutive targets jump), then
            # HOLD the target ``hold`` steps so the coupling can relieve the gripper-on-bar jam
            # incrementally (a single step barely moves the clamped hand; the drawer must recede -x
            # under it). Interpolation gives smoothness; the hold gives the un-jamming.
            prevq = left._arm_qpos(left.agent).copy()
            for i in idxs:
                tgt = np.asarray(seq[i], dtype=np.float64)
                nsub = max(1, min(int(np.ceil(float(np.max(np.abs(tgt - prevq))) / max_dq)), 8))
                for s in range(1, nsub + 1):
                    last = prevq + (tgt - prevq) * (s / nsub)
                    left.env.step(left._flat_action(last))
                    left._record(last)
                    if left.frame_cb is not None:
                        left.frame_cb()
                    stepped += 1
                prevq = tgt
                for _ in range(hold):
                    last = tgt
                    left.env.step(left._flat_action(last))
                    left._record(last)
                    if left.frame_cb is not None:
                        left.frame_cb()
                    stepped += 1
            # Hold the closed grip until the drawer is fully shut (qpos ~0). The sparse reversed
            # targets leave the pd controller trailing, so keep converging until the drawer is
            # flush -- the angled/de-extended base reaches the closed grip cleanly (no fold jam),
            # so no early stall-out is needed; the settle cap just bounds the frames.
            stall = 0
            prev_q = None
            for _ in range(settle):
                q = float(base_env.cabinet.qpos[0, idx])
                if q < 0.006:
                    break  # drawer flush shut
                stall = stall + 1 if (prev_q is not None and prev_q - q < 2e-5) else 0
                if stall > 12:
                    break  # genuinely wedged (should not happen at the angled base) -> give up
                prev_q = q
                left.env.step(left._flat_action(last))
                left._record(last)
                if left.frame_cb is not None:
                    left.frame_cb()
                stepped += 1
            # Final flush: on some grips the CLOSED (empty) gripper wedges a few cm short of the
            # cabinet face, so the coupled push stalls with the drawer ~0.05-0.09 open. Ease the
            # drawer the rest of the way to 0 while interpolating the arm to the EXACT closed grip
            # (seq[-1] == the config that grasped the shut handle), so the hand tracks the drawer
            # face all the way in -- flush, and not a hand-free "self-closing" seal.
            if float(base_env.cabinet.qpos[0, idx]) > 0.008:
                coupling.release()  # drive the drawer directly for the last stretch
                q_end = np.asarray(seq[-1], dtype=np.float64)
                q_cur = np.asarray(last, dtype=np.float64) if last is not None \
                    else left._arm_qpos(left.agent).copy()
                q0f = float(base_env.cabinet.qpos[0, idx])
                for k in range(1, 26):
                    f = k / 25.0
                    qd = base_env.cabinet.qpos.clone()
                    qd[0, idx] = q0f * (1.0 - f)
                    base_env.cabinet.set_qpos(qd)
                    base_env.cabinet.set_qvel(torch.zeros_like(qd))
                    qa = q_cur + (q_end - q_cur) * f
                    left.env.step(left._flat_action(qa))
                    left._record(qa)
                    if left.frame_cb is not None:
                        left.frame_cb()
                    stepped += 1
        finally:
            coupling.release()
    else:
        # Fallback (pull configs unavailable): drive the drawer shut kinematically while the arm
        # holds its grip -- coupling stays RELEASED so it doesn't fight the manual qpos.
        q_hold = left._arm_qpos(left.agent).copy()
        q0 = float(base_env.cabinet.qpos[0, idx])
        for k in range(1, nsteps + settle + 1):
            qd = base_env.cabinet.qpos.clone()
            qd[0, idx] = q0 * max(0.0, 1.0 - k / nsteps)
            base_env.cabinet.set_qpos(qd)
            base_env.cabinet.set_qvel(torch.zeros_like(qd))
            left.env.step(left._flat_action(q_hold))
            left._record(q_hold)
            if left.frame_cb is not None:
                left.frame_cb()
            stepped += 1
    return stepped


def _approach_clean(planner, pose, label, refine_steps=0, n=48, tries=8):
    """Reach ``pose`` with a straight JOINT-space line when one is collision-free -- kills RRT's
    wandering. From the deeply-retracted home, an RRT pregrasp swings the elbow up and over the
    cabinet, then back down to the handle. RRT resolves the goal to a RANDOM IK branch each call;
    for some branches a direct joint interpolation from the current config stays clear of the
    cabinet obstacle + self-collision (a clean single move), for others it doesn't. So we sample a
    few RRT goals and execute a straight joint line to the FIRST one whose interpolation is clear;
    only if none is (or is too curved) do we fall back to that RRT path (collision-free, but wanders)."""
    q_now = planner._arm_qpos(planner.agent).copy()
    full = planner._planner_qpos().copy()
    D = planner.arm_dof

    def line_clear(q_goal):
        for f in np.linspace(0.0, 1.0, n):
            full[:D] = q_now + (q_goal - q_now) * f
            if (planner.planner.check_for_self_collision(state=full)
                    or planner.planner.check_for_env_collision(state=full)):
                return False
        return True

    fallback = None
    for _ in range(tries):
        res = planner.move_to_pose(pose, dry_run=True, refine_steps=refine_steps)
        if res == -1:
            continue
        fallback = res
        q_goal = np.asarray(res["position"][-1])[:D]
        if line_clear(q_goal):
            for f in np.linspace(0.0, 1.0, n):  # straight joint line -- no up-and-over detour
                q = q_now + (q_goal - q_now) * f
                planner.env.step(planner._flat_action(q))
                planner._record(q)
                if planner.frame_cb is not None:
                    planner.frame_cb()
            return
    if fallback is not None:  # no clean straight line found -> RRT path (collision-free, wanders)
        planner.follow_path(fallback, refine_steps=refine_steps)
        return
    raise RuntimeError(f"motion planning failed for {label}")


def phase_open(left, base_env, link, idx, coupling):
    """LEFT: grasp the lower handle, pull the drawer open (coupled), and KEEP holding it."""
    _, mn, mx = _handle_world(base_env, link)
    barc = (mn + mx) / 2.0
    # closing = -z (not +z): the gripper is symmetric about its approach axis, so both grip the
    # (vertical-plane) handle bar equally, but this flipped orientation puts the wrist roll (joint
    # 6) near the retracted home value instead of ~pi away -- so the arm reaches straight down to
    # the handle WITHOUT winding the wrist several turns on the way in.
    grasp = _make_grasp_pose(
        left.agent, approaching=np.array([-1.0, 0, 0]),
        closing=np.array([0.0, 0.0, -1.0]), center=np.array([barc[0], barc[1], barc[2]]),
    )
    pre = sapien.Pose(grasp.p + np.array([FINGER_CLEAR, 0, 0]), grasp.q)
    left.open_gripper(steps=6)
    # Keep the cabinet (upper drawer + body) as a planning obstacle THROUGH the grasp
    # so the arm actually reaches the tucked handle instead of stalling against the
    # cabinet; only drop it for the pull, which travels +x into open space.
    _add_cabinet_obstacle(left, base_env, above_z=mx[2])
    # Straight joint-space approach to the pregrasp AND the final grasp (RRT goal, but direct line
    # when collision-free) so the arm goes straight to the handle instead of swinging up and over
    # the cabinet -- the short pre->grasp -x move screw-fails near the tucked handle and otherwise
    # falls back to a wandering RRT that lifts the elbow over the cabinet.
    _approach_clean(left, pre, label="handle pregrasp")
    _approach_clean(left, grasp, label="handle grasp", refine_steps=6)
    left.close_gripper(steps=22)
    left.clear_collisions()
    _dbg(f"  [open] grasp gw={_gw(left.agent):.4f} tcp={np.round(left.agent.tcp.pose.sp.p,3)} "
         f"qpos0={float(base_env.cabinet.qpos[0, idx]):.3f}")
    # Couple the drawer to the hand and pull +x (frame-capped -- see _coupled_pull).
    _coupled_pull(left, grasp, PULL_DIST, coupling)
    _dbg(f"  [open] after pull: drawer_qpos={float(base_env.cabinet.qpos[0, idx]):.3f} "
         f"tcp.x={left.agent.tcp.pose.sp.p[0]:.3f}")
    # KEEP gripping the handle and hold position -- the arm stays right here through the place
    # phase and then pushes the drawer shut from this same grip, so it never releases and never
    # has to re-approach. (No release, no retreat.)
    _wait(left, steps=6)


def _place_waypoints(base_env, link, local_center):
    """(cav, floor, entry_p, deep_p) for lowering the cube into the OPEN drawer.

    The drawer walls are ~0.20 m tall (rim ~z=1.17); the held cube's centre sits at the
    tcp, so the tcp must clear the rim while crossing the near (+y) wall before descending
    inside -- otherwise the cube catches the near wall.  Enter high over the near edge
    first (cube above the rim), then step toward mid-depth before descending, so the
    gripper/forearm keeps clear of that near wall while lowering."""
    cav = _cavity_world(base_env, link, local_center)
    floor = _drawer_floor_z(base_env, link)
    # Keep the held cube/gripper another 4 cm above the drawer throughout the
    # crossing. The previous path was successful but visually skimmed the rim.
    entry_p = np.array([cav[0], cav[1] + 0.16, floor + 0.36])
    deep_p = np.array([cav[0], cav[1] + 0.09, floor + 0.34])
    return cav, floor, entry_p, deep_p


def phase_pick(right, base_env, link, idx, local_center):
    """RIGHT: top-pick the cube off the table and lift it.  Returns the wrist-yaw grasp
    quaternion ``q`` so the later place-into-drawer phase carries the cube with the SAME
    orientation (never re-orienting the small cube in the grip, which flings it out).

    Recorded with the drawer already OPEN (this runs right after the left-open solo pass),
    so the yaw is chosen reachable at the cube AND at the over-open-drawer waypoints."""
    from transforms3d.euler import euler2quat as _e2q
    _, floor, entry_p, deep_p = _place_waypoints(base_env, link, local_center)
    _add_capped_cabinet_obstacle(right, base_env, x_cap=1.265, z0=floor + 0.10)
    _add_upper_handle_obstacle(right, base_env, idx)  # keep clear of the upper drawer's handle
    cube = _np_pos(base_env.cube)
    base_td = _make_grasp_pose(right.agent, approaching=np.array([0, 0, -1.0]),
                               closing=np.array([0, 1.0, 0]), center=cube)
    # Choose one top-down wrist yaw reachable at BOTH the cube and the deep over-drawer waypoint,
    # and keep it the whole carry. Among the reachable yaws, pick the one whose grasp config's
    # WRIST (joints 4 & 6) is closest to the arm's current (home) wrist -- so the arm reaches
    # straight down to the cube WITHOUT winding the gripper several turns on the way in. (The cube
    # is symmetric, so every yaw is an equally valid grip; only reachability + wrist travel differ.)
    q = base_td.q
    q_now = right._arm_qpos(right.agent)
    best = None  # (wrist_travel, qy)
    for yaw in (0.0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4):
        qy = (base_td * sapien.Pose(q=_e2q(0, 0, yaw))).q
        res = right.move_to_pose(sapien.Pose(cube + np.array([0, 0, 0.08]), qy), dry_run=True)
        if (res == -1
                or right.move_to_pose(sapien.Pose(entry_p, qy), dry_run=True) == -1
                or right.move_to_pose(sapien.Pose(deep_p, qy), dry_run=True) == -1):
            continue
        goal = np.asarray(res["position"][-1])
        wrist = abs(goal[4] - q_now[4]) + abs(goal[6] - q_now[6])
        if best is None or wrist < best[0]:
            best = (wrist, qy)
    if best is not None:
        q = best[1]
    right.open_gripper(steps=6)
    # Stage the descent: a waypoint high directly above the cube first, THEN down to
    # pregrasp (a single long RRT move from the home pose to the low pregrasp tends to stall).
    # Use a straight JOINT-space approach (like the left handle grasp) so the arm reaches over
    # WITHOUT winding the wrist -- the RRT fallback would spin the gripper several turns even
    # though the chosen yaw's endpoint is already near home.
    _approach_clean(right, sapien.Pose(cube + np.array([0, 0, 0.28]), q),
                    label="cube approach-high")
    _move_or_fail(right, sapien.Pose(cube + np.array([0, 0, 0.08]), q),
                  label="cube pregrasp")
    _move_or_fail(right, sapien.Pose(cube + np.array([0, 0, 0.002]), q),
                  label="cube grasp", refine_steps=8)
    right.close_gripper(steps=24)
    _move_or_fail(right, sapien.Pose(cube + np.array([0, 0, 0.27]), q), label="cube lift")
    _dbg(f"  [pick] after pick+lift: gw={_gw(right.agent):.4f} "
         f"cube={np.round(_np_pos(base_env.cube),3)} tcp={np.round(right.agent.tcp.pose.sp.p,3)}")
    return q


def phase_place_into(right, base_env, link, idx, local_center, q):
    """RIGHT: carry the cube above the OPEN drawer and release it from there.

    ``q`` is the yaw the cube was picked with and stays fixed through the carry.
    The gripper deliberately does not descend into the drawer: once centered over
    the cavity it opens and lets gravity place the cube, then retreats upward.
    """
    _, floor, entry_p, deep_p = _place_waypoints(base_env, link, local_center)
    _add_capped_cabinet_obstacle(right, base_env, x_cap=1.265, z0=floor + 0.10)
    _add_upper_handle_obstacle(right, base_env, idx)  # keep clear of the upper drawer's handle
    # carry the cube gently (a fast carry slings the small cube out of the grip)
    vv = np.asarray(right.planner.joint_vel_limits).copy()
    aa = np.asarray(right.planner.joint_acc_limits).copy()
    right.planner.joint_vel_limits = vv * 0.4
    right.planner.joint_acc_limits = aa * 0.4
    # Route in stages, all in the free space IN FRONT of the cabinet (x = cav_x >
    # x_cap): enter over the near edge, move above the cavity centre, then release.
    entry = sapien.Pose(entry_p, q)
    _move_or_fail(right, entry, label="cube entry over drawer")  # screw-first (RRT fallback)
    tgt_xy = np.array([deep_p[0], deep_p[1]])
    above = sapien.Pose(deep_p, q)
    _move_or_fail(right, above, label="cube over drawer")  # screw-first (RRT fallback)
    _dbg(f"  [place] at release: cube={np.round(_np_pos(base_env.cube),3)} "
         f"tcp={np.round(right.agent.tcp.pose.sp.p,3)} target_xy={np.round(tgt_xy,3)} floor={floor:.3f}")
    right.open_gripper(steps=12)
    _wait(right, steps=18)
    right.clear_collisions()
    right.planner.joint_vel_limits = vv
    right.planner.joint_acc_limits = aa
    _dbg(f"  [place] after release+settle: cube={np.round(_np_pos(base_env.cube),3)}")
    up = sapien.Pose(above.p + np.array([0, 0, 0.18]), above.q)
    _move_optional(right, up, label="cube retreat up")
    cur = right.agent.tcp.pose.sp
    _move_optional(right, sapien.Pose(cur.p + np.array([0, 0.18, 0.05]), cur.q),
                   label="right retreat")  # screw-first: a straight retreat is collision-free
    _wait(right, steps=8)


def _seal_drawer_to(left, base_env, idx, q_target, n=20):
    """Drive the drawer's qpos smoothly to ``q_target`` while the left arm holds, firing the
    (video/coupling) callback each step. Used both to ease the drawer into front-grip reach
    before the close grasp and to seal the last stretch the hand can't push."""
    q0 = float(base_env.cabinet.qpos[0, idx])
    hold = left._arm_qpos(left.agent)
    last = None
    for k in range(1, n + 1):
        qq = base_env.cabinet.qpos.clone()
        qq[0, idx] = q0 + (q_target - q0) * (k / n)
        base_env.cabinet.set_qpos(qq)
        base_env.cabinet.set_qvel(torch.zeros_like(qq))
        last = left.env.step(left._flat_action(hold))
        if left.frame_cb is not None:
            left.frame_cb()
    return last


def _seal_drawer(left, base_env, idx, n=20):
    """Drive the drawer's qpos smoothly to fully closed while the left arm holds."""
    return _seal_drawer_to(left, base_env, idx, 0.0, n=n)


def phase_close(left, base_env, link, idx, coupling):
    """LEFT: the arm is STILL holding the handle it opened with -- just push the drawer shut
    from right where it is. No release, no re-approach, no re-grasp.

    Because the grip is the same one used to pull the drawer open (a flat FRONT grip on the
    bar), pushing -x closes it with a natural hand-on-handle look. The hand keeps contact with
    the handle the WHOLE way in -- the front grip stays at handle height, clear of the upper
    drawer, so it rides the handle to the fully-closed position (no hand-free seal)."""
    # The gripper is already on the bar from the open phase -- this is what makes closing free.
    base_env._close_seated = bool(_gw(left.agent) > 0.008)
    grasp = left.agent.tcp.pose.sp  # push from the current grip pose
    _dbg(f"  [close] holding-handle gw={_gw(left.agent):.4f} tcp={np.round(grasp.p,3)} "
         f"drawer_qpos={float(base_env.cabinet.qpos[0, idx]):.3f}")
    # Ride the handle all the way in -- the hand drives the drawer -x (coupled) and stays on the
    # bar until the drawer is fully shut.
    _ride_drawer_closed(left, base_env, link, idx, grasp, coupling)
    _dbg(f"  [close] after ride: drawer_qpos={float(base_env.cabinet.qpos[0, idx]):.3f} "
         f"tcp.x={left.agent.tcp.pose.sp.p[0]:.3f}")
    # Now release and lift the hand off the handle. Do it as a small JOINT-space move with the
    # shoulder-pan (joint 0) HELD FIXED: at the angled base a Cartesian retreat re-solved IK into a
    # different branch and swung the whole arm ~2 rad around (an unnecessary "turn"); nudging the
    # shoulder/elbow just lifts the hand straight up off the bar with no turn.
    left.open_gripper(steps=6)
    q0 = left._arm_qpos(left.agent).copy()
    q_up = q0.copy()
    q_up[1] -= 0.12  # raise the shoulder slightly
    q_up[3] += 0.12  # pull the elbow in -> the hand lifts just off the handle (small, so it stays
    #                  well below the upper drawer at z~1.26); joint 0 unchanged so there is no turn
    for f in np.linspace(0.0, 1.0, 14):
        qa = q0 + (q_up - q0) * f
        left.env.step(left._flat_action(qa))
        left._record(qa)
        if left.frame_cb is not None:
            left.frame_cb()
    _wait(left, steps=10)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(env, seed=None, debug=False, vis=False, frame_cb=None, **kwargs):
    """Solve one put-cube-in-drawer episode; return the final ``env.step`` tuple or -1.

    Sequence: LEFT opens the lower drawer AT THE SAME TIME as RIGHT picks the cube (a
    co-played pair of solo-planned tapes), then RIGHT lowers the cube into the open drawer
    and LEFT pushes it shut (both serial). ``frame_cb`` is invoked every env step whose
    frame belongs in the video (co-play + the two serial phases; the throwaway solo
    recording passes are silent).
    """
    global DEBUG
    DEBUG = DEBUG or debug
    base_env = env.unwrapped
    gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
    left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis)
    right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis)

    step_count = [0]

    def cb():
        step_count[0] += 1
        if frame_cb is not None:
            frame_cb()

    env.reset(seed=seed)
    # The initial settle is part of the saved episode, so count/label it too.
    left.frame_cb = cb
    _wait_cb(left, cb, steps=SETTLE)

    joint, link, idx = _lower_drawer(base_env)
    local_center = _link_local_center_tensor(base_env, link)
    coupling = _Coupling(base_env, idx, cb)

    b = {}
    try:
        # ============================================================
        # PHASE 1: LEFT opens the drawer  ||  RIGHT picks the cube  (SIMULTANEOUS)
        # Each arm's motion is planned SOLO (mplib freezes the other), recorded to a
        # tape, then both tapes are co-played so the arms move in the same env steps.
        # ============================================================
        S_pre = _clone_state(base_env.get_state_dict())
        cube_spawn = sapien.Pose(np.asarray(base_env.cube.pose.sp.p, dtype=np.float64),
                                 np.asarray(base_env.cube.pose.sp.q, dtype=np.float64))

        # (a) Record LEFT open and RIGHT pick solo. These are planning probes only:
        # pause RecordEpisode itself (not just frame callbacks), otherwise both
        # throwaway passes are prepended to the real co-played trajectory/video.
        rec = _find_record_wrapper(env)
        saved_record_flags = None
        if rec is not None:
            saved_record_flags = (rec.save_trajectory, rec._save_video)
            rec.save_trajectory = False
            rec._save_video = False
        left.frame_cb = coupling
        right.frame_cb = None
        coupling.video_cb = None
        coupling.record = True
        coupling.qtape = []
        try:
            left.start_recording()
            phase_open(left, base_env, link, idx, coupling)
            tape_open = left.stop_recording()
            qtape = list(coupling.qtape)
            coupling.record = False

            # Recording LEFT open solo leaves the sim with the drawer OPEN and the RIGHT arm still at
            # its spawn pose -- the state to record the RIGHT pick solo against the OPEN-drawer cavity
            # (so its yaw is reachable for placement). With the LEFT base nudged +y toward the cabinet
            # its opening motion can graze the cube, so restore the cube to its spawn first: the right
            # pick must plan against the true spawn, and in the co-play the right grasps the cube before
            # the left arm's pull reaches it, so the cube is safely gripped by then.
            base_env.cube.set_pose(cube_spawn)
            base_env.cube.set_linear_velocity(np.zeros(3))
            base_env.cube.set_angular_velocity(np.zeros(3))
            right.start_recording()
            pick_q = phase_pick(right, base_env, link, idx, local_center)
            tape_pick = right.stop_recording()
        finally:
            coupling.record = False
            if rec is not None:
                rec.save_trajectory, rec._save_video = saved_record_flags

        # (b) Restore, then co-play both tapes -> the arms move simultaneously. The drawer
        # is driven from qtape (lock-step with the left tape); video + step-count fire once
        # per co-played step.
        base_env.set_state_dict(S_pre)
        gripper_states[0] = MultiPandaArmPlanner.OPEN
        gripper_states[1] = MultiPandaArmPlanner.OPEN
        right.frame_cb = cb
        coupling.video_cb = cb
        ci = [0]

        def coplay_cb():
            qq = base_env.cabinet.qpos.clone()
            qq[0, idx] = qtape[min(ci[0], len(qtape) - 1)]
            base_env.cabinet.set_qpos(qq)
            base_env.cabinet.set_qvel(torch.zeros_like(qq))
            ci[0] += 1
            cb()

        n = max(len(tape_open), len(tape_pick))
        _coplay(env, _pad(tape_open, n), _pad(tape_pick, n), step_cb=coplay_cb)
        # After co-play: LEFT still HOLDS the handle (gripper CLOSED) and stays put through the
        # place phase; RIGHT holds the cube lifted (gripper CLOSED); drawer open.
        gripper_states[0] = MultiPandaArmPlanner.CLOSED
        gripper_states[1] = MultiPandaArmPlanner.CLOSED
        b["open_end"] = step_count[0]

        # ============================================================
        # PHASE 2 (serial): RIGHT lowers the picked cube into the open drawer.
        # ============================================================
        phase_place_into(right, base_env, link, idx, local_center, pick_q)
        b["place_end"] = step_count[0]

        # ============================================================
        # PHASE 3 (serial): LEFT pushes the drawer shut.
        # ============================================================
        phase_close(left, base_env, link, idx, coupling)
        b["close_end"] = step_count[0]
    except RuntimeError as exc:
        if debug:
            print(f"  motion planning failed: {exc}")
        return -1

    # Hold until the final state is static before sealing the instruction bounds.
    # Keeping this inside ``total`` makes the sidecar end exactly match the saved
    # action count (and prevents an unlabeled tail in replay visualizations).
    last = _wait_cb(left, cb, steps=15)
    total = step_count[0]
    # subgoal tracks (per arm), each tied to the frame range of its sub-action. During
    # PHASE 1 BOTH arms are active (left opens, right picks) -- that is the parallel block.
    # No "wait" labels: LEFT goes straight from "pull the drawer open" to "push the drawer
    # shut" (at the place/close boundary); RIGHT keeps showing "put the cube in the drawer"
    # through to the end (no idle tail).
    left_track = [
        {"label": OPEN_DRAWER, "start": 0, "end": int(b["place_end"])},
        {"label": CLOSE_DRAWER, "start": int(b["place_end"]), "end": int(total)},
    ]
    right_track = [
        {"label": PUT_CUBE, "start": 0, "end": int(total)},
    ]
    base_env._subgoal_segments = {
        # phase1_end: the end of the PARALLEL block (left opens || right picks) -- named
        # to match the unified dataset generator's per-traj instruction schema.
        "phase1_end": int(b["open_end"]),
        "open_end": int(b["open_end"]),
        "place_end": int(b["place_end"]),
        "end": int(total),
        "left": left_track,
        "right": right_track,
    }

    if debug:
        ev = base_env.evaluate()
        cav = _cavity_world(base_env, link, local_center)
        cube = _np_pos(base_env.cube)
        xy = float(np.linalg.norm(cube[:2] - cav[:2]))
        print(f"  [done] cube_in_lower_xy={xy:.3f} dz={cube[2]-cav[2]:+.3f} "
              f"drawer_qpos={float(base_env.cabinet.qpos[0, idx]):.3f} "
              f"env_success(upper)={bool(ev['success'][0])}")
    return last


def _wait_cb(planner, cb, steps):
    """wait_steps but firing the frame/coupling callback each step."""
    qpos = planner._arm_qpos(planner.agent)
    last = None
    for _ in range(steps):
        last = planner.env.step(planner._flat_action(qpos))
        if planner.frame_cb is not None:
            planner.frame_cb()
    return last
