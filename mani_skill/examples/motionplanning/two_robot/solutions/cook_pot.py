"""Keypoint motion-planning solver for ``TwoRobotCookPot-v1``.

Two Panda arms cooperate to cook with a lidded pot.  ``-y`` is the LEFT arm
(agents[0], base at y=-0.75); ``+y`` is the RIGHT arm (agents[1], base at
y=+0.75).  A lidded pot sits at ``x=±0.16, y≈0`` (centred between the arms); a
square target platform sits at the opposite ``x=∓0.28``; a carrot lies in front
of one arm.

Scene geometry (measured, pot scale 0.25):

* pot body centre z = 0.052, opening rim top at z ≈ 0.117.
* the lid is on a PRISMATIC joint along world +z, travel 0..0.06 m (qpos == lift
  in metres).  Fully open (qpos 0.06) the lid bottom is at z ≈ 0.177, leaving a
  ~0.06 m side gap above the rim.  The lid has a thin (≈0.014 m in x) handle BAR
  on top running along y, centred over the pot, top at z ≈ 0.179.
* the carrot is a long thin box ≈0.19 x 0.056 x 0.049 m lying flat.
* the pot has a big loop HANDLE on each ±y side, outer tip at |Δy| ≈ 0.165 from
  the pot centre, top at z ≈ pot_z + 0.06.  The LEFT arm takes the -y handle, the
  RIGHT arm the +y handle -- 0.33 m apart, so the two grippers never collide.

User-requested sequence (two phases):

* Phase 1 -- lid + carrot:
    - LEFT  "open the lid, then close it"  : grasp the lid bar, lift it to the
      joint limit and HOLD it up while the right arm inserts the carrot, then push
      the lid back down (close) and release.
    - RIGHT "put the carrot in the pot"    : because the open gap (~0.06 m) is
      barely taller than the carrot and the gripper itself cannot fit under the
      wide lid, the carrot is *spear-inserted*: grasp it near one END horizontally
      and slide it in through the side gap so its centre lands over the pot, with
      the gripper staying OUTSIDE the lid footprint; release so it drops in.
* Phase 2 -- lift the pot:
    - BOTH arms grasp their own side handle and lift the pot together, carrying it
      over the target platform and up.  The grippers are 0.33 m apart -> no
      hand-hand collision (an explicit user requirement).

Execution mirrors ``food_serve.py``/``bottle_exchange.py``: each arm is driven
SOLO (the other frozen) to record a per-step "tape" of joint targets; the tapes
are time-aligned and replayed together so both robots move at once, and the
recorded co-play is the run whose success is verified.
"""

from dataclasses import dataclass, field

import numpy as np
import sapien

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

# Subgoal instruction strings, assigned by ROLE (which arm gets the lid vs carrot). Each role's
# phase-1 work is decomposed into sub-steps so the on-screen text tracks the CURRENT action.
LID_LIFT = "lift the lid off the pot"
LID_REPLACE = "put the lid back on"
CARROT_IN = "put the carrot in the pot"
HANDLE_GRASP = "grasp the pot handle"
POT_LIFT = "lift the pot onto the target"

SETTLE = 12
FINAL_HOLD = 30


@dataclass
class CookPotConfig:
    # Offsets are relative to the LIVE pot-body centre z. Pot rests with origin at
    # +0.0815 (base on table): rim ≈ +0.0856, lid knob bar ≈ +0.126, side handle bars
    # ≈ +0.079 at |Δy| 0.205.
    # --- lid (top bar) grasp + remove/replace ---
    lid_bar_grasp_dz: float = 0.117    # bar grasp height above the live lid pose
    lid_pregrasp_up: float = 0.05
    lid_high_clear: float = 0.11       # clearance height of the over-lid waypoint
    pot_obstacle_top: float = 0.11     # obstacle covers the pot+lid up to here (rel pot)
    pot_obstacle_xy: float = 0.30      # obstacle box footprint
    lid_grasp_close_steps: int = 45
    lid_lift_delay: int = 22           # settle-pause before lifting the lid (a lift straight
                                       # from the fresh grasp can fling it)
    lid_clear_lift: float = 0.10       # lift the lid this far up to clear the rim
    lid_aside_dx: float = -0.14        # withdraw the lid toward the front, off the pot
    lid_aside_dy: float = 0.32         # hold the lid well to the side, clear of the
                                       # carrot arm's path to the pot mouth
    lid_replace_delay: int = 26        # after the carrot is dropped, the lid arm waits this
                                       # long (while the carrot arm starts withdrawing, fed by
                                       # the partner tape) before carrying the lid back
    lid_seat_drop: float = 0.004       # tiny press past the lid's resting height: enough to
                                       # land it flush on the rim, gentle enough not to slide
                                       # it off-centre
    lid_move_speed_scale: float = 0.4  # soften the lid arm's speed while carrying the lid so
                                       # the plate does not swing/slide on its thin handle bar
    lid_release_steps: int = 14
    # --- carrot top-drop into the open pot ---
    carrot_grasp_dz: float = 0.0       # grasp at carrot centre height
    carrot_pregrasp_up: float = 0.10
    carrot_lift: float = 0.18          # lift the picked carrot this high -- ABOVE the pot rim --
                                       # before carrying it over, so it clears the rim instead of
                                       # brushing it on the way to the pot mouth
    carrot_close_steps: int = 45
    carrot_drop_z_above_rim: float = 0.05   # over-the-rim waypoint height (clears the rim)
    pot_floor_z_rel: float = -0.0575        # pot inner-floor top, relative to the pot origin
    carrot_floor_clear: float = 0.04        # release just above the floor so it settles centrally
                                            # settles instead of free-falling and bouncing the pot
    carrot_release_steps: int = 16
    carrot_mirror_dy: float = 0.34     # carrot arm retracts to its own side by this |Δy| (at
    carrot_mirror_dx: float = 0.08     # home height, stepping this far back in x off the lid
                                       # edge), MIRRORING the lid arm's first retract sideways
    carrot_hold_lower: float = 0.012   # hold the carrot arm this far below home during the lid
                                       # grasp so its controller settles AT home (not lifting)
    carrot_turn_dz: float = 0.18       # then (lid aside) come down to this height above the
                                       # carrot before descending onto it -- a normal pick
    carrot_start_delay: int = 0        # once the lid is grasped, the carrot arm turns toward the
                                       # carrot immediately (no pause) -- it retracted during the
                                       # lid approach, so it is already clear
    carrot_retreat_dy: float = 0.34    # withdraw to this |Δy| (own side) after the drop
    carrot_retreat_dz: float = 0.22    # height of the retreat-aside waypoint
    # --- pot handle grasp + lift (handle bars are long in x, THIN in y) ---
    handle_dy: float = 0.205           # |Δy| of the handle-bar centre from pot centre
    handle_dx: float = 0.0
    handle_grasp_dz: float = 0.088     # grasp near the top of the handle bar
    handle_pregrasp_up: float = 0.10
    handle_close_steps: int = 45
    lift_height: float = 0.12          # final lift above the table
    lift_set_down: float = 0.065       # lower the pot this far onto the stove at the end
    lift_speed_scale: float = 0.3      # 3x the old post-grasp speed; still moderate enough to
                                       # keep the lid seated and carrot inside during the carry
    lift_steps: int = 8
    lift_refine: int = 6
    rim_top_z_rel: float = 0.0856      # rim top relative to pot-body centre z


# ---------------------------------------------------------------------------
# Grasp-pose helpers
# ---------------------------------------------------------------------------

def _carrot_long_axis(base_env):
    """World-frame unit long axis (xy) of the carrot (its local +x is the length)."""
    R = base_env.meat.pose.sp.to_transformation_matrix()[:3, :3]
    axis = R[:, 0].copy()
    axis[2] = 0.0
    n = np.linalg.norm(axis)
    return axis / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def _top_grasp(planner, center, closing, yaw_angles=None):
    pose = _make_grasp_pose(
        planner.agent, approaching=np.array([0, 0, -1.0]), closing=closing, center=center
    )
    return _try_pose_variants(planner, pose, yaw_angles=yaw_angles)


# ---------------------------------------------------------------------------
# LEFT arm: lid open / hold / close
# ---------------------------------------------------------------------------

def _lid_bar_grasp_pose(left, cfg, base_env, extra_z=0.0):
    """Top-down grasp of the lid's top handle bar. Anchored to the LIVE lid pose (the
    lid rests a little proud of the pot, so anchoring to the pot would aim too low and
    knock the lid off)."""
    lidp = _np_pos(base_env.lid_link)
    center = np.array([lidp[0], lidp[1], lidp[2] + cfg.lid_bar_grasp_dz + extra_z])
    # the lid handle bar is long in x and THIN in y -> close the fingers across y so
    # the hand is aligned with the bar's LONG side.
    pose = _make_grasp_pose(
        left.agent, approaching=np.array([0, 0, -1.0]),
        closing=np.array([0.0, 1.0, 0.0]), center=center,
    )
    return _try_pose_variants(left, pose, yaw_angles=[0, np.pi])


def _add_pot_obstacle(planner, base_env, cfg, top_rel=None, xy=None):
    """Register the pot+lid (up to ``top_rel`` above the pot origin) as a box obstacle
    so the planner routes the arm ABOVE/AROUND the tall pot instead of sweeping it."""
    potc = _np_pos(base_env.pot_body_link)
    top = potc[2] + (cfg.pot_obstacle_top if top_rel is None else top_rel)
    foot = cfg.pot_obstacle_xy if xy is None else xy
    planner.clear_collisions()
    planner.add_box_collision(
        extents=[foot, foot, top],
        pose=sapien.Pose([potc[0], potc[1], top / 2.0]),
    )


def _lid_approach(left, cfg, base_env):
    """APPROACH the lid bar (the winding obstacle-aware part) and stop at the grasp pose with
    the gripper still OPEN -- the lid is NOT yet gripped. Returns the grasp pose. (Splitting
    the approach from the close lets the carrot arm move alongside this approach while the lid
    is still loose, so it can't drop the lid; the firm CLOSE then happens live afterwards.)"""
    grasp = _lid_bar_grasp_pose(left, cfg, base_env)
    left.open_gripper(steps=8)
    # The arm now STARTS retracted (back and up, clear of the pot), so it can go straight to
    # the lid instead of first retracting to the side. With the pot+lid as a planning obstacle,
    # RRT directly to a waypoint high over the lid (routing ABOVE the pot -- so the arm can't
    # sweep the lid off), then drop the obstacle and descend straight down onto the bar.
    _add_pot_obstacle(left, base_env, cfg)
    high = sapien.Pose(grasp.p + np.array([0, 0, cfg.lid_high_clear]), grasp.q)
    _move_or_fail(left, high, use_rrt=True, label="lid over lid")
    left.clear_collisions()
    _move_or_fail(left, grasp, label="lid grasp", refine_steps=8)
    return grasp


def _lid_grasp(left, cfg, base_env):
    """Approach AND grasp the lid bar (approach + firm close). Returns the grasp pose."""
    grasp = _lid_approach(left, cfg, base_env)
    left.close_gripper(steps=cfg.lid_grasp_close_steps)
    return grasp


def _lid_lift_aside(left, cfg, base_env, grasp):
    """Lift the grasped lid straight up off the rim and hold it aside on the arm's own
    side. A clean up+sideways motion -> safe to run in PARALLEL with the carrot pick."""
    # Optionally hold the lid still first (parallel co-play timing).
    if cfg.lid_lift_delay > 0:
        _wait(left, steps=cfg.lid_lift_delay)
    # Lift at full speed: a slow lift lets the heavy lid slip out of the grip before it
    # clears the rim. (The carry-BACK is softened instead, where speed is safe.)
    lift = sapien.Pose(grasp.p + np.array([0, 0, cfg.lid_clear_lift]), grasp.q)
    _move_or_fail(left, lift, label="lid lift off", refine_steps=6)
    side = -1.0 if left.agent_idx == 0 else 1.0
    # After the straight lift, withdraw in ONE straight (screw) move to the hold-aside pose on
    # the arm's OWN side. A straight line runs diagonally back toward this arm's own side, away
    # from the other arm (opposite +y) -- so it neither arcs across the pot (which used to shove
    # the other arm) nor doubles back forward-then-backward.
    aside = sapien.Pose(
        grasp.p + np.array([cfg.lid_aside_dx, side * cfg.lid_aside_dy, cfg.lid_clear_lift]),
        grasp.q,
    )
    _move_or_fail(left, aside, label="lid withdraw aside", refine_steps=4)
    _wait(left, steps=8)


def _left_replace_lid(left, cfg, base_env, grasp, lid_rest_z):
    """Carry the held lid back over the pot and seat it exactly where it started, then
    release+retreat.

    The lid may have slid in the grip while it was carried, so aim the lid's actual CENTRE
    at the pot centre AND its actual HEIGHT back to ``lid_rest_z`` (its measured resting
    height before it was lifted) -> the lid seats flush on the rim with no gap, exactly as
    it looked initially."""
    potc = _np_pos(base_env.pot_body_link)

    # Hold the lid aside a moment first so the carrot arm (driven by the partner tape) gets a
    # head start withdrawing before the lid arm swings in over the pot -- they don't crowd the
    # pot mouth at the same instant. The wait still steps the partner tape, so the carrot arm
    # IS moving away during it.
    if cfg.lid_replace_delay > 0:
        _wait(left, steps=cfg.lid_replace_delay)

    def centred_tcp():
        # gripper (xy, z) that lands the lid's actual CENTRE on the pot centre and its
        # underside back at its original resting height on the rim.
        lid_in_tcp = _np_pos(base_env.lid_link) - left.agent.tcp.pose.sp.p
        xy = potc[:2] - lid_in_tcp[:2]
        z = lid_rest_z - lid_in_tcp[2]
        return xy, z

    # carry + seat the lid GENTLY (soften arm speed) so the plate does not swing/slide on
    # its thin handle bar -> the lid stays centred and lands flush.
    v = np.asarray(left.planner.joint_vel_limits, dtype=np.float64).copy()
    a = np.asarray(left.planner.joint_acc_limits, dtype=np.float64).copy()
    left.planner.joint_vel_limits = v * cfg.lid_move_speed_scale
    left.planner.joint_acc_limits = a * cfg.lid_move_speed_scale
    try:
        xy, _ = centred_tcp()
        above = sapien.Pose(
            np.array([xy[0], xy[1], grasp.p[2] + cfg.lid_clear_lift]), grasp.q
        )
        _move_or_fail(left, above, use_rrt=True, label="lid carry back", refine_steps=4)
        # re-measure after the carry (the lid may have shifted in the grip) and seat to the
        # corrected centre + original resting height, pressing slightly past it so the lid
        # settles FLUSH on the rim (it cannot sink below the rim) with no gap.
        xy, z = centred_tcp()
        seat = sapien.Pose(
            np.array([xy[0], xy[1], z - cfg.lid_seat_drop]), grasp.q
        )
        _move_or_fail(left, seat, label="lid seat", refine_steps=10)
        _wait(left, steps=8)
    finally:
        left.planner.joint_vel_limits = v
        left.planner.joint_acc_limits = a
    left.open_gripper(steps=cfg.lid_release_steps)
    cur = left.agent.tcp.pose.sp
    up = sapien.Pose(cur.p + np.array([0, 0, 0.14]), cur.q)
    _move_optional(left, up, use_rrt=True, label="lid retreat up")
    _wait(left, steps=6)


# ---------------------------------------------------------------------------
# RIGHT arm: carrot spear-insert
# ---------------------------------------------------------------------------

def _carrot_mirror_retract(right, cfg, base_env):
    """Move the carrot arm out to its OWN side, MIRRORING the lid arm's first retract -- fed
    during the live lid grasp so both arms move at the SAME time, and the carrot arm is never
    sitting at the lid's near edge when the lid is lifted (so it can't catch/lift the lid).

    Unlike the lid arm (whose retract also goes UP to clear the lid), the carrot arm stays at
    its home HEIGHT and steps a little back in x: its home gripper rests right at the lid's
    edge, so rising there would lift the lid -- it must clear sideways, not up."""
    potc = _np_pos(base_env.pot_body_link)
    home_pose = right.agent.tcp.pose.sp
    home_x, home_z, home_q = float(home_pose.p[0]), float(home_pose.p[2]), np.array(home_pose.q)
    side = 1.0 if right.agent_idx == 1 else -1.0
    back_x = home_x - np.sign(potc[0] - home_x) * cfg.carrot_mirror_dx
    retract = sapien.Pose(
        np.array([back_x, potc[1] + side * cfg.carrot_mirror_dy, home_z]), home_q
    )
    _move_or_fail(right, retract, use_rrt=True, label="carrot mirror retract")


def _carrot_approach(right, cfg, base_env):
    """Turn to ABOVE the carrot and stop at the pregrasp pose (gripper OPEN, NOT yet gripping).
    Returns the grasp pose. This is pure POSITIONING (no grip), so it is safe to REPLAY as a
    partner tape alongside the live lid grab -- the actual grasp (which must be live, a replayed
    grip drops the carrot) is done afterwards by ``_carrot_grasp_lift``."""
    carrot_c = _np_pos(base_env.meat)
    axis = _carrot_long_axis(base_env)
    closing = np.array([-axis[1], axis[0], 0.0])
    grasp = _make_grasp_pose(
        right.agent, approaching=np.array([0, 0, -1.0]), closing=closing,
        center=np.array([carrot_c[0], carrot_c[1], carrot_c[2] + cfg.carrot_grasp_dz]),
    )
    grasp = _try_pose_variants(right, grasp, yaw_angles=[0, np.pi])
    pregrasp = sapien.Pose(grasp.p + np.array([0, 0, cfg.carrot_pregrasp_up]), grasp.q)
    home_q = np.array(right.agent.tcp.pose.sp.q)
    turn_z = carrot_c[2] + cfg.carrot_turn_dz
    right.open_gripper(steps=8)
    turn = sapien.Pose(np.array([carrot_c[0], carrot_c[1], turn_z]), home_q)
    _move_or_fail(right, turn, label="carrot turn to side")
    _move_or_fail(right, pregrasp, use_rrt=True, label="carrot pregrasp")
    return grasp


def _carrot_grasp_lift(right, cfg, base_env, grasp):
    """From the pregrasp pose, descend onto the carrot, close (LIVE grip), and lift it clear of
    the pot rim. Done live (not replayed) so the grip actually establishes."""
    _move_or_fail(right, grasp, label="carrot grasp", refine_steps=8)
    right.close_gripper(steps=cfg.carrot_close_steps)
    lift = sapien.Pose(grasp.p + np.array([0, 0, cfg.carrot_lift]), grasp.q)
    _move_or_fail(right, lift, use_rrt=True, label="carrot lift")
    _wait(right, steps=4)
    return grasp


def _carrot_drop(right, cfg, base_env, grasp):
    """Carry the (already-held) carrot over the now-open pot, lower it DEEP -- to just above the
    pot floor -- and release so it settles instead of free-falling and bouncing the pot around.
    (The retreat is a separate step, see ``_carrot_retreat``, so it overlaps the lid replace.)"""
    potc = _np_pos(base_env.pot_body_link)
    rim_top = potc[2] + cfg.rim_top_z_rel
    floor = potc[2] + cfg.pot_floor_z_rel
    carrot_in_tcp = _np_pos(base_env.meat) - right.agent.tcp.pose.sp.p
    # first come to a waypoint with the carrot centred ABOVE the rim (clears the rim), ...
    over_carrot = np.array([potc[0], potc[1], rim_top + cfg.carrot_drop_z_above_rim])
    over_pose = _try_pose_variants(right, sapien.Pose(over_carrot - carrot_in_tcp, grasp.q))
    drop_q = over_pose.q
    _move_or_fail(right, over_pose, use_rrt=True, label="carrot over pot")
    # ... then lower the carrot down close to the floor before releasing, so it barely drops and
    # doesn't bounce (a bounce off the floor nudges the whole pot).
    drop_carrot = np.array([potc[0], potc[1], floor + cfg.carrot_floor_clear])
    _move_or_fail(right, sapien.Pose(drop_carrot - carrot_in_tcp, drop_q),
                  label="carrot lower", refine_steps=6)
    right.open_gripper(steps=cfg.carrot_release_steps)
    _wait(right, steps=12)


def _carrot_retreat(right, cfg, base_env):
    """Withdraw the (now empty) carrot arm well clear of the pot, out to its own side, so
    the lid can be carried back over the pot. First lift straight up off the mouth, then
    swing out to the side. Designed to run WHILE the lid arm replaces the lid."""
    potc = _np_pos(base_env.pot_body_link)
    side = 1.0 if right.agent_idx == 1 else -1.0
    cur = right.agent.tcp.pose.sp
    _move_optional(right, sapien.Pose(cur.p + np.array([0, 0, 0.16]), cur.q),
                   label="carrot retreat up")
    retreat = sapien.Pose(
        np.array([potc[0], potc[1] + side * cfg.carrot_retreat_dy, potc[2] + cfg.carrot_retreat_dz]),
        cur.q,
    )
    _move_optional(right, retreat, use_rrt=True, label="carrot retreat aside")
    _wait(right, steps=8)


# ---------------------------------------------------------------------------
# Both arms: pot handle grasp + coordinated lift
# ---------------------------------------------------------------------------

def _handle_grasp_pose(planner, cfg, base_env, side_sign):
    """Top-down grasp of the side handle (side_sign -1 = -y/left, +1 = +y/right)."""
    potc = _np_pos(base_env.pot_body_link)
    center = np.array([potc[0] + cfg.handle_dx,
                       potc[1] + side_sign * cfg.handle_dy,
                       potc[2] + cfg.handle_grasp_dz])
    # the handle bar is long in x and THIN in y -> close the fingers across y.
    pose = _make_grasp_pose(
        planner.agent, approaching=np.array([0, 0, -1.0]),
        closing=np.array([0.0, 1.0, 0.0]), center=center,
    )
    return _try_pose_variants(planner, pose, yaw_angles=[0, np.pi])


def _grasp_handle(planner, cfg, base_env, side_sign):
    grasp = _handle_grasp_pose(planner, cfg, base_env, side_sign)
    pregrasp = sapien.Pose(grasp.p + np.array([0, 0, cfg.handle_pregrasp_up]), grasp.q)
    planner.open_gripper(steps=8)
    _move_or_fail(planner, pregrasp, use_rrt=True, label="handle pregrasp")
    _move_or_fail(planner, grasp, label="handle grasp", refine_steps=8)
    planner.close_gripper(steps=cfg.handle_close_steps)
    return grasp


# ---------------------------------------------------------------------------
# Co-play machinery
# ---------------------------------------------------------------------------

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


def _coplay(env, left_tape, right_tape, step_cb=None):
    """Step the env feeding BOTH arms their per-step [qpos(7), gripper] sub-actions so
    they move simultaneously. ``step_cb`` (no-arg) fires after every step."""
    n = max(len(left_tape), len(right_tape))
    last = None
    for i in range(n):
        la = left_tape[min(i, len(left_tape) - 1)]
        ra = right_tape[min(i, len(right_tape) - 1)]
        last = env.step(np.hstack([la, ra]))
        if step_cb is not None:
            step_cb()
    return last


def _finish_partner(live_arm, env, step_cb=None):
    """Play out any remaining partner-tape steps while the live arm holds its current pose,
    so the partner arm's motion completes after the live arm has finished."""
    while (live_arm.partner_tape is not None
           and live_arm._partner_cursor < len(live_arm.partner_tape)):
        env.step(live_arm._flat_action())   # holds the live arm, advances+feeds partner
        if step_cb is not None:
            step_cb()


def _find_record_wrapper(env):
    e = env
    while e is not None:
        if e.__class__.__name__ == "RecordEpisode":
            return e
        e = getattr(e, "env", None)
    return None


def _hold_action(planner):
    """The (8,) sub-action that holds an arm still at its current qpos+gripper."""
    return np.hstack([planner._arm_qpos(planner.agent),
                      planner.gripper_states[planner.agent_idx]])


def _dual_lift(env, left, right, gripper_states, cfg, base_env):
    """With BOTH arms already grasping their handle, lift+carry the pot over the target
    together (the rigid pot is carried between them). Drives the env directly."""
    potc = _np_pos(base_env.pot_body_link)
    target = _np_pos(base_env.target_square)
    dx = float(target[0] - potc[0])
    last = [None]

    def comove(dx, dy, dz, n):
        # base each move on the CURRENT tcp so a carry holds whatever height the lift
        # reached (basing it on the pre-lift tcp would drag the pot back down).
        lt0 = left.agent.tcp.pose.sp
        rt0 = right.agent.tcp.pose.sp
        for k in range(n):
            f = (k + 1) / n
            lp = sapien.Pose([lt0.p[0] + dx * f, lt0.p[1] + dy * f, lt0.p[2] + dz * f], lt0.q)
            rp = sapien.Pose([rt0.p[0] + dx * f, rt0.p[1] + dy * f, rt0.p[2] + dz * f], rt0.q)
            try:
                left_goal = left._mplib.Pose(lp.p, lp.q)
                right_goal = right._mplib.Pose(rp.p, rp.q)
            except (TypeError, AttributeError):
                # MPLib <0.2 accepted flat [p, q] targets; >=0.2 requires Pose.
                left_goal = np.concatenate([lp.p, lp.q])
                right_goal = np.concatenate([rp.p, rp.q])
            ra = left.planner.plan_screw(
                left_goal, left._planner_qpos(), time_step=base_env.control_timestep
            )
            rb = right.planner.plan_screw(
                right_goal, right._planner_qpos(), time_step=base_env.control_timestep
            )
            if ra["status"] != "Success":
                ra = left.move_to_pose(lp, dry_run=True, attempts=6)
            if rb["status"] != "Success":
                rb = right.move_to_pose(rp, dry_run=True, attempts=6)
            if ra == -1 or rb == -1:
                raise RuntimeError("dual lift fallback RRT failed")
            if ra["status"] != "Success" or rb["status"] != "Success":
                raise RuntimeError(f"dual lift plan failed L={ra['status']} R={rb['status']}")
            nl = max(len(ra["position"]), len(rb["position"]))
            for i in range(nl):
                la = ra["position"][min(i, len(ra["position"]) - 1)]
                rc = rb["position"][min(i, len(rb["position"]) - 1)]
                last[0] = env.step(np.hstack([np.hstack([la, gripper_states[0]]),
                                              np.hstack([rc, gripper_states[1]])]))
                if left.frame_cb is not None:
                    left.frame_cb()

    def settle(steps):
        for _ in range(steps):
            last[0] = env.step(np.hstack([
                np.hstack([left._arm_qpos(left.agent), gripper_states[0]]),
                np.hstack([right._arm_qpos(right.agent), gripper_states[1]])]))
            if left.frame_cb is not None:
                left.frame_cb()

    # Soften both arms' speed so the lift/carry/set-down are GENTLE -> the lid stays
    # seated and the carrot doesn't slosh out of the tight centre zone.
    saved = []
    for pl in (left, right):
        v = np.asarray(pl.planner.joint_vel_limits, dtype=np.float64)
        a = np.asarray(pl.planner.joint_acc_limits, dtype=np.float64)
        saved.append((pl, v.copy(), a.copy()))
        pl.planner.joint_vel_limits = v * cfg.lift_speed_scale
        pl.planner.joint_acc_limits = a * cfg.lift_speed_scale
    try:
        # (1) lift straight up, (2) carry over the stove AT THE SAME HEIGHT, (3) lower
        # the pot straight down onto the stove. Each move keeps the pot level.
        comove(0.0, 0.0, cfg.lift_height, cfg.lift_steps)  # lift straight up
        settle(10)
        carry_steps = max(cfg.lift_steps, int(np.ceil(abs(dx) / 0.04)))
        comove(dx, 0.0, 0.0, carry_steps)             # carry over the stove, same height
        settle(8)
        comove(0.0, 0.0, -cfg.lift_set_down, cfg.lift_steps)  # set down gently
        settle(FINAL_HOLD)
    finally:
        for pl, v, a in saved:
            pl.planner.joint_vel_limits = v
            pl.planner.joint_acc_limits = a
    return last[0]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(env, seed=None, debug=False, vis=False, frame_cb=None, **kwargs):
    """Solve one cook-pot episode and return the final ``env.step`` tuple (or -1).

    Sequence (matching the requested task):
      1. the LID arm lifts the lid off the pot and holds it aside;
      2. the CARROT arm (whichever arm the carrot spawned in front of) picks the
         carrot and drops it into the open pot;
      3. the LID arm seats the lid back on;
      4. BOTH arms grasp the side handles and lift the pot together onto the target.

    Phase 1 is inherently sequential (one arm works the shared pot while the other
    holds), so the whole episode is a SINGLE recorded pass: each sub-phase drives the
    env (captured by the RecordEpisode wrapper) and the lift moves both arms together.
    ``frame_cb`` (if given) is invoked every env step so a caller can stream a video of
    this exact verified run. Returns the final ``env.step`` tuple, or -1 on failure.
    """
    base_env = env.unwrapped
    cfg = CookPotConfig()
    gripper_states = [MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.OPEN]
    left = MultiPandaArmPlanner(env, 0, gripper_states, debug=debug, vis=vis)
    right = MultiPandaArmPlanner(env, 1, gripper_states, debug=debug, vis=vis)
    planners = [left, right]
    # count every env step (so the subgoal boundaries are exact frame indices) and
    # forward to the caller's frame callback for video streaming.
    step_count = [0]

    def cb():
        step_count[0] += 1
        if frame_cb is not None:
            frame_cb()

    left.frame_cb = right.frame_cb = cb
    rec = _find_record_wrapper(env)
    OPEN, CLOSED = MultiPandaArmPlanner.OPEN, MultiPandaArmPlanner.CLOSED

    try:
        env.reset(seed=seed)
        _wait(left, steps=SETTLE)
        # role assignment: the carrot is in front of one arm; that arm handles it. Compare the
        # carrot's y to the MIDLINE between the two arm bases (agent 0 on -y, agent 1 on +y) --
        # NOT world 0 -- so it's correct even when the whole workspace is shifted off the origin
        # (e.g. the ReplicaCAD apartment variant offsets the table to y ~= -1.2).
        mid_y = 0.5 * (float(base_env.agent.agents[0].robot.pose.sp.p[1])
                       + float(base_env.agent.agents[1].robot.pose.sp.p[1]))
        carrot_y = float(base_env.meat.pose.sp.p[1])
        carrot_idx = 1 if carrot_y >= mid_y else 0
        lid_idx = 1 - carrot_idx
        carrot_arm, lid_arm = planners[carrot_idx], planners[lid_idx]
        if debug:
            print(f"  carrot on {'+y/right' if carrot_idx == 1 else '-y/left'}; "
                  f"lid arm = {'right' if lid_idx == 1 else 'left'}")
        # measured resting height of the lid before anything touches it -> the lid is
        # seated back to exactly this height at the end (flush on the rim, no gap).
        lid_rest_z = float(_np_pos(base_env.lid_link)[2])
        # --- PHASE 1a (CONCURRENT GRAB): the right arm moves to grab the carrot at the SAME TIME
        # the left arm grabs the lid off the pot. Grips must be LIVE (a replayed grip drops the
        # object), but a replay is fine for POSITIONING and for moving an ALREADY-gripped object.
        # So the phase runs in two concurrent halves:
        #   (i)  lid GRAB live  ||  carrot APPROACH (partner replay, no grip) -> both travel in;
        #   (ii) carrot GRASP+lift live  ||  lid LIFT-ASIDE (partner replay of the held lid).
        # This keeps both grips live AND keeps the lid held-aside only briefly (just the drop),
        # so it doesn't drift out of reach before it's put back. ---
        S_init = _clone_state(base_env.get_state_dict())
        left.frame_cb = right.frame_cb = None
        saved = None
        if rec is not None:
            saved = (rec.save_trajectory, rec._save_video)
            rec.save_trajectory = False
            rec._save_video = False
        try:
            carrot_arm.start_recording()
            carrot_grasp = _carrot_approach(carrot_arm, cfg, base_env)
            tape_approach = carrot_arm.stop_recording()
        finally:
            if rec is not None:
                rec.save_trajectory, rec._save_video = saved
            left.frame_cb = right.frame_cb = cb
        base_env.set_state_dict(S_init)
        gripper_states[lid_idx] = OPEN
        gripper_states[carrot_idx] = OPEN

        # (i) grab the lid live while the carrot arm travels to the carrot (partner replay)
        lid_arm.partner_tape = tape_approach
        lid_arm.partner_idx = carrot_idx
        lid_arm._partner_cursor = 0
        grasp = _lid_grasp(lid_arm, cfg, base_env)          # LIVE: grab the lid off the pot
        _finish_partner(lid_arm, env, cb)     # let the carrot reach its pregrasp pose
        lid_arm.partner_tape = lid_arm.partner_idx = None
        gripper_states[lid_idx] = CLOSED      # lid gripped
        gripper_states[carrot_idx] = OPEN     # carrot poised above, not yet gripped
        S_grabbed = _clone_state(base_env.get_state_dict())

        # (ii) record the lid LIFT-ASIDE solo (lid already gripped -> safe to replay), then grasp
        # the carrot LIVE while the lid arm lifts it aside via that partner tape -- concurrent.
        left.frame_cb = right.frame_cb = None
        if rec is not None:
            rec.save_trajectory = False
            rec._save_video = False
        try:
            lid_arm.start_recording()
            _lid_lift_aside(lid_arm, cfg, base_env, grasp)
            tape_lift = lid_arm.stop_recording()
        finally:
            if rec is not None:
                rec.save_trajectory, rec._save_video = saved
            left.frame_cb = right.frame_cb = cb
        base_env.set_state_dict(S_grabbed)
        gripper_states[lid_idx] = CLOSED
        gripper_states[carrot_idx] = OPEN
        carrot_arm.partner_tape = tape_lift
        carrot_arm.partner_idx = lid_idx
        carrot_arm._partner_cursor = 0
        _carrot_grasp_lift(carrot_arm, cfg, base_env, carrot_grasp)   # LIVE grip
        _finish_partner(carrot_arm, env, cb)  # finish lifting the lid aside if it outlasts
        carrot_arm.partner_tape = carrot_arm.partner_idx = None
        gripper_states[lid_idx] = CLOSED      # lid held aside
        gripper_states[carrot_idx] = CLOSED   # carrot held
        b_grab = step_count[0]

        # --- PHASE 1b: drop the carrot into the open pot (live) ---
        _carrot_drop(carrot_arm, cfg, base_env, carrot_grasp)   # drop carrot in pot
        b_drop = step_count[0]

        # --- PHASE 1c: put the lid back WHILE the carrot arm withdraws, at the same time. ---
        # Record the carrot RETREAT solo (throwaway), then seat the lid LIVE (firm, accurate)
        # with the carrot retreat fed as the lid arm's "partner" tape, so both arms move at
        # once. The carrot arm pulls up-and-aside while the lid arm carries the lid back over
        # the pot -- they work opposite sides, so they stay clear.
        gripper_states[carrot_idx] = OPEN
        S_dropped = _clone_state(base_env.get_state_dict())
        left.frame_cb = right.frame_cb = None
        if rec is not None:
            saved = (rec.save_trajectory, rec._save_video)
            rec.save_trajectory = False
            rec._save_video = False
        try:
            base_env.set_state_dict(S_dropped)
            gripper_states[lid_idx] = CLOSED
            gripper_states[carrot_idx] = OPEN
            carrot_arm.start_recording()
            _carrot_retreat(carrot_arm, cfg, base_env)
            tape_retreat = carrot_arm.stop_recording()
        finally:
            if rec is not None:
                rec.save_trajectory, rec._save_video = saved
            left.frame_cb = right.frame_cb = cb

        base_env.set_state_dict(S_dropped)
        gripper_states[lid_idx] = CLOSED
        gripper_states[carrot_idx] = OPEN
        lid_arm.partner_tape = tape_retreat
        lid_arm.partner_idx = carrot_idx
        lid_arm._partner_cursor = 0
        _left_replace_lid(lid_arm, cfg, base_env, grasp, lid_rest_z)  # seat the lid back
        _finish_partner(lid_arm, env, cb)     # let the carrot retreat finish if still running
        lid_arm.partner_tape = lid_arm.partner_idx = None
        phase1_end = step_count[0]
        if debug:
            mx = float(np.linalg.norm(_np_pos(base_env.meat)[:2]
                                      - _np_pos(base_env.pot_body_link)[:2]))
            print(f"  before lift: meat_inside={bool(base_env.evaluate()['meat_inside_pot'][0])} "
                  f"meat_xy_off={mx:.3f} lid_closed={bool(base_env.evaluate()['lid_closed'][0])} "
                  f"lid_xy_off={float(base_env._lid_xy_offset()[0]):.3f} "
                  f"lid_z_off={float(base_env._lid_z_offset()[0]):.3f}")
        # --- PHASE 2: both arms grasp their handle AT THE SAME TIME (co-play the two
        # solo grasps -> simultaneous), then lift+carry the pot together. ---
        S_pre = _clone_state(base_env.get_state_dict())

        def restore_pre():
            base_env.set_state_dict(S_pre)
            gripper_states[0] = OPEN
            gripper_states[1] = OPEN

        left.frame_cb = right.frame_cb = None
        if rec is not None:
            saved = (rec.save_trajectory, rec._save_video)
            rec.save_trajectory = False
            rec._save_video = False
        try:
            restore_pre()
            left.start_recording()
            _grasp_handle(left, cfg, base_env, -1)
            tape_lh = left.stop_recording()
            restore_pre()
            right.start_recording()
            _grasp_handle(right, cfg, base_env, +1)
            tape_rh = right.stop_recording()
        finally:
            if rec is not None:
                rec.save_trajectory, rec._save_video = saved
            left.frame_cb = right.frame_cb = cb

        restore_pre()
        n_h = max(len(tape_lh), len(tape_rh))
        _coplay(env, _pad(tape_lh, n_h), _pad(tape_rh, n_h), step_cb=cb)
        gripper_states[0] = gripper_states[1] = CLOSED
        b_handle = step_count[0]
        last = _dual_lift(env, left, right, gripper_states, cfg, base_env)
    except RuntimeError as exc:
        if debug:
            print(f"  motion planning failed: {exc}")
        return -1

    total = step_count[0]
    # Decomposed sub-step labels, each tied to the frame range of its actual sub-action so the
    # on-screen text changes as the arm moves on. Built per ROLE, then mapped to left/right.
    lid_track = [
        # "lift the lid off the pot" covers the grab, lift AND the hold-aside, then flips straight
        # to "put the lid back on" -- no separate hold label.
        {"label": LID_LIFT, "start": 0, "end": int(b_drop)},
        {"label": LID_REPLACE, "start": int(b_drop), "end": int(phase1_end)},
        {"label": HANDLE_GRASP, "start": int(phase1_end), "end": int(b_handle)},
        {"label": POT_LIFT, "start": int(b_handle), "end": int(total)},
    ]
    carrot_track = [
        {"label": CARROT_IN, "start": 0, "end": int(phase1_end)},
        {"label": HANDLE_GRASP, "start": int(phase1_end), "end": int(b_handle)},
        {"label": POT_LIFT, "start": int(b_handle), "end": int(total)},
    ]
    base_env._subgoal_segments = {
        "phase1_end": int(phase1_end), "end": int(total),
        "left": lid_track if lid_idx == 0 else carrot_track,
        "right": lid_track if lid_idx == 1 else carrot_track,
    }
    if debug:
        ev = base_env.evaluate()
        print(f"  [done] success={bool(ev['success'][0])} "
              f"opened_once={bool(ev['lid_opened_once'][0])} "
              f"meat_inside={bool(ev['meat_inside_pot'][0])} "
              f"lid_closed={bool(ev['lid_closed'][0])} "
              f"both_grasp={bool(ev['both_grasping_pot'][0])} "
              f"on_target={bool(ev['pot_on_target'][0])} lifted={bool(ev['pot_lifted'][0])}")
    return last
