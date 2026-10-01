"""Keypoint solver for ``TwoRobotBreadExchangeReplicaCAD-v1``."""

from dataclasses import dataclass

from . import bottle_exchange


LABELS = (
    "pick up bread and place in middle",
    "pick up bread and place in box",
    "pick up bread and place in box",
    "pick up bread and place in middle",
)


@dataclass
class BreadExchangeConfig(bottle_exchange.BottleExchangeConfig):
    # Match Prepare Snack's gentle thin-bread grasp.
    mug_pregrasp: float = 0.08
    close_steps: int = 90
    bread_grasp_above: float = 0.006
    max_joint_step: float = 0.020
    # Move to the above-box transit pose and open there. The flat bread can fall
    # into the box cleanly and does not need the cup solver's downward insertion.
    box_release_from_above: bool = True
    box_retreat_after_release: bool = False
    box_release_wait: int = 8
    box_post_release_wait: int = 28


def solve(env, seed=None, debug=False, vis=False, frame_cb=None, **kwargs):
    return bottle_exchange.solve(
        env,
        seed=seed,
        debug=debug,
        vis=vis,
        frame_cb=frame_cb,
        config=BreadExchangeConfig(),
        labels=LABELS,
        **kwargs,
    )


def breads_face_up(base_env, max_deg=8.0):
    return bottle_exchange.mugs_upright(base_env, max_deg=max_deg)


def left_second_pick_wrist_motion(base_env):
    return bottle_exchange.left_second_pick_wrist_motion(base_env)
