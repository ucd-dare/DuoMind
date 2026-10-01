"""Motion-planning solver for TwoRobotPrepareSnackReplicaCAD-v1."""

from dataclasses import dataclass

from . import prepare_fruit


LABELS = (
    "put bread on brown plate",
    "put bread on white plate",
    "put bread on white plate",
    "put bread on brown plate",
)


@dataclass
class PrepareSnackConfig(prepare_fruit.PrepareFruitConfig):
    square_bread_grasp_above: float = 0.006
    orange_release_z: float = 0.010
    orange_settle: int = 25


def solve(env, seed=None, debug=False, vis=False, frame_cb=None, **kwargs):
    return prepare_fruit.solve(
        env,
        seed=seed,
        debug=debug,
        vis=vis,
        frame_cb=frame_cb,
        config=PrepareSnackConfig(),
        labels=LABELS,
        **kwargs,
    )
