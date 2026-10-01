"""Unified two-robot dataset generator (state-only .h5 + .instructions.json sidecar).

ONE generator for every two-robot dataset task. Pick a task with ``--task``; each
task's env id, motion-plan solver, constant high-level instruction, any extra
success gates (e.g. bottles-upright / wrist-winding) and any extra recorded fields
live in the ``TASKS`` registry below -- so a per-task tweak happens in a single
place instead of in five near-duplicate scripts.

Each kept trajectory is a FULL success (the env's ``success`` flag, plus any task
specific gate). For every kept demo we record the per-arm subgoal instructions (with
step boundaries) *and* the constant high-level instruction to a sidecar JSON:

  {
    "task_name": "food_serve",
    "high_level_instruction": "<constant, whole-task>",
    "trajectories": {
      "traj_0": {
        "seed": 12, "phase1_end": 324, "end": 858,
        "low_level_instructions": {
          "left":  [{"label": ..., "start": 0, "end": 324}, ...],
          "right": [{"label": ..., "start": 0, "end": 324}, ...]
        },
        "high_level_instruction": "<constant, whole-task>"
      }
    }
  }

This writes a STATE-only ``.h5`` (actions + env_states, fast, no images). Add the
3-camera RGB observations afterwards with the task's rerender step (or the generic
``robopoly.trajectory.replay_trajectory --use-env-states --obs-mode rgb``).

Usage:
  python dataset_generation/generate_dataset_unified.py --task food_serve -n 50
  python dataset_generation/generate_dataset_unified.py --task hang_bag -n 2
"""
import argparse
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Callable, Optional

# Keep each worker single-threaded so many parallel shards don't oversubscribe the
# CPU (mplib/BLAS/torch otherwise each spawn many threads). Must be set before the
# heavy imports below.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

torch.set_num_threads(1)

from robopoly.examples.motionplanning.two_robot.solutions import (
    solveTwoRobotFoodServe,
    solveTwoRobotHangBag,
    solveTwoRobotBreadExchange,
    solveTwoRobotCleanTable,
    solveTwoRobotCookPot,
    solveTwoRobotPrepareSnack,
    solveTwoRobotPutObjectCabinet,
)
from robopoly.examples.motionplanning.two_robot.solutions.bread_exchange import (
    breads_face_up,
    left_second_pick_wrist_motion as bread_left_second_pick_wrist_motion,
)
from robopoly.examples.motionplanning.two_robot.solutions.clean_table import (
    cube_operation_max_wrist_path,
    cube_operation_max_wrist_net_rotation,
    grasp_approach_max_joint_winding,
    left_place_counterclockwise_wrist_path,
    wrist_grab_max_path,
)
from robopoly.utils.wrappers.flatten import FlattenActionSpaceWrapper
from robopoly.utils.wrappers.record import RecordEpisode


# --------------------------------------------------------------------------- #
# Per-task validation gates. Each takes (base_env, args) and returns
# (ok: bool, extra_fields: dict) -- extra_fields are merged into that traj's
# instruction record (e.g. the CleanTable wrist winding).
# --------------------------------------------------------------------------- #
def _validate_bread_exchange(base_env, args):
    del args
    path_deg, net_deg = bread_left_second_pick_wrist_motion(base_env)
    clean_wrist = path_deg <= 45.0 and net_deg <= 45.0
    return breads_face_up(base_env) and clean_wrist, {
        "left_second_pick_wrist_path_deg": round(path_deg, 1),
        "left_second_pick_wrist_net_deg": round(net_deg, 1),
        "breads_face_up": breads_face_up(base_env),
    }


def _validate_prepare_food(base_env, args):
    """Reject recovered, dropped, sliding, or side-resting food trajectories."""
    del args
    info = base_env.evaluate()
    had_retry = bool(getattr(base_env, "_had_object_retry", True))
    breads_face_up = bool(info["breads_face_up"][0].item())
    food_static = bool(info["food_static"][0].item())
    all_plates_ready = bool(info["all_plates_ready"][0].item())
    max_bread_grasp_jump = float(
        getattr(base_env, "_max_bread_grasp_step_displacement", float("inf"))
    )
    ok = (
        all_plates_ready
        and breads_face_up
        and food_static
        and not had_retry
        and max_bread_grasp_jump <= 0.001
    )
    return ok, {
        "breads_face_up": breads_face_up,
        "food_static": food_static,
        "had_object_retry_or_drop": had_retry,
        "max_bread_grasp_frame_displacement_m": round(max_bread_grasp_jump, 5),
    }


def _validate_food_serve(base_env, args):
    """Reject successful demos with gratuitous wrist motion around a grasp."""
    left_tape, right_tape, _, _ = base_env._coplay_tapes
    max_pregrasp_reversal = 0.0
    max_carry_path = 0.0
    max_carry_net = 0.0
    max_carry_reversal = 0.0
    measurements = {}
    for side, tape in (("left", left_tape), ("right", right_tape)):
        actions = np.asarray(tape)
        closed = actions[:, 7] < 0
        starts = np.flatnonzero(closed & np.r_[True, ~closed[:-1]])
        ends = np.flatnonzero(closed & np.r_[~closed[1:], True])
        previous_end = 0
        for grasp_index, (start, end) in enumerate(zip(starts, ends), 1):
            wrist = np.unwrap(actions[previous_end : start + 1, 6])
            path = float(np.abs(np.diff(wrist)).sum())
            net = float(abs(wrist[-1] - wrist[0]))
            reversal_deg = float(np.degrees(max(0.0, path - net)))
            measurements[f"{side}_grasp{grasp_index}_wrist_reversal_deg"] = round(
                reversal_deg, 1
            )
            max_pregrasp_reversal = max(max_pregrasp_reversal, reversal_deg)

            carry_wrist = np.unwrap(actions[start : end + 1, 6])
            carry_path = float(np.degrees(np.abs(np.diff(carry_wrist)).sum()))
            carry_net = float(np.degrees(abs(carry_wrist[-1] - carry_wrist[0])))
            carry_reversal = max(0.0, carry_path - carry_net)
            measurements[f"{side}_grasp{grasp_index}_carry_wrist_path_deg"] = round(
                carry_path, 1
            )
            measurements[f"{side}_grasp{grasp_index}_carry_wrist_net_deg"] = round(
                carry_net, 1
            )
            measurements[f"{side}_grasp{grasp_index}_carry_wrist_reversal_deg"] = round(
                carry_reversal, 1
            )
            max_carry_path = max(max_carry_path, carry_path)
            max_carry_net = max(max_carry_net, carry_net)
            max_carry_reversal = max(max_carry_reversal, carry_reversal)
            previous_end = int(end) + 1
    measurements["max_pregrasp_wrist_reversal_deg"] = round(
        max_pregrasp_reversal, 1
    )
    measurements["max_carry_wrist_path_deg"] = round(max_carry_path, 1)
    measurements["max_carry_wrist_net_deg"] = round(max_carry_net, 1)
    measurements["max_carry_wrist_reversal_deg"] = round(max_carry_reversal, 1)
    ok = (
        max_pregrasp_reversal <= args.max_pregrasp_wrist_reversal
        and max_carry_path <= args.max_carry_wrist_path
        and max_carry_net <= args.max_carry_wrist_net_rotation
        and max_carry_reversal <= args.max_carry_wrist_reversal
    )
    return ok, measurements


def _validate_clean_table(base_env, args):
    # Reject successes with either a spinning wrist or a large whole-arm RRT loop
    # during a color-specific cube approach. Record both quality measurements.
    winding = wrist_grab_max_path(base_env._action_tape)
    joint_winding = grasp_approach_max_joint_winding(
        base_env._action_tape, base_env._subgoal_segments
    )
    wrist_net_rotation = cube_operation_max_wrist_net_rotation(
        base_env._action_tape, base_env._subgoal_segments
    )
    wrist_path = cube_operation_max_wrist_path(
        base_env._action_tape, base_env._subgoal_segments
    )
    left_place_ccw = left_place_counterclockwise_wrist_path(base_env._action_tape)
    basket_contact_force = float(
        getattr(base_env, "_max_robot_basket_contact_force", float("inf"))
    )
    background_displacement = float(
        getattr(base_env, "_max_background_prop_displacement", float("inf"))
    )
    background_speed = float(
        getattr(base_env, "_max_background_prop_speed", float("inf"))
    )
    background_count = int(getattr(base_env, "_background_prop_count", 0))
    ok = (
        winding <= args.max_winding
        and joint_winding <= args.max_joint_winding
        and wrist_net_rotation <= args.max_wrist_net_rotation
        and wrist_path <= args.max_wrist_path
        and left_place_ccw <= args.max_left_place_ccw_rotation
        and basket_contact_force <= args.max_robot_basket_force
        and background_count > 0
        and background_displacement <= args.max_background_prop_displacement
        and background_speed <= args.max_background_prop_speed
    )
    return ok, {
        "max_wrist_winding_deg": round(float(winding), 1),
        "max_grasp_joint_winding_deg": round(float(joint_winding), 1),
        "max_cube_wrist_net_rotation_deg": round(float(wrist_net_rotation), 1),
        "max_cube_wrist_path_deg": round(float(wrist_path), 1),
        "max_left_place_ccw_wrist_path_deg": round(float(left_place_ccw), 1),
        "max_robot_basket_contact_force_n": round(basket_contact_force, 3),
        "restored_loose_background_props": background_count,
        "max_background_prop_displacement_m": round(background_displacement, 4),
        "max_background_prop_speed_mps": round(background_speed, 4),
        "max_background_prop_displacement_name": str(
            getattr(base_env, "_max_background_prop_displacement_name", "")
        ),
        "max_background_prop_speed_name": str(
            getattr(base_env, "_max_background_prop_speed_name", "")
        ),
    }


@dataclass
class TaskSpec:
    env_id: str                          # default env; overridable with --env-id
    high_level_instruction: str          # constant whole-task instruction
    solve: Callable                      # solve(env, seed) -> res (or -1)
    source_desc: str                     # RecordEpisode provenance string
    out: str                             # default trajectory-name (file stem)
    attempts: int = 1                    # re-plan the SAME seed up to N times
    validate: Optional[Callable] = None  # (base_env, args) -> (ok, extra_fields)


# --------------------------------------------------------------------------- #
# Task registry -- the single place to add a task or tweak an existing one.
# --------------------------------------------------------------------------- #
TASKS = {
    "food_serve": TaskSpec(
        env_id="TwoRobotFoodServeReplicaCAD-v1",
        high_level_instruction=(
            "Right arm pass the bread to the middle and Left arm put both mug "
            "and bread onto the tray."
        ),
        solve=lambda env, seed: solveTwoRobotFoodServe(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot fixed-tray mug-and-bread serving solver (full success)",
        out="food_serve_dataset",
        attempts=4,
        validate=_validate_food_serve,
    ),
    "clean_table": TaskSpec(
        env_id="TwoRobotCleanTableReplicaCAD-v1",
        high_level_instruction="Pick up all the cubes and put them into the box",
        solve=lambda env, seed: solveTwoRobotCleanTable(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot clean-table cooperative solver (clean full success)",
        out="clean_table_dataset",
        validate=_validate_clean_table,
    ),
    "cook_pot": TaskSpec(
        env_id="TwoRobotCookPotReplicaCAD-v1",
        high_level_instruction=(
            "Open the lid and put carrot in. Then move the pot to the target together."
        ),
        solve=lambda env, seed: solveTwoRobotCookPot(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot cook-pot keypoint solver (full success)",
        out="cook_pot_dataset",
        attempts=4,  # RRT/IK is randomized; retry the same (fixed) scene a few times
    ),
    "exchange_bread": TaskSpec(
        env_id="TwoRobotBreadExchangeReplicaCAD-v1",
        high_level_instruction="Put each bread into the box on the other side.",
        solve=lambda env, seed: solveTwoRobotBreadExchange(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot square-bread exchange keypoint solver (full success)",
        out="exchange_bread_dataset",
        attempts=4,
        validate=_validate_bread_exchange,
    ),
    "put_object_cabinet": TaskSpec(
        env_id="TwoRobotPutObjectCabinetReplicaCAD-v1",
        high_level_instruction=(
            "Open the drawer and put the cube inside. Then close the drawer."
        ),
        solve=lambda env, seed: solveTwoRobotPutObjectCabinet(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot put-object-in-cabinet-drawer co-play solver (full success)",
        out="put_object_cabinet_dataset",
        attempts=2,  # RRT/IK randomized; retry the same scene a couple times
    ),
    "prepare_snack": TaskSpec(
        env_id="TwoRobotPrepareSnackReplicaCAD-v1",
        high_level_instruction="Put two square breads onto each plate.",
        solve=lambda env, seed: solveTwoRobotPrepareSnack(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot prepare-snack four-square-bread solver (full success)",
        out="prepare_snack_dataset",
        attempts=4,
        validate=_validate_prepare_food,
    ),
    "hang_bag": TaskSpec(
        env_id="TwoRobotHangBagReplicaCAD-v1",
        high_level_instruction=(
            "Pass the bag from the left arm to the right arm and hang it on the hook."
        ),
        solve=lambda env, seed: solveTwoRobotHangBag(
            env, seed=seed, debug=False, vis=False),
        source_desc="two-robot hang-bag handoff solver (full success)",
        out="hang_bag_dataset",
    ),
}


def parse_args(args=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS),
                   help="which two-robot dataset task to generate")
    p.add_argument("--env-id", type=str, default=None,
                   help="override the task's default env id (e.g. TwoRobotCookPot-v1)")
    p.add_argument("-n", "--num-traj", type=int, default=50,
                   help="number of FULL-success trajectories to collect")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=100000,
                   help="give up after trying this many seeds")
    p.add_argument("--attempts", type=int, default=None,
                   help="re-plan each seed up to N times (default: per-task)")
    p.add_argument("--max-winding", type=float, default=30.0,
                   help="clean_table only: reject a success if any grasp-approach "
                        "wrist winding (deg) exceeds this")
    p.add_argument("--max-joint-winding", type=float, default=45.0,
                   help="clean_table only: reject a success if any arm joint's "
                        "grasp-approach backtracking (deg) exceeds this")
    p.add_argument("--max-wrist-net-rotation", type=float, default=180.0,
                   help="clean_table only: reject a success if joint 7 makes more "
                        "than this absolute net turn (deg) during one cube operation")
    p.add_argument("--max-wrist-path", type=float, default=240.0,
                   help="clean_table only: reject a success if joint 7 travels more "
                        "than this total angle (deg) during one cube operation")
    p.add_argument("--max-left-place-ccw-rotation", type=float, default=5.0,
                   help="clean_table only: reject a success if the left wrist makes "
                        "more than this much positive/counter-clockwise joint-7 "
                        "travel while carrying a cube to its placement")
    p.add_argument("--max-pregrasp-wrist-reversal", type=float, default=10.0,
                   help="food_serve only: reject a success if joint 7 reverses by "
                        "more than this excess path angle (deg) before a grasp")
    p.add_argument("--max-carry-wrist-path", type=float, default=120.0,
                   help="food_serve only: maximum joint-7 total travel (deg) while "
                        "holding an object")
    p.add_argument("--max-carry-wrist-net-rotation", type=float, default=120.0,
                   help="food_serve only: maximum joint-7 net rotation (deg) while "
                        "holding an object")
    p.add_argument("--max-carry-wrist-reversal", type=float, default=15.0,
                   help="food_serve only: maximum joint-7 excess path over net "
                        "rotation (deg) while holding an object")
    p.add_argument("--max-robot-basket-force", type=float, default=1.0,
                   help="clean_table only: reject a success if either robot contacts "
                        "the basket above this force in newtons")
    p.add_argument("--max-background-prop-displacement", type=float, default=0.30,
                   help="clean_table only: reject a success if any restored loose "
                        "ReplicaCAD prop moves farther than this many metres")
    p.add_argument("--max-background-prop-speed", type=float, default=2.0,
                   help="clean_table only: reject a success if any restored loose "
                        "ReplicaCAD prop exceeds this speed in metres/second")
    p.add_argument("--out", type=str, default=None,
                   help="trajectory name (file stem); default: per-task")
    p.add_argument("--record-dir", type=str, default=None,
                   help="default: demos/<env_id>/dataset")
    p.add_argument("--save-video", action="store_true",
                   help="also save the render_camera video per kept demo")
    p.add_argument("--stop-file", type=str, default=None,
                   help="if this file appears, finish the current seed and exit "
                        "cleanly (used to stop a fleet of shards once the global "
                        "success count is reached)")
    return p.parse_args(args)


def main(args):
    spec = TASKS[args.task]
    env_id = args.env_id or spec.env_id
    out = args.out or spec.out
    record_dir = args.record_dir or f"demos/{env_id}/dataset"
    attempts = args.attempts if args.attempts is not None else spec.attempts

    env = gym.make(
        env_id, obs_mode="none", control_mode="pd_joint_pos",
        robot_init_qpos_noise=0, render_mode="rgb_array", sim_backend="physx_cpu",
    )
    env = FlattenActionSpaceWrapper(env)
    env = RecordEpisode(
        env,
        output_dir=record_dir,
        trajectory_name=out,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc=spec.source_desc,
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    base_env = env.unwrapped
    instr_path = osp.join(record_dir, out + ".instructions.json")

    print(f"[{args.task}] generating {args.num_traj} full-success demos "
          f"-> {env._h5_file.filename}", flush=True)
    instructions = {}
    seed = args.start_seed
    passed = 0
    tried = 0
    pbar = tqdm(total=args.num_traj)
    while passed < args.num_traj and tried < args.max_seeds:
        if args.stop_file is not None and osp.exists(args.stop_file):
            print(f"stop-file seen; exiting with {passed} demos", flush=True)
            break

        # A seed's scene is fixed but the motion planner (RRT/IK) is randomized, so a
        # borderline pose may fail only some of the time -- retry the SAME seed up to
        # `attempts` times before moving on (attempts=1 for deterministic tasks).
        success = False
        res = -1
        extra_fields = {}
        for _ in range(attempts):
            tried += 1
            try:
                res = spec.solve(env, seed)
            except Exception as exc:
                res = -1
                print(f"seed {seed}: motion-plan error: {str(exc)[:60]}", flush=True)
            env_success = (res != -1) and bool(res[-1]["success"].item())
            success = env_success
            if env_success and spec.validate is not None:
                success, extra_fields = spec.validate(base_env, args)
                if not success:
                    print(
                        f"seed {seed}: quality-gate rejection "
                        f"{json.dumps(extra_fields, sort_keys=True)}",
                        flush=True,
                    )
                    # FoodServe's pre-grasp reversal is determined by the scene
                    # pose and Cartesian IK branch, so retrying the same rejected
                    # seed only reproduces the same wrist path.
                    if args.task == "food_serve":
                        break
            if success:
                break
            # A retry of the same scene starts a new recorded episode.  Discard
            # this failed/quality-rejected attempt immediately; otherwise
            # RecordEpisode's grow-only buffer prepends all earlier attempts to
            # the eventual successful trajectory, misaligning its instructions.
            if env._trajectory_buffer is not None:
                env.flush_trajectory(save=False)
            env._trajectory_buffer = None
            if args.save_video:
                env.flush_video(save=False)

        if not success:
            if env._trajectory_buffer is not None:
                env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            # IMPORTANT: RecordEpisode never shrinks its step buffer (only advances a
            # pointer), so it grows unbounded across seeds and eventually OOMs. Drop
            # it after every seed so each episode starts fresh.
            env._trajectory_buffer = None
            seed += 1
            continue

        env.flush_trajectory(save=True)
        env._h5_file.flush()  # persist so an external kill leaves a readable file
        env._trajectory_buffer = None  # free the accumulated step buffer (see above)
        if args.save_video:
            env.flush_video()
        traj_id = f"traj_{env._episode_id}"
        seg = base_env._subgoal_segments
        instructions[traj_id] = {
            "seed": seed,
            **extra_fields,
            "phase1_end": seg["phase1_end"],
            "end": seg["end"],
            **({"right_cube_order": seg["right_cube_order"]}
               if "right_cube_order" in seg else {}),
            "low_level_instructions": {"left": seg["left"], "right": seg["right"]},
            "high_level_instruction": spec.high_level_instruction,
        }
        with open(instr_path, "w") as f:
            json.dump({"task_name": args.task,
                       "high_level_instruction": spec.high_level_instruction,
                       "trajectories": instructions}, f, indent=2)
        passed += 1
        seed += 1
        pbar.update(1)
        pbar.set_postfix(dict(seed=seed, tried=tried, rate=round(passed / tried, 3)))

    env.close()
    print(f"[{args.task}] done: {passed} demos, {tried} seeds tried. "
          f"Instructions -> {instr_path}", flush=True)
    if passed != args.num_traj and args.stop_file is None:
        raise RuntimeError(
            f"generated {passed}/{args.num_traj} requested trajectories before "
            f"exhausting {args.max_seeds} attempts"
        )


if __name__ == "__main__":
    main(parse_args())
