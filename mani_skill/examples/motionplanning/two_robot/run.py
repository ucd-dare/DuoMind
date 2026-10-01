import argparse
import os.path as osp
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from mani_skill.examples.motionplanning.two_robot import solutions
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

MP_SOLUTIONS = {
    "TwoRobotCleanTableReplicaCAD-v1": solutions.solveTwoRobotCleanTable,
    "TwoRobotCookPotReplicaCAD-v1": solutions.solveTwoRobotCookPot,
    "TwoRobotBreadExchangeReplicaCAD-v1": solutions.solveTwoRobotBreadExchange,
    "TwoRobotFoodServeReplicaCAD-v1": solutions.solveTwoRobotFoodServe,
    "TwoRobotHangBagReplicaCAD-v1": solutions.solveTwoRobotHangBag,
    "TwoRobotPrepareSnackReplicaCAD-v1": solutions.solveTwoRobotPrepareSnack,
    "TwoRobotPutObjectCabinetReplicaCAD-v1": solutions.solveTwoRobotPutObjectCabinet,
}



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="TwoRobotFoodServeReplicaCAD-v1",
        help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}",
    )
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-n", "--num-traj", type=int, default=1)
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First seed to try (seeds increment from here). Use distinct, well-"
        "separated start seeds across parallel workers to avoid duplicate demos.",
    )
    parser.add_argument("--only-count-success", action="store_true")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--traj-name", type=str)
    parser.add_argument("--shader", default="default", type=str)
    parser.add_argument("--record-dir", type=str, default="demos")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--include-tray-lift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After serving, grasp the tray edge and lift it level (stage 4). "
        "On by default; pass --no-include-tray-lift to stop after the bowl is served.",
    )
    return parser.parse_args(args)


def main(args):
    if args.env_id not in MP_SOLUTIONS:
        raise RuntimeError(
            f"No motion planning solution for {args.env_id}. Available options are {list(MP_SOLUTIONS.keys())}"
        )

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        robot_init_qpos_noise=0,
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )
    env = FlattenActionSpaceWrapper(env)

    traj_name = args.traj_name or time.strftime("%Y%m%d_%H%M%S")
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, args.env_id, "motionplanning"),
        trajectory_name=traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="experimental two-robot motion planning solution",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )

    solve = MP_SOLUTIONS[args.env_id]
    print(f"Motion Planning Running on {args.env_id}")
    print(f"Trajectory output: {env._h5_file.filename}")

    seed = args.start_seed
    passed = 0
    successes = []
    failed_motion_plans = 0
    pbar = tqdm(range(args.num_traj))
    while passed < args.num_traj:
        try:
            res = solve(
                env,
                seed=seed,
                debug=args.debug,
                vis=args.vis,
                include_tray_lift=args.include_tray_lift,
            )
        except Exception as exc:
            print(f"Cannot find valid solution because of an error: {exc}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = bool(res[-1]["success"].item())
        successes.append(success)

        if args.only_count_success and not success:
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            seed += 1
            continue

        env.flush_trajectory()
        if args.save_video:
            env.flush_video()
        passed += 1
        pbar.update(1)
        pbar.set_postfix(
            dict(
                success_rate=np.mean(successes),
                failed_motion_plan_rate=failed_motion_plans
                / (seed - args.start_seed + 1),
            )
        )
        seed += 1

    env.close()


if __name__ == "__main__":
    main(parse_args())
