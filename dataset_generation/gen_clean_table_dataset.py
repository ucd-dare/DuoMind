"""Generate a dataset of N successful TwoRobotCleanTable trajectories with the
THREE training cameras (2 wrist + global) and the per-arm natural-language
instructions saved as a sidecar JSON.

What it produces (under demos/<ENV_ID>/dataset/):
  * <name>.h5 / <name>.json  -- ManiSkill trajectories. With an rgb/rgbd obs_mode
    each step carries ONLY the 3 kept cameras (a CameraFilter wrapper drops the
    other scene cameras so the file stays small); with obs_mode=none only
    actions + env_states are stored (re-render images later from env_states).
  * <name>_instructions.json -- per-trajectory, per-arm subgoal instructions:
        { "traj_0": {
            "phase1_end": int, "end": int,
            "instructions": {"left": "...", "right": "..."},
            "left":  [{"label","start","end"}, ...],
            "right": [{"label","start","end"}, ...] },
          ... }

Usage:
  python gen_clean_table_dataset.py --env-id TwoRobotCleanTableReplicaCAD-v1 \
      -n 50 --obs-mode rgb --start-seed 0 [--save-video]

Tip: obs_mode=none is small + fast (recommended); regenerate the 3 camera streams
on demand with dataset_generation/rerender_dataset_cleantable.py or replay_trajectory --use-env-states.
"""
import argparse
import json
import os.path as osp
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import robopoly.envs  # noqa: F401
from robopoly.examples.motionplanning.two_robot.solutions import solveTwoRobotCleanTable
from robopoly.utils.wrappers.flatten import FlattenActionSpaceWrapper
from robopoly.utils.wrappers.record import RecordEpisode

KEEP_CAMERAS = (
    "global_camera",
    "panda_wristcam-0-hand_camera",
    "panda_wristcam-1-hand_camera",
)


class CameraFilter(gym.ObservationWrapper):
    """Keep only ``KEEP_CAMERAS`` in the visual obs so the saved dataset carries
    just the 3 training views (drops the other scene cameras). No-op for non-visual
    obs modes (then ``sensor_data`` is absent)."""

    def __init__(self, env, keep=KEEP_CAMERAS):
        super().__init__(env)
        self.keep = set(keep)

    def observation(self, obs):
        for key in ("sensor_data", "sensor_param"):
            if isinstance(obs.get(key), dict):
                obs[key] = {k: v for k, v in obs[key].items() if k in self.keep}
        return obs


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="TwoRobotCleanTableReplicaCAD-v1")
    p.add_argument("-n", "--num-traj", type=int, default=50)
    p.add_argument("--obs-mode", default="rgb", help="rgb | rgbd | state | none")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--traj-name", default="clean_table_dataset")
    p.add_argument("--record-dir", default="demos")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("-b", "--sim-backend", default="auto")
    p.add_argument("--max-seeds", type=int, default=10000)
    return p.parse_args(args)


def main(args):
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        robot_init_qpos_noise=0,
        render_mode="rgb_array",
        reward_mode="sparse",
        sim_backend=args.sim_backend,
    )
    env = FlattenActionSpaceWrapper(env)
    env = CameraFilter(env)
    out_dir = osp.join(args.record_dir, args.env_id, "dataset")
    env = RecordEpisode(
        env,
        output_dir=out_dir,
        trajectory_name=args.traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="two-robot clean-table cooperative MP demo",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    base_env = env.unwrapped
    print(f"Saving dataset to {env._h5_file.filename}")

    seed = args.start_seed
    passed = 0
    instructions = {}
    pbar = tqdm(range(args.num_traj))
    while passed < args.num_traj and (seed - args.start_seed) < args.max_seeds:
        try:
            res = solveTwoRobotCleanTable(env, seed=seed, debug=False, vis=False)
        except Exception as exc:  # noqa: BLE001
            print(f"seed {seed}: solver error {exc}")
            res = -1
        success = res != -1 and bool(res[-1]["success"].item())
        if not success:
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            seed += 1
            continue

        seg = dict(base_env._subgoal_segments)
        instructions[f"traj_{passed}"] = {
            "seed": int(seed),
            "phase1_end": seg["phase1_end"],
            "end": seg["end"],
            "instructions": {"left": seg["left"][0]["label"], "right": seg["right"][0]["label"]},
            "left": seg["left"],
            "right": seg["right"],
        }
        env.flush_trajectory()
        if args.save_video:
            env.flush_video()
        passed += 1
        seed += 1
        pbar.update(1)

    env.close()
    instr_path = osp.join(out_dir, f"{args.traj_name}_instructions.json")
    with open(instr_path, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"\nWrote {passed} successful trajectories.")
    print(f"Per-arm instructions: {instr_path}")
    print(f"Kept cameras: {list(KEEP_CAMERAS)}")


if __name__ == "__main__":
    main(parse_args())
