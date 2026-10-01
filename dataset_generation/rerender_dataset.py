"""Re-render the 3 camera views for a state-only TwoRobotFoodServe dataset.

Replays each recorded trajectory by EXACT env states (`set_state_dict`, which is
deterministic and reproduces the verified success -- unlike action replay) and
captures the 3 observation cameras, writing a new h5 with per-step images, actions,
env_states, and the per-arm subgoal instructions.

(We use this instead of `mani_skill.trajectory.replay_trajectory --use-env-states`
because that path errors on the ReplicaCAD background actor names for this env.)

Usage:
  python rerender_dataset.py --in <state>.h5 --out <rgb>.h5 [--instructions <json>]
"""
import argparse
import json
import os.path as osp

import h5py
import numpy as np
import gymnasium as gym
import contextlib
import io

import mani_skill.envs  # noqa: F401
import mani_skill.envs.tasks.tabletop.two_robot_food_serve as _fsmod
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.trajectory.utils import dict_to_list_of_dicts

ENV_ID = "TwoRobotFoodServeReplicaCAD-v1"
CAMS = ["global_camera", "panda_wristcam-0-hand_camera", "panda_wristcam-1-hand_camera"]


def _limit_to_global_camera():
    """Exclude visualization-only static cameras from dataset re-rendering."""
    cls = _fsmod.TwoRobotFoodServeReplicaCADEnv
    orig = cls._default_sensor_configs.fget

    def only_global(self):
        return [c for c in orig(self) if c.uid == "global_camera"]

    cls._default_sensor_configs = property(only_global)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="state-only h5")
    p.add_argument("--out", required=True, help="output rgb h5")
    p.add_argument("--instructions", default=None,
                   help="instructions json to embed per-traj (optional)")
    p.add_argument("--traj", default=None,
                   help="optional single trajectory id to re-render, e.g. traj_0")
    p.add_argument("--shard", type=int, default=None,
                   help="optional zero-based trajectory shard index")
    p.add_argument("--nshard", type=int, default=None,
                   help="number of trajectory shards (required with --shard)")
    args = p.parse_args()

    instr = {}
    if args.instructions and osp.exists(args.instructions):
        instr = json.load(open(args.instructions))
        instr = instr.get("trajectories", instr)  # accept the nested sidecar layout

    _limit_to_global_camera()
    env = gym.make(ENV_ID, obs_mode="rgb", control_mode="pd_joint_pos",
                   robot_init_qpos_noise=0, sim_backend="physx_cpu",
                   render_mode="rgb_array")
    env = FlattenActionSpaceWrapper(env)
    be = env.unwrapped
    env.reset(seed=0)  # build the scene once; states are set explicitly below

    fin = h5py.File(args.inp, "r")
    fout = h5py.File(args.out, "w")
    tids = sorted(fin.keys(), key=lambda t: int(t.split("_")[1]))
    if args.traj is not None:
        if args.shard is not None or args.nshard is not None:
            raise ValueError("--traj cannot be combined with --shard/--nshard")
        if args.traj not in fin:
            raise KeyError(f"{args.traj!r} is not present in {args.inp}")
        tids = [args.traj]
    elif args.shard is not None or args.nshard is not None:
        if args.shard is None or args.nshard is None:
            raise ValueError("--shard and --nshard must be provided together")
        if args.nshard < 1 or not 0 <= args.shard < args.nshard:
            raise ValueError("require 0 <= --shard < --nshard")
        tids = [tid for index, tid in enumerate(tids) if index % args.nshard == args.shard]
    n_ok = 0
    for tid in tids:
        t = fin[tid]
        states = dict_to_list_of_dicts(t["env_states"])
        actions = t["actions"][:]
        frames = {c: [] for c in CAMS}
        for st in states:
            with contextlib.redirect_stdout(io.StringIO()):
                be.set_state_dict(st)
                obs = be.get_obs()
            for c in CAMS:
                frames[c].append(obs["sensor_data"][c]["rgb"][0].cpu().numpy().astype(np.uint8))
        # verify the replayed trajectory still ends in a full success
        final_info = be.evaluate()
        ok = bool(final_info["success"].item())
        g = fout.create_group(tid, track_order=True)
        og = g.create_group("obs", track_order=True).create_group("sensor_data", track_order=True)
        for c in CAMS:
            cg = og.create_group(c, track_order=True)
            cg.create_dataset("rgb", data=np.asarray(frames[c]),
                              compression="gzip", compression_opts=4)
        g.create_dataset("actions", data=np.asarray(actions))
        # copy env_states verbatim
        fin.copy(t["env_states"], g, "env_states")
        if tid in instr:
            g.attrs["instructions"] = json.dumps(instr[tid])
        g.attrs["full_success"] = ok
        n_ok += int(ok)
        print(f"{tid}: {len(states)} steps, success={ok}", flush=True)
    fin.close()
    fout.close()
    env.close()
    print(f"wrote {args.out}: {len(tids)} trajs, {n_ok} verified full successes", flush=True)


if __name__ == "__main__":
    main()
