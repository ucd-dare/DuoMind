"""Re-render the 3 camera views for a state-only TwoRobotCleanTable dataset.

Replays each recorded trajectory by EXACT env states (`set_state_dict`, which is
deterministic and reproduces the verified success) and captures the 3 observation
cameras, writing a new h5 with per-step images, actions, env_states, and the
per-arm subgoal instructions.

Usage:
  python rerender_dataset_cleantable.py --in <state>.h5 --out <rgb>.h5 \
      [--instructions <json>] [--env-id ...]
"""
import argparse
import contextlib
import io
import json
import os.path as osp

import gymnasium as gym
import h5py
import numpy as np

import robopoly.envs  # noqa: F401
import robopoly.envs.tasks.tabletop.two_robot_clean_table as _ctmod
from robopoly.trajectory.utils import dict_to_list_of_dicts
from robopoly.utils.wrappers.flatten import FlattenActionSpaceWrapper

CAMS = ["global_camera", "panda_wristcam-0-hand_camera", "panda_wristcam-1-hand_camera"]


def _limit_to_global_camera():
    """Drop the 6 unused ReplicaCAD scene cameras from the env so each replay step
    renders only the global camera (the 2 wrist cameras come from the panda_wristcam
    robot and stay). Rendering 3 cams instead of 9 is ~3x faster -- the only cost in
    this otherwise render-bound re-render."""
    cls = _ctmod.TwoRobotCleanTableReplicaCADEnv
    orig = cls._default_sensor_configs.fget

    def only_global(self):
        return [c for c in orig(self) if c.uid == "global_camera"]

    cls._default_sensor_configs = property(only_global)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="state-only h5")
    p.add_argument("--out", required=True, help="output rgb h5")
    p.add_argument("--instructions", default=None)
    p.add_argument("--env-id", default="TwoRobotCleanTableReplicaCAD-v1")
    p.add_argument("--shard", type=int, default=0, help="this shard index")
    p.add_argument("--nshard", type=int, default=1, help="total shards (round-robin "
                   "split of trajs for parallel rendering)")
    args = p.parse_args()

    instr = {}
    if args.instructions and osp.exists(args.instructions):
        instr = json.load(open(args.instructions))
        instr = instr.get("trajectories", instr)  # accept the nested sidecar layout

    _limit_to_global_camera()
    env = gym.make(args.env_id, obs_mode="rgb", control_mode="pd_joint_pos",
                   robot_init_qpos_noise=0, sim_backend="physx_cpu",
                   render_mode="rgb_array")
    env = FlattenActionSpaceWrapper(env)
    be = env.unwrapped
    env.reset(seed=0)  # build the scene once; states are set explicitly below

    fin = h5py.File(args.inp, "r")
    fout = h5py.File(args.out, "w")
    tids = sorted(fin.keys(), key=lambda t: int(t.split("_")[1]))
    tids = [t for j, t in enumerate(tids) if j % args.nshard == args.shard]
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
        # verify the replayed trajectory still ends with every cube in the basket
        ok = bool(be._items_in_basket()[0].all().item())
        g = fout.create_group(tid, track_order=True)
        og = g.create_group("obs", track_order=True).create_group("sensor_data", track_order=True)
        for c in CAMS:
            cg = og.create_group(c, track_order=True)
            cg.create_dataset("rgb", data=np.asarray(frames[c]),
                              compression="gzip", compression_opts=4)
        g.create_dataset("actions", data=np.asarray(actions))
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
