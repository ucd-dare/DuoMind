"""Re-render the 3 camera views for a state-only TwoRobotPrepareFruit dataset.

Replays each recorded trajectory by EXACT env states (`set_state_dict`, which is
deterministic and reproduces the verified success -- unlike action replay) and
captures the 3 observation cameras, writing a new h5 with per-step images, actions,
env_states, and the per-arm subgoal instructions -- exactly like the other two-robot
dataset re-render scripts (FoodServe / CookPot / BottleExchange / CleanTable).

Success is state-computable for this task (evaluate() checks fruit-on-plate + both
arms static), so we just read be.evaluate()["success"] at the final replayed state.

Usage:
  python rerender_dataset_preparefruit.py --in <state>.h5 --out <rgb>.h5 [--instructions <json>]
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
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.trajectory.utils import dict_to_list_of_dicts

DEFAULT_ENV_ID = "TwoRobotPrepareSnackReplicaCAD-v1"
CAMS = ["global_camera", "panda_wristcam-0-hand_camera", "panda_wristcam-1-hand_camera"]

# Pin all three resolutions explicitly so this replay remains compatible with the
# shared two-robot dataset format even if an environment default changes later.
SENSOR_CONFIGS = {
    "global_camera": dict(width=512, height=384),
    "panda_wristcam-0-hand_camera": dict(width=128, height=128),
    "panda_wristcam-1-hand_camera": dict(width=128, height=128),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="state-only h5")
    p.add_argument("--out", required=True, help="output rgb h5")
    p.add_argument("--env-id", default=DEFAULT_ENV_ID,
                   help="two-robot environment id (default: Prepare Food)")
    p.add_argument("--instructions", default=None,
                   help="instructions json to embed per-traj (optional)")
    args = p.parse_args()

    instr = {}
    if args.instructions and osp.exists(args.instructions):
        instr = json.load(open(args.instructions))
        instr = instr.get("trajectories", instr)  # accept the nested sidecar layout

    env = gym.make(args.env_id, obs_mode="rgb", control_mode="pd_joint_pos",
                   robot_init_qpos_noise=0, sim_backend="physx_cpu",
                   render_mode="rgb_array", sensor_configs=SENSOR_CONFIGS)
    env = FlattenActionSpaceWrapper(env)
    be = env.unwrapped
    env.reset(seed=0)  # build the scene once; states are set explicitly below

    fin = h5py.File(args.inp, "r")
    fout = h5py.File(args.out, "w")
    tids = sorted(fin.keys(), key=lambda t: int(t.split("_")[1]))
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
        # verify the replayed trajectory still ends in a full success (final state)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = bool(be.evaluate()["success"].item())
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
