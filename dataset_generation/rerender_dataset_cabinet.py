"""Re-render the 3 camera views for a state-only TwoRobotPutObjectCabinet dataset.

The cabinet solver uses CO-PLAY: it records each arm SOLO (stepping the real env)
then restores state and replays both together. RecordEpisode therefore captures the
throwaway solo passes (which sit at the FRONT of the buffer, before the co-play), so
the recorded trajectory has ``recorded - end`` extra leading steps. The real, video-
matching trajectory is the CONTIGUOUS LAST ``end`` steps (``end`` from the instruction
sidecar). We trim to those, then replay by EXACT env states (set_state_dict) and
capture the 3 cameras -- exactly like the other two-robot rerender scripts.

Success (now the cube resting in the closed LOWER drawer) is latched via
``drawer_opened_once``, so we reset that flag per trajectory and call ``evaluate()``
on every replayed step to re-latch it, then read the final ``success``.

Usage:
  python rerender_dataset_cabinet.py --in <state>.h5 --out <rgb>.h5 [--instructions <json>]
"""
import argparse
import contextlib
import io
import json
import os.path as osp

import h5py
import numpy as np
import gymnasium as gym

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.trajectory.utils import dict_to_list_of_dicts

ENV_ID = "TwoRobotPutObjectCabinetReplicaCAD-v1"
CAMS = ["global_camera", "panda_wristcam-0-hand_camera", "panda_wristcam-1-hand_camera"]
# Pin the 3-camera resolutions to the shared two-robot dataset standard (global 512x384,
# wrists 128x128) so the RGB h5 format matches the other tasks / the HF dataset.
SENSOR_CONFIGS = {
    "global_camera": dict(width=512, height=384),
    "panda_wristcam-0-hand_camera": dict(width=128, height=128),
    "panda_wristcam-1-hand_camera": dict(width=128, height=128),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="state-only h5")
    p.add_argument("--out", required=True, help="output rgb h5")
    p.add_argument("--instructions", default=None,
                   help="instructions json (needed for the per-traj real-step count `end`)")
    args = p.parse_args()

    instr = {}
    if args.instructions and osp.exists(args.instructions):
        instr = json.load(open(args.instructions))
        instr = instr.get("trajectories", instr)  # accept the nested sidecar layout

    env = gym.make(ENV_ID, obs_mode="rgb", control_mode="pd_joint_pos",
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
        # Trim the leading co-play solo passes: keep the contiguous LAST `end` real steps.
        end = int(instr[tid]["end"]) if tid in instr and "end" in instr[tid] else len(actions)
        end = min(end, len(states), len(actions))
        states = states[-end:]
        actions = actions[-end:]
        frames = {c: [] for c in CAMS}
        # reset the latched "was opened" flag so success reflects THIS trajectory
        if hasattr(be, "drawer_opened_once"):
            be.drawer_opened_once[:] = False
        ok = False
        for st in states:
            with contextlib.redirect_stdout(io.StringIO()):
                be.set_state_dict(st)
                obs = be.get_obs()
                info = be.evaluate()  # re-latch drawer_opened_once each step
            ok = bool(info["success"].item())
            for c in CAMS:
                frames[c].append(obs["sensor_data"][c]["rgb"][0].cpu().numpy().astype(np.uint8))
        g = fout.create_group(tid, track_order=True)
        og = g.create_group("obs", track_order=True).create_group("sensor_data", track_order=True)
        for c in CAMS:
            cg = og.create_group(c, track_order=True)
            cg.create_dataset("rgb", data=np.asarray(frames[c]),
                              compression="gzip", compression_opts=4)
        g.create_dataset("actions", data=np.asarray(actions))
        # copy the (trimmed) env_states verbatim by re-stacking the kept per-step dicts
        sg = g.create_group("env_states", track_order=True)
        _write_states(sg, states)
        if tid in instr:
            g.attrs["instructions"] = json.dumps(instr[tid])
        g.attrs["full_success"] = ok
        n_ok += int(ok)
        print(f"{tid}: {len(states)} steps (trimmed from {t['actions'].shape[0]}), "
              f"success={ok}", flush=True)
    fin.close()
    fout.close()
    env.close()
    print(f"wrote {args.out}: {len(tids)} trajs, {n_ok} verified full successes", flush=True)


def _write_states(group, states):
    """Re-stack a list of per-step state dicts back into the nested [T, ...] h5 layout."""
    def rec(g, list_of_vals, key):
        sample = list_of_vals[0]
        if isinstance(sample, dict):
            sub = g.create_group(key, track_order=True)
            for k in sample.keys():
                rec(sub, [v[k] for v in list_of_vals], k)
        else:
            g.create_dataset(key, data=np.asarray(list_of_vals))
    for k in states[0].keys():
        rec(group, [s[k] for s in states], k)


if __name__ == "__main__":
    main()
