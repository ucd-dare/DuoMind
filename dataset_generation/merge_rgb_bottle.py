"""Concatenate per-shard 3-camera RGB h5s (from rerender_dataset_bottle.py) into one
dataset, renumbering trajectories traj_0..traj_{N-1} and capping at --limit. Each
traj group (obs/sensor_data/*/rgb, actions, env_states, attrs incl. instructions) is
copied verbatim. Also writes a merged instructions sidecar.

Usage:
  python merge_rgb_bottle.py --shards <dir> --pattern 'shard_*.rgb.h5' \
      --out <dataset>.rgb.h5 --instructions <dataset>.instructions.json [--limit 100]
"""
import argparse
import json
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--pattern", default="shard_*.rgb.h5")
    p.add_argument("--out", required=True)
    p.add_argument("--instructions", required=True)
    p.add_argument("--task-name", default=None,
                   help="task_name for the merged nested sidecar (e.g. bottle_exchange)")
    p.add_argument("--limit", type=int, default=10**9)
    args = p.parse_args()

    paths = sorted(Path(args.shards).rglob(args.pattern))
    fout = h5py.File(args.out, "w")
    merged = {}
    high_level = None
    n = 0
    for sp in paths:
        try:
            fin = h5py.File(sp, "r")
        except Exception as e:
            print(f"  skip unreadable shard {sp.name}: {str(e)[:50]}", flush=True)
            continue
        try:
            for tid in sorted(fin.keys(), key=lambda t: int(t.split("_")[1])):
                if n >= args.limit:
                    break
                new_id = f"traj_{n}"
                try:
                    fin.copy(fin[tid], fout, name=new_id)
                except Exception as e:
                    print(f"  skip bad traj {sp.name}/{tid}: {str(e)[:50]}", flush=True)
                    if new_id in fout:
                        del fout[new_id]
                    continue
                if "instructions" in fin[tid].attrs:
                    d = json.loads(fin[tid].attrs["instructions"])
                    merged[new_id] = d
                    if high_level is None:
                        high_level = d.get("high_level_instruction")
                n += 1
        finally:
            fin.close()
        if n >= args.limit:
            break
    fout.close()
    # Emit the nested sidecar layout when we have the high-level context; otherwise
    # stay flat so legacy per-traj dicts round-trip unchanged.
    if args.task_name is not None or high_level is not None:
        out = {"task_name": args.task_name,
               "high_level_instruction": high_level,
               "trajectories": merged}
    else:
        out = merged
    with open(args.instructions, "w") as f:
        json.dump(out, f, indent=2)
    print(f"merged {n} trajectories -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
