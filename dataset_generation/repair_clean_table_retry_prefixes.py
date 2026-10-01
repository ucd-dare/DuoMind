"""Remove failed retry prefixes from a merged CleanTable state dataset.

Older ``generate_dataset_unified.py`` kept RecordEpisode's grow-only buffer when
retrying a scene.  A successful trajectory could therefore contain one or more
failed attempts before the final successful attempt, while its instruction
boundaries correctly described only the final attempt.  This utility keeps the
last ``instructions[traj_id]["end"]`` actions and matching ``end + 1`` states.
"""
import argparse
import json

import h5py


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_group(src, dst, old_steps, kept_steps):
    _copy_attrs(src, dst)
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            child = dst.create_group(name, track_order=True)
            _copy_group(obj, child, old_steps, kept_steps)
            continue

        if obj.ndim and obj.shape[0] == old_steps:
            data = obj[-kept_steps:]
        elif obj.ndim and obj.shape[0] == old_steps + 1:
            data = obj[-(kept_steps + 1):]
        else:
            data = obj[...]
        kwargs = {}
        if obj.compression is not None:
            kwargs["compression"] = obj.compression
            kwargs["compression_opts"] = obj.compression_opts
        copied = dst.create_dataset(name, data=data, **kwargs)
        _copy_attrs(obj, copied)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--instructions", required=True)
    parser.add_argument("--trajectory-json", default=None)
    parser.add_argument("--out-trajectory-json", default=None)
    args = parser.parse_args()

    sidecar = json.load(open(args.instructions))
    instructions = sidecar.get("trajectories", sidecar)
    repaired = []
    with h5py.File(args.inp, "r") as src, h5py.File(args.out, "x") as dst:
        _copy_attrs(src, dst)
        for traj_id in sorted(src, key=lambda x: int(x.split("_")[1])):
            old_steps = len(src[traj_id]["actions"])
            kept_steps = int(instructions[traj_id]["end"])
            if kept_steps > old_steps:
                raise ValueError(f"{traj_id}: instruction end exceeds actions")
            group = dst.create_group(traj_id, track_order=True)
            _copy_group(src[traj_id], group, old_steps, kept_steps)
            if old_steps != kept_steps:
                repaired.append((traj_id, old_steps, kept_steps))

    if args.trajectory_json:
        if not args.out_trajectory_json:
            raise ValueError("--out-trajectory-json is required")
        meta = json.load(open(args.trajectory_json))
        for episode in meta.get("episodes", []):
            traj_id = f"traj_{episode['episode_id']}"
            episode["elapsed_steps"] = int(instructions[traj_id]["end"])
        with open(args.out_trajectory_json, "x") as file:
            json.dump(meta, file, indent=2)

    print(f"wrote {args.out}; repaired {len(repaired)} retry-prefixed trajectories")
    for traj_id, old_steps, kept_steps in repaired:
        print(f"  {traj_id}: {old_steps} -> {kept_steps}")


if __name__ == "__main__":
    main()
