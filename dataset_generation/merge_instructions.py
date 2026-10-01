"""Merge per-shard instruction sidecars into one, matching the traj-id numbering
that ``mani_skill.trajectory.merge_trajectory`` produces.

``merge_trajectory`` globs ``sorted(input_dir.rglob(pattern))`` and re-numbers
every episode sequentially as ``traj_0, traj_1, ...`` in that order. This mirrors
exactly that ordering so the merged instructions line up with the merged h5.

Usage:
  python merge_instructions.py --input-dir <dir> --pattern 'shard_*.h5' -o out.instructions.json
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", default="shard_*.h5")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    h5_paths = sorted(Path(args.input_dir).rglob(args.pattern))
    merged = {}
    task_name = None
    high_level = None
    cnt = 0
    for h5p in h5_paths:
        instr_path = str(h5p).replace(".h5", ".instructions.json")
        # A shard with 0 successes has no instructions sidecar (and contributes no
        # episodes to merge_trajectory either), so skip it -- ids stay aligned.
        if not Path(instr_path).exists():
            continue
        data = json.load(open(instr_path))
        # Accept both the new nested layout ({task_name, high_level_instruction,
        # trajectories:{...}}) and the legacy flat {traj_id: {...}} sidecar.
        trajs = data.get("trajectories", data)
        if task_name is None:
            task_name = data.get("task_name")
            high_level = data.get("high_level_instruction")
        for tid in sorted(trajs, key=lambda t: int(t.split("_")[1])):
            merged[f"traj_{cnt}"] = trajs[tid]
            cnt += 1
    # Re-wrap in the nested layout when the shards carried it; otherwise stay flat
    # so legacy datasets round-trip unchanged.
    if task_name is not None or high_level is not None:
        out = {"task_name": task_name,
               "high_level_instruction": high_level,
               "trajectories": merged}
    else:
        out = merged
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"merged {cnt} instruction entries from {len(h5_paths)} shards "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
