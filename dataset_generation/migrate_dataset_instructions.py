"""In-place migration of published two-robot demo datasets to the unified
instructions format (adds the constant high-level instruction, nests the per-arm
subgoal instructions under ``low_level_instructions``).

Touches ONLY the instructions:
  * every ``*_dataset*.instructions.json`` sidecar, and
  * each trajectory group's ``attrs["instructions"]`` inside every ``*.h5``.
All obs / actions / env_states / other attrs are left byte-for-byte unchanged
(the h5 is opened ``r+`` and only the one string attribute is rewritten).

The migration is idempotent: a file already in the new layout is a no-op, so it is
safe to re-run.

OLD per-traj layout:   {seed, [max_wrist_winding_deg,] phase1_end, end, left, right}
NEW per-traj layout:   {seed, [max_wrist_winding_deg,] phase1_end, end,
                        low_level_instructions: {left, right}, high_level_instruction}
OLD sidecar:           {traj_id: <per-traj>, ...}
NEW sidecar:           {task_name, high_level_instruction, trajectories: {traj_id: <per-traj>}}

Usage:
  python migrate_dataset_instructions.py --root <downloaded_dataset_dir>
  python migrate_dataset_instructions.py --root <dir> --dry-run
"""
import argparse
import json
import os
import os.path as osp

import h5py


# Constant whole-task high-level instructions (must match generate_dataset_unified.py).
KNOWN_HLI = {
    "food_serve": "Place the bowl on the tray and the fruit in the bowl. "
                  "Then lift up the tray.",
    "clean_table": "Pick up all the cubes and put them into the box",
    "cook_pot": "Open the lid and put carrot in. Then move the pot to the target together.",
    "bottle_exchange": "Put each bottle into the box on the other side.",
    "prepare_fruit": "Put a banana and an orange onto each plate.",
}


def infer_task_name(path, override=None):
    """Infer the task key from a file/dir path: match any known task name that
    appears in the path (prefer the longest match). --task overrides this."""
    if override:
        return override
    base = path.replace("\\", "/").lower()
    hits = [t for t in KNOWN_HLI if t in base]
    if not hits:
        return None
    return max(hits, key=len)


def migrate_traj(d, hli):
    """Old per-traj dict -> new per-traj dict. Idempotent. Preserves every extra
    field (seed, phase1_end, end, max_wrist_winding_deg, ...) and their order."""
    if "low_level_instructions" in d:  # already new -> only backfill hli if missing
        if "high_level_instruction" not in d:
            d["high_level_instruction"] = hli
        return d
    out = {}
    left = None
    right = None
    for k, v in d.items():
        if k == "left":
            left = v
        elif k == "right":
            right = v
        else:
            out[k] = v  # keep seed / phase1_end / end / winding / etc. in order
    lli = {}
    if left is not None:
        lli["left"] = left
    if right is not None:
        lli["right"] = right
    out["low_level_instructions"] = lli
    out["high_level_instruction"] = hli
    return out


def migrate_sidecar(data, task_name, hli):
    """Old flat {traj_id:{...}} or new nested sidecar -> new nested sidecar. Idempotent."""
    if isinstance(data, dict) and "trajectories" in data:  # already nested
        data.setdefault("task_name", task_name)
        data.setdefault("high_level_instruction", hli)
        data["trajectories"] = {tid: migrate_traj(t, hli)
                                for tid, t in data["trajectories"].items()}
        return data
    trajs = {tid: migrate_traj(t, hli) for tid, t in data.items()}
    return {"task_name": task_name, "high_level_instruction": hli, "trajectories": trajs}


def migrate_json_file(path, task_name, hli, dry_run):
    with open(path) as f:
        data = json.load(f)
    was_new = isinstance(data, dict) and "trajectories" in data
    n = len(data.get("trajectories", data)) if isinstance(data, dict) else 0
    new = migrate_sidecar(data, task_name, hli)
    if not dry_run:
        with open(path, "w") as f:
            json.dump(new, f, indent=2)
    print(f"  [json] {osp.basename(path)}: {n} trajs "
          f"({'already-nested, refreshed' if was_new else 'flat -> nested'})"
          f"{' (dry-run)' if dry_run else ''}")


def migrate_h5_file(path, task_name, hli, dry_run):
    mode = "r" if dry_run else "r+"
    changed = 0
    skipped = 0
    with h5py.File(path, mode) as f:
        for tid in f.keys():
            g = f[tid]
            if "instructions" not in g.attrs:
                continue
            d = json.loads(g.attrs["instructions"])
            if "low_level_instructions" in d and "high_level_instruction" in d:
                skipped += 1
                continue
            nd = migrate_traj(d, hli)
            if not dry_run:
                g.attrs["instructions"] = json.dumps(nd)  # rewrites ONLY this attr
            changed += 1
    print(f"  [ h5 ] {osp.basename(path)}: {changed} traj attrs migrated, "
          f"{skipped} already-new{' (dry-run)' if dry_run else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="downloaded dataset directory")
    p.add_argument("--task", default=None,
                   help="force the task key for ALL files (else inferred per path)")
    p.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = p.parse_args()

    json_paths = []
    h5_paths = []
    for dirpath, _, files in os.walk(args.root):
        for fn in files:
            fp = osp.join(dirpath, fn)
            if fn.endswith(".instructions.json"):
                json_paths.append(fp)
            elif fn.endswith(".h5"):
                h5_paths.append(fp)

    print(f"root: {args.root}")
    print(f"found {len(json_paths)} instruction sidecars, {len(h5_paths)} h5 files"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")

    unknown = []
    for fp in sorted(json_paths):
        task = infer_task_name(fp, args.task)
        if task is None or task not in KNOWN_HLI:
            unknown.append(fp)
            continue
        migrate_json_file(fp, task, KNOWN_HLI[task], args.dry_run)
    for fp in sorted(h5_paths):
        task = infer_task_name(fp, args.task)
        if task is None or task not in KNOWN_HLI:
            unknown.append(fp)
            continue
        migrate_h5_file(fp, task, KNOWN_HLI[task], args.dry_run)

    if unknown:
        print("\nWARNING: could not infer a known task for these files "
              "(skipped -- pass --task to force):")
        for fp in unknown:
            print("  ", fp)
    print("\ndone." + (" (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
