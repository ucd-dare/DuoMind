"""Copy the first N trajectories of an rgb dataset h5 (and its instructions json)
into new files, for a fixed-size training subset.

Usage: python subset_50.py <in_rgb.h5> <in_instructions.json> <out_rgb.h5> <out_instructions.json> [N]
"""
import json
import sys

import h5py

inp, in_json, outp, out_json = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 50

fin = h5py.File(inp, "r")
fout = h5py.File(outp, "w")
ids = sorted([k for k in fin if k.startswith("traj_")], key=lambda t: int(t.split("_")[1]))
keep = ids[:N]
for tid in keep:
    fin.copy(fin[tid], fout, tid)  # preserves datasets + attrs (instructions, full_success)
fin.close()
fout.close()

instr = json.load(open(in_json))
sub = {k: instr[k] for k in keep if k in instr}
json.dump(sub, open(out_json, "w"))
print(f"wrote {outp}: {len(keep)} trajs; {out_json}: {len(sub)} instructions")
