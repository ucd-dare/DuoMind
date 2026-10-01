# Generate RoboPoly demonstrations

Run from the repository root in the benchmark environment:

```bash
TASK=clean_table
DEMOS=50
OUT="data/robopoly/generated/$TASK"
python dataset_generation/generate_dataset_unified.py \
  --task "$TASK" --num-traj "$DEMOS" --record-dir "$OUT" \
  --out "${TASK}_dataset" --start-seed 0
```

Use a fresh output directory. The generator counts successful demonstrations,
retries failed plans, and raises an error if `--max-seeds` is exhausted before
reaching the requested count. Outputs are a state/action HDF5 trajectory file,
its episode metadata JSON, and an `.instructions.json` sidecar. `--save-video`
records preview videos; it does not add RGB observations to the HDF5.

## RGB rendering

Render the recorded states with the task-specific helper below. Each accepts
`--in STATE.h5 --out OUTPUT.rgb.h5 --instructions INPUT.instructions.json`.
Paths are relative to `dataset_generation/`.

| Task | Helper and extra arguments |
| --- | --- |
| `clean_table` | `rerender_dataset_cleantable.py` |
| `cook_pot` | `rerender_dataset_cookpot.py` |
| `food_serve` | `rerender_dataset.py` |
| `put_object_cabinet` | `rerender_dataset_cabinet.py` |
| `exchange_bread` | `rerender_dataset_bottle.py --env-id TwoRobotBreadExchangeReplicaCAD-v1` |
| `prepare_snack` | `rerender_dataset_preparefruit.py --env-id TwoRobotPrepareSnackReplicaCAD-v1` |
| `hang_bag` | `rerender_dataset_preparefruit.py --env-id TwoRobotHangBagReplicaCAD-v1` |

For example, after generating Clean Table demonstrations:

```bash
python dataset_generation/rerender_dataset_cleantable.py \
  --in "$OUT/${TASK}_dataset.h5" --out "$OUT/${TASK}_dataset.rgb.h5" \
  --instructions "$OUT/${TASK}_dataset.instructions.json"
```

Rendering retains the global and both wrist cameras for training data; evaluation
routes only the global image and a robot's own wrist image to that robot's policy.
