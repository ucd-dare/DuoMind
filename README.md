<h1 align="center">DuoMind: Enabling Multi-Robot Coordination via Communication</h1>

<p align="center">
  <img src="docs/paper.svg" alt="Paper">
  <a href="https://hanchuzhou.github.io/duomind_project_page/"><img src="docs/project-page.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/ucd-dare/multi-agent-demo/tree/main"><img src="docs/dataset.svg" alt="Dataset"></a>
</p>

<p align="center">
  <img src="docs/overview.png" alt="DuoMind overview" width="100%">
</p>

DuoMind is a distributed hierarchical framework for multi-robot coordination through semantic communication.

 We also introduce RoboPoly, a multi-robot benchmark comprising long-horizon manipulation tasks that require coordinated execution under distributed control. It contains seven multi-robot tasks: `hang_bag`, `food_serve`, `prepare_snack`, `clean_table`, `cook_pot`,
`put_object_cabinet`, and `exchange_bread`.

![DuoMind code is coming soon. The RoboPoly dataset will be released in a few days.](docs/news.svg)

## Installation

Use Linux, Python 3.11, an NVIDIA GPU with a CUDA-compatible driver, Vulkan, and `uv`.
Run the commands from the **repository root**:

```bash
uv venv --python 3.11 .venv-robopoly
source .venv-robopoly/bin/activate
uv pip install -r script/requirements_robopoly.txt
uv pip install --no-deps -e .

export MS_ASSET_DIR="$PWD/.cache/robopoly"
python -m robopoly.utils.download_asset ReplicaCAD
python -m robopoly.utils.download_asset RoboCasa
```

The Python package is `robopoly`. After updating an existing checkout, rerun
`uv pip install --no-deps -e .` and change imports in your own code from
`mani_skill` to `robopoly`. Existing `MS_ASSET_DIR` settings remain supported.

## Dataset

We provide expert demonstrations for all seven tasks in the
[expert demonstration dataset](https://huggingface.co/datasets/ucd-dare/multi-agent-demo).
To generate more data with our planning-based pipeline:

```bash
TASK=clean_table
DEMOS=50
python dataset_generation/generate_dataset_unified.py \
  --task "$TASK" --num-traj "$DEMOS" --start-seed 0 \
  --record-dir "data/robopoly/generated/$TASK" --out "${TASK}_dataset"
```

Change `TASK` and `DEMOS` to select the task and dataset size. Use a new output
directory for each run. The generator saves successful expert trajectories
(states and actions) and per-robot instruction metadata. Add `--save-video` for
preview videos; use the task-specific `rerender_dataset*.py` scripts in
[dataset_generation/](dataset_generation/) to add camera observations.

## Register your algorithm

We provide a registered [policy template](policy/custom/my_policy.py) that follows the distributed settings. Implement only `load_model(config)` to load your checkpoint;

The loaded model must implement `infer(observation)` and return
`{"actions": own_action}`.

Each policy receives the global
RGB image, **its own wrist RGB image**, its own 8-value state, and the **high-level
task instruction**. Images use `(3, H, W)` format. The other robot's wrist image
and state are not exposed. Each robot has a separate policy instance.

The returned actions have shape `(8,)` or `(T, 8)` for an action chunk:
seven absolute joint targets in radians and a normalized gripper command
(`-1` closed, `+1` open). The evaluator combines the two robots' actions.

## Evaluation

To check installation with the included empty policy (no checkpoint required):

```bash
python script/benchmark.py eval robopoly clean_table \
  --mode custom --custom-policy policy.custom.example:example \
  --gpus 0 --episodes 1 --videos 1 --max-episode-steps 2
```

After implementing model loading in the template, evaluate your policy:

```bash
python script/benchmark.py eval robopoly clean_table \
  --mode custom --custom-policy policy.custom.my_policy:my_algorithm \
  --gpus 0 --episodes 20 --videos 2
```

Replace `clean_table` with any task above. Use `--policy-config config.json` for
model settings (for example, `{"checkpoint": "/path/to/checkpoint"}`), and
`--experiment exp_name` to customize the run name.
Use `--episodes 400` for full evaluation, `--seed SEED` to change the starting
seed, and `--videos 0` to disable recording.

Metrics and videos are saved under
`eval_result/robopoly/<environment>/custom/<experiment>/`.

## Development

Install the optional Git hooks from the repository root:

```bash
uv pip install pre-commit
pre-commit install --config .github/pre-commit-config.yaml
```

## Citation

Citation template (publication details to be added):

```bibtex
@misc{duomind,
  title = {DuoMind: Enabling Multi-Robot Coordination via Communication},
  author = {TODO},
  year = {TODO},
  howpublished = {TODO},
  url = {TODO}
}
```

## Acknowledgments

The DuoMind codebase is developed based on RoboTwin 2.0, and RoboPoly is built on
ManiSkill 3. We thank Keyu Zhu and Vivian Xie for their contributions to building
RoboPoly.

See [LICENSE](LICENSE) and the
[RoboTwin MIT license](licenses/LICENSE-EMBODIED-MA) for licensing details.
