"""Evaluate a registered custom controller using RoboPoly's standard environments."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from policy.custom import load_policy
from policy.video_instructions import add_instruction_panel
# Simulator helpers do not import OpenPI, Transformers, or a planner.
from script.robopoly_runtime import TASKS, _make_env, _success, _to_numpy, _video_frame, _agent_state_from_obs, _rgb_from_obs, _policy_image, GLOBAL_CAMERA, WRIST_CAMERAS

TASK_NAMES = tuple(TASKS)


def agent_observation(observation, agent_id, instruction):
    """Match DuoMind's native per-robot VLA input without exposing peer observations."""
    return {
        'images': {
            'head_camera': _policy_image(_rgb_from_obs(observation, GLOBAL_CAMERA)).copy(),
            'wrist_camera': _policy_image(_rgb_from_obs(observation, WRIST_CAMERAS[agent_id])).copy(),
        },
        'state': _agent_state_from_obs(observation, agent_id).copy(),
        'prompt': instruction,
    }


def policy_step(policy, observation):
    result = policy.act(observation)
    if not isinstance(result, dict) or 'actions' not in result:
        raise ValueError('act() must return actions and an optional instruction string')
    actions = _to_numpy(result['actions']).astype(np.float32)
    if actions.shape == (8,):
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < 1 or not np.isfinite(actions).all():
        raise ValueError('Own-robot actions must be finite with shape (8,) or (T, 8)')
    instruction = result.get('instruction', '')
    if not isinstance(instruction, str):
        raise ValueError('instruction must be a string')
    return actions, instruction


def run_episode(args, policies, seed, video_path=None):
    task = TASKS[args.task]
    env = _make_env(args, task['env_id'])
    writer = None
    try:
        observation, _ = env.reset(seed=seed)
        for agent_id, policy in enumerate(policies):
            policy.reset(seed=seed, robot_id=agent_id)
        pending = [[], []]
        instructions = ['', '']
        success = False
        if video_path is not None:
            import imageio.v2 as imageio
            writer = imageio.get_writer(video_path, fps=20, macro_block_size=1)
        for step in range(args.max_episode_steps):
            own_actions = []
            for agent_id, policy in enumerate(policies):
                if not pending[agent_id]:
                    chunk, instructions[agent_id] = policy_step(
                        policy, agent_observation(observation, agent_id, task['high_level_instruction']))
                    pending[agent_id] = list(chunk)
                own_actions.append(pending[agent_id].pop(0))
            actions = np.concatenate(own_actions)
            if writer is not None:
                writer.append_data(add_instruction_panel(
                    _video_frame(env, observation, "global_camera"), task['high_level_instruction'], instructions))
            observation, _, terminated, truncated, info = env.step(actions)
            success = success or _success(info)
            if success or bool(_to_numpy(terminated).any()) or bool(_to_numpy(truncated).any()):
                break
        if writer is not None:
            writer.append_data(add_instruction_panel(
                _video_frame(env, observation, "global_camera"), task['high_level_instruction'], instructions))
        return {'seed': seed, 'success': bool(success), 'steps': step + 1}
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_NAMES, required=True)
    parser.add_argument('--custom-policy', required=True, help='module:registered_name')
    parser.add_argument('--policy-config', type=Path, help='JSON object passed to the policy factory')
    parser.add_argument('--experiment')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--videos', type=int, default=2)
    parser.add_argument('--max-episode-steps', type=int, default=1500)
    parser.add_argument('--result-dir', type=Path, default=ROOT / 'eval_result/robopoly')
    args = parser.parse_args()
    if args.episodes < 1 or args.videos < 0 or args.max_episode_steps < 1:
        parser.error('Episode/step counts must be positive and videos nonnegative')
    # Identical simulator/control defaults to the reference evaluator, without
    # task-specific diagnostic overrides or a reference policy's chunk limits.
    args.obs_mode, args.render_mode, args.shader = 'rgb', 'rgb_array', 'default'
    args.sim_backend = 'physx_cpu'
    args.food_serve_friction = args.food_serve_bowl_mass_scale = args.food_serve_gripper_friction = None
    args.prepare_fruit_orange_scale = None
    config = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(config, dict):
        parser.error('--policy-config must contain a JSON object')
    name = args.experiment or args.custom_policy.split(':')[-1]
    if not name or name in ('.', '..') or Path(name).name != name:
        parser.error('Experiment name must be a single directory name')
    output = args.result_dir / TASKS[args.task]['env_id'] / 'custom' / name / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    policies = []
    try:
        for _ in range(2):
            policies.append(load_policy(args.custom_policy, config.copy()))
        with (output / 'episode_results.jsonl').open('w') as stream:
            for i in range(args.episodes):
                record = run_episode(args, policies, args.seed + i, output / f'episode_{i}.mp4' if i < args.videos else None)
                records.append(record)
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                print(record, flush=True)
        summary = dict(task=args.task, policy=args.custom_policy, mode="custom",
                       episodes=len(records), successes=sum(r['success'] for r in records),
                       success_rate=sum(r['success'] for r in records) / len(records),
                       start_seed=args.seed, max_episode_steps=args.max_episode_steps,
                       policy_config=config)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Results: {output}', flush=True)
    finally:
        for policy in policies:
            if callable(getattr(policy, 'close', None)):
                policy.close()


if __name__ == '__main__':
    main()
