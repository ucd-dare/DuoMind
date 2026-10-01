"""Evaluate custom distributed policies on RoboPoly."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.robopoly_runtime import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('eval',))
    parser.add_argument('benchmark', choices=('robopoly',))
    parser.add_argument('task', choices=tuple(TASKS))
    parser.add_argument('--mode', choices=('custom',), default='custom')
    parser.add_argument('--custom-policy', required=True, help='module:registered_name')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--policy-config', type=Path)
    parser.add_argument('--experiment')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--videos', type=int, default=2)
    parser.add_argument('--max-episode-steps', type=int, default=1500)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / 'script/eval_robopoly_custom.py'), '--task', args.task,
               '--custom-policy', args.custom_policy, '--episodes', str(args.episodes),
               '--seed', str(args.seed), '--videos', str(args.videos),
               '--max-episode-steps', str(args.max_episode_steps)]
    if args.policy_config:
        command += ['--policy-config', str(args.policy_config.resolve())]
    if args.experiment:
        command += ['--experiment', args.experiment]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), env.get('PYTHONPATH', '')])
    if args.dry_run:
        import shlex
        print(shlex.join(command))
    else:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
