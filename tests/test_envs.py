"""Seven-task registration and real distributed custom-policy smoke checks."""
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import pytest

from policy.custom import load_policy
from policy.custom.my_policy import MyPolicy
from script.eval_robopoly_custom import TASK_NAMES, TASKS, run_episode
import mani_skill.envs
from mani_skill.utils.registration import REGISTERED_ENVS


def test_only_benchmark_tasks_are_registered():
    assert set(REGISTERED_ENVS) == {TASKS[task]['env_id'] for task in TASK_NAMES}


class EmptyModel:
    """No trained checkpoint; exercise the exact MyPolicy loader/inference path."""
    def infer(self, observation):
        assert set(observation) == {'images', 'state', 'prompt'}
        assert set(observation['images']) == {'head_camera', 'wrist_camera'}
        assert observation['images']['head_camera'].shape[0] == 3
        assert observation['images']['wrist_camera'].shape == (3, 128, 128)
        assert observation['state'].shape == (8,)
        assert isinstance(observation['prompt'], str)
        return {'actions': np.zeros(8, dtype=np.float32)}


@pytest.mark.parametrize('task', TASK_NAMES)
def test_custom_template_reset_step_and_video(task, tmp_path):
    # Only model loading is supplied; MyPolicy inherits reset/act unchanged.
    with patch.object(MyPolicy, 'load_model', side_effect=lambda config: EmptyModel()):
        policies = [load_policy('policy.custom.my_policy:my_algorithm', {}) for _ in range(2)]
    args = SimpleNamespace(task=task, max_episode_steps=2, obs_mode='rgb',
                           render_mode='rgb_array', shader='default', sim_backend='physx_cpu',
                           food_serve_friction=None, food_serve_bowl_mass_scale=None,
                           food_serve_gripper_friction=None, prepare_fruit_orange_scale=None)
    video = tmp_path / f'{task}.mp4'
    try:
        result = run_episode(args, policies, seed=100000, video_path=video)
        assert result['steps'] == 2
        assert video.stat().st_size > 0
        reader = imageio.get_reader(video)
        try:
            assert reader.get_data(0).shape[0] > 384  # global image plus instruction panel
        finally:
            reader.close()
    finally:
        for policy in policies:
            policy.close()


def test_generation_pipeline_matches_seven_tasks():
    from dataset_generation.generate_dataset_unified import TASKS as generation_tasks
    assert set(generation_tasks) == set(TASK_NAMES)
    for task, spec in generation_tasks.items():
        assert spec.env_id == TASKS[task]['env_id']
        assert callable(spec.solve)
