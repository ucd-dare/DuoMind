"""Check dynamically loaded entry points and bundled assets after package renames."""

from importlib.resources import files

import gymnasium as gym
from gymnasium.envs.registration import load_env_creator

import robopoly
from robopoly.utils.registration import REGISTERED_ENVS, TimeLimitWrapper


def test_registered_wrappers_resolve():
    assert REGISTERED_ENVS
    for env_id, env_spec in REGISTERED_ENVS.items():
        assert env_spec.cls.__module__.startswith("robopoly.")
        wrappers = gym.spec(env_id).additional_wrappers
        assert wrappers
        assert any(
            load_env_creator(wrapper.entry_point) is TimeLimitWrapper
            for wrapper in wrappers
        )


def test_bundled_robot_assets_resolve():
    asset = "robots/panda/panda_v2.urdf"
    assert (robopoly.PACKAGE_ASSET_DIR / asset).is_file()
    assert files("robopoly").joinpath("assets", asset).is_file()
