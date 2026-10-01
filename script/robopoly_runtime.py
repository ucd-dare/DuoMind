"""RoboPoly simulator and observation helpers, independent of policy backends."""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np


TASKS = {'clean_table': {'env_id': 'TwoRobotCleanTableReplicaCAD-v1',
                 'high_level_instruction': 'Pick up all the cubes and put them into the box'},
 'cook_pot': {'env_id': 'TwoRobotCookPotReplicaCAD-v1',
              'high_level_instruction': 'Open the lid and put carrot in. Then move the pot to the '
                                        'target together.'},
 'food_serve': {'env_id': 'TwoRobotFoodServeReplicaCAD-v1',
                'high_level_instruction': 'Right arm pass the bread to the middle and Left arm put '
                                          'both mug and bread onto the tray.'},
 'hang_bag': {'env_id': 'TwoRobotHangBagReplicaCAD-v1',
              'high_level_instruction': 'Pass the bag from the left arm to the right arm and hang '
                                        'it on the hook.'},
 'exchange_bread': {'env_id': 'TwoRobotBreadExchangeReplicaCAD-v1',
                    'high_level_instruction': 'Put each bread into the box on the other side.'},
 'prepare_snack': {'env_id': 'TwoRobotPrepareSnackReplicaCAD-v1',
                   'high_level_instruction': 'Put two square breads onto each plate.'},
 'put_object_cabinet': {'env_id': 'TwoRobotPutObjectCabinetReplicaCAD-v1',
                        'high_level_instruction': 'Open the drawer and put the cube inside. Then '
                                                  'close the drawer.'}}

GLOBAL_CAMERA = "global_camera"

WRIST_CAMERAS = {
    0: "panda_wristcam-0-hand_camera",
    1: "panda_wristcam-1-hand_camera",
}

AGENT_SIDES = {0: "left", 1: "right"}

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)

def _success(info: dict) -> bool:
    value = info.get("success", False)
    arr = _to_numpy(value)
    return bool(arr.reshape(-1)[0]) if arr.size else bool(value)

def _rgb_from_obs(observation: dict, camera_name: str) -> np.ndarray:
    rgb = _to_numpy(observation["sensor_data"][camera_name]["rgb"])
    if rgb.ndim == 5 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

def _policy_image(rgb: np.ndarray) -> np.ndarray:
    return np.moveaxis(rgb, -1, 0)

def _video_frame(env, observation: dict, camera_name: str | None = None) -> np.ndarray:
    if camera_name is None or camera_name == "render":
        frame = _to_numpy(env.render())
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    if camera_name == "global_and_wrists":
        global_frame = _rgb_from_obs(observation, GLOBAL_CAMERA)
        wrist_frames = [_rgb_from_obs(observation, WRIST_CAMERAS[agent_id]) for agent_id in AGENT_SIDES]
        global_h, global_w = global_frame.shape[:2]
        wrist_h, wrist_w = wrist_frames[0].shape[:2]
        if any(frame.shape[:2] != (wrist_h, wrist_w) for frame in wrist_frames):
            raise ValueError("Both wrist cameras must have the same native resolution")
        # Preserve every policy input pixel: 512x384 global and two native
        # 128x128 wrists. Padding, rather than resize, fills the unused strip.
        canvas = np.zeros((max(global_h, 2 * wrist_h), global_w + wrist_w, 3), dtype=np.uint8)
        canvas[:global_h, :global_w] = global_frame
        for agent_id, frame in enumerate(wrist_frames):
            start = agent_id * wrist_h
            canvas[start : start + wrist_h, global_w : global_w + wrist_w] = frame
        return canvas
    return _rgb_from_obs(observation, camera_name)

def _agent_state_from_obs(observation: dict, agent_id: int) -> np.ndarray:
    def panda_state(qpos_value: Any) -> np.ndarray:
        qpos = _to_numpy(qpos_value)
        if qpos.ndim > 1 and qpos.shape[0] == 1:
            qpos = qpos[0]
        qpos = qpos.reshape(-1)
        if qpos.size < 8:
            raise ValueError(f"Expected at least 8 Panda qpos values, got {qpos.shape}")
        return np.concatenate([qpos[:7], qpos[7:8]]).astype(np.float32)

    agent_obs = observation.get("agent", {})
    for key in (f"panda_wristcam-{agent_id}", f"panda-{agent_id}", str(agent_id)):
        if key in agent_obs and "qpos" in agent_obs[key]:
            return panda_state(agent_obs[key]["qpos"])
    if "qpos" in agent_obs:
        qpos = _to_numpy(agent_obs["qpos"]).reshape(-1)
        start = agent_id * 9
        if qpos.size >= start + 9:
            return panda_state(qpos[start : start + 9])
    raise KeyError(f"Could not extract observed qpos for panda-{agent_id}")

def _make_env(args: argparse.Namespace, env_id: str):
    import gymnasium as gym
    import robopoly.envs  # noqa: F401
    import sapien.physx as physx
    from robopoly.utils.wrappers.flatten import FlattenActionSpaceWrapper

    if args.task == "clean_table":
        # The legacy visualization camera remains available on the task class,
        # but evaluation renders only the new global training view. The two
        # wrist cameras are robot-mounted and remain available automatically.
        import robopoly.envs.tasks.tabletop.two_robot_clean_table as clean_table_module

        clean_table_cls = clean_table_module.TwoRobotCleanTableReplicaCADEnv
        all_sensor_configs = clean_table_cls._default_sensor_configs.fget

        def only_global_camera(self):
            return [config for config in all_sensor_configs(self) if config.uid == GLOBAL_CAMERA]

        clean_table_cls._default_sensor_configs = property(only_global_camera)

    task_env_kwargs = {}
    if args.task == "prepare_fruit" and args.prepare_fruit_orange_scale is not None:
        task_env_kwargs["orange_scale"] = args.prepare_fruit_orange_scale
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        reward_mode=None,
        control_mode="pd_joint_pos",
        robot_init_qpos_noise=0,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        **task_env_kwargs,
    )
    if args.task == "clean_table":
        # The three-cube Simple task is defined with 80%-scale cubes in the
        # registered environment itself. Keep an explicit guard here so eval
        # cannot silently drift back to the original 64 mm cube geometry.
        base_env = env.unwrapped
        expected_half_size = 0.032 * 0.8
        actual_half_size = float(base_env.item_half_size)
        if not np.isclose(actual_half_size, expected_half_size, atol=1e-9):
            raise RuntimeError(
                "Clean-table eval cube-size mismatch: expected "
                f"half_size={expected_half_size:.4f} m (51.2 mm width), got "
                f"{actual_half_size:.4f} m"
            )
        print(
            f"[env] clean-table cube width={2 * actual_half_size * 1000:.1f} mm "
            "(80% of original)",
            flush=True,
        )
    if args.food_serve_friction is not None:
        if args.task != "food_serve":
            raise ValueError("--food-serve-friction is only valid for --task food_serve")
        friction = float(args.food_serve_friction)
        if friction <= 0:
            raise ValueError("--food-serve-friction must be positive")
        base_env = env.unwrapped
        shape_count = 0
        for actor_name in ("bowl", "fruit"):
            actor = getattr(base_env, actor_name)
            material = physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
            )
            for body in actor._bodies:
                for shape in body.collision_shapes:
                    shape.set_physical_material(material)
                    shape_count += 1
        print(
            f"[env] temporary bowl/apple friction={friction:g}; updated {shape_count} collision shapes",
            flush=True,
        )
    if args.food_serve_bowl_mass_scale is not None:
        if args.task != "food_serve":
            raise ValueError("--food-serve-bowl-mass-scale is only valid for --task food_serve")
        mass_scale = float(args.food_serve_bowl_mass_scale)
        if not 0 < mass_scale <= 1:
            raise ValueError("--food-serve-bowl-mass-scale must be in (0, 1]")
        bowl = env.unwrapped.bowl
        mass_changes = []
        for body in bowl._bodies:
            original_mass = float(body.mass)
            new_mass = original_mass * mass_scale
            body.set_mass(new_mass)
            mass_changes.append((original_mass, new_mass))
        print(
            f"[env] temporary same-asset bowl mass scale={mass_scale:g}; "
            f"mass changes={mass_changes}",
            flush=True,
        )
    if args.food_serve_gripper_friction is not None:
        if args.task != "food_serve":
            raise ValueError("--food-serve-gripper-friction is only valid for --task food_serve")
        gripper_friction = float(args.food_serve_gripper_friction)
        if gripper_friction <= 0:
            raise ValueError("--food-serve-gripper-friction must be positive")
        gripper_material = physx.PhysxMaterial(
            static_friction=gripper_friction,
            dynamic_friction=gripper_friction,
            restitution=0.0,
        )
        shape_count = 0
        for agent_name in ("left_agent", "right_agent"):
            agent = getattr(env.unwrapped, agent_name)
            for link_name in ("finger1_link", "finger2_link"):
                link = getattr(agent, link_name)
                for body in link._bodies:
                    for shape in body.collision_shapes:
                        shape.set_physical_material(gripper_material)
                        shape_count += 1
        print(
            f"[env] temporary Panda gripper friction={gripper_friction:g}; "
            f"updated {shape_count} collision shapes",
            flush=True,
        )
    return FlattenActionSpaceWrapper(env)
