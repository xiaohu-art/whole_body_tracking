from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms, quat_apply_yaw, quat_inv

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_global_ref_position_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_ref_pose_w[:, :3] - command.robot_ref_pose_w[:, :3], dim=-1)


def motion_global_ref_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_global_ref_position_error(env, command_name) ** 2 / std ** 2)


def motion_global_ref_orientation_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return quat_error_magnitude(command.motion_ref_pose_w[:, 3:7], command.robot_ref_pose_w[:, 3:7])


def motion_global_ref_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_global_ref_orientation_error(env, command_name) ** 2 / std ** 2)


def motion_global_body_position_error(env: ManagerBasedRLEnv, command_name: str,
                                      body_names: list[str] | None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    return torch.norm(command.motion_body_pose_w[:, body_indexes, :3] - command.robot_body_pose_w[:, body_indexes, :3],
                      dim=-1).mean(-1)


def motion_global_body_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                                          body_names: Optional[list[str]] = None) -> torch.Tensor:
    return torch.exp(-motion_global_body_position_error(env, command_name, body_names) ** 2 / std ** 2)


def motion_relative_body_position_error(env: ManagerBasedRLEnv, command_name: str,
                                        body_names: list[str] | None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    robot_pos_b, _ = subtract_frame_transforms(
        command.robot_ref_pose_w[:, None, :3].repeat(1, len(body_indexes), 1),
        command.robot_ref_pose_w[:, None, 3:7].repeat(1, len(body_indexes), 1),
        command.robot_body_pose_w[:, :, :3], command.robot_body_pose_w[:, :, 3:7],
    )

    motion_pos_b, _ = subtract_frame_transforms(
        command.motion_ref_pose_w[:, None, :3].repeat(1, len(body_indexes), 1),
        command.motion_ref_pose_w[:, None, 3:7].repeat(1, len(body_indexes), 1),
        command.motion_body_pose_w[:, body_indexes, :3], command.motion_body_pose_w[:, body_indexes, 3:7],
    )
    return torch.norm(motion_pos_b - robot_pos_b, dim=-1).mean(-1)


def motion_relative_body_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                                            body_names: Optional[list[str]] = None) -> torch.Tensor:
    return torch.exp(-motion_relative_body_position_error(env, command_name, body_names) ** 2 / std ** 2)


def motion_relative_body_velocity_error(env: ManagerBasedRLEnv, command_name: str,
                                        body_names: list[str] | None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    robot_vel_b = quat_apply_yaw(quat_inv(command.robot_ref_pose_w[:, 3:7]).repeat(1, len(body_indexes), 1),
                                 command.robot_body_vel_w[:, body_indexes, :3] - command.robot_ref_vel_w[:, None, :3])
    motion_vel_b = quat_apply_yaw(quat_inv(command.motion_ref_pose_w[:, 3:7]).repeat(1, len(body_indexes), 1),
                                  command.motion_body_vel_w[:, body_indexes, :3] - command.motion_ref_vel_w[:, None,
                                                                                   :3])
    return torch.norm(motion_vel_b - robot_vel_b, dim=-1).mean(-1)


def motion_relative_body_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                                            body_names: Optional[list[str]] = None) -> torch.Tensor:
    return torch.exp(-motion_relative_body_velocity_error(env, command_name, body_names) ** 2 / std ** 2)


def motion_joint_pos_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_joint_pos - command.robot_joint_pos, dim=-1)


def motion_joint_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_joint_pos_error(env, command_name) ** 2 / std ** 2)


def motion_joint_vel_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_joint_vel - command.robot_joint_vel, dim=-1)


def motion_joint_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_joint_vel_error(env, command_name) ** 2 / std ** 2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
