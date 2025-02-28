from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms
from whole_body_tracking.tasks.locomotion.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_global_root_position_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return (command.motion_ref_pose_w[:, :3] - command.robot_ref_pose_w[:, :3]).pow(2).mean(-1)

def motion_global_root_height_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return (command.motion_ref_pose_w[:, 2] - command.robot_ref_pose_w[:, 2]).pow(2)


def motion_global_root_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_global_root_position_error(env, command_name) * std)


def motion_global_root_orientation_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return quat_error_magnitude(command.motion_ref_pose_w[:, 3:7], command.robot_ref_pose_w[:, 3:7]).pow(2).mean(-1)


def motion_global_root_lin_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return torch.exp(-((command.motion_ref_vel_w[:, :3] - command.robot_ref_vel_w[:, :3]).pow(2).mean(-1)) * std)


def motion_global_root_ang_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return torch.exp(-((command.motion_ref_vel_w[:, 3:6] - command.robot_ref_vel_w[:, 3:6]).pow(2).mean(-1)) * std)

def motion_global_body_position_error(env: ManagerBasedRLEnv, command_name: str, body_names: list[str] | None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    return (command.motion_body_pose_w[:, body_indexes, :3] - command.robot_body_pose_w[:, body_indexes, :3]).pow(2).mean(-1).mean(-1)


def motion_global_body_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: Optional[list[str]] = None) -> torch.Tensor:
    return torch.exp(-motion_global_body_position_error(env, command_name, body_names) * std)


def motion_global_body_orientation_error(env: ManagerBasedRLEnv, command_name: str, body_names: Optional[list[str]] = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    return quat_error_magnitude(command.motion_body_pose_w[:, body_indexes, 3:7], command.robot_body_pose_w[:, body_indexes, 3:7]).pow(2).mean(-1).mean(-1)

def motion_global_body_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: Optional[list[str]] = None) -> torch.Tensor:
    return torch.exp(-motion_global_body_orientation_error(env, command_name, body_names) * std)

def motion_global_body_lin_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: Optional[list[str]] = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    return torch.exp(-((command.motion_body_vel_w[:, body_indexes, :3] - command.robot_body_vel_w[:, body_indexes, :3]).pow(2).mean(-1).mean(-1)) * std)


def motion_global_body_ang_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: Optional[list[str]] = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    return torch.exp(-((command.motion_body_vel_w[:, body_indexes, 3:6] - command.robot_body_vel_w[:, body_indexes, 3:6]).pow(2).mean(-1).mean(-1)) * std)


def motion_relative_body_position_error(env: ManagerBasedRLEnv, command_name: str,
                                        body_names: list[str] | None) -> torch.Tensor:  # TODO doesn't work
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if
                    body_name in body_names]  # TODO hurt performance?
    relative_pos_motion, _ = subtract_frame_transforms(
        command.motion_ref_pose_w[:, :3], command.motion_ref_pose_w[:, 3:7],
        command.motion_body_pose_w[:, body_indexes, :3]
    )
    relative_pos_robot, _ = subtract_frame_transforms(
        command.robot_ref_pose_w[:, :3], command.robot_ref_pose_w[:, 3:7],
        command.robot_body_pose_w[:, body_indexes, :3]
    )
    return (relative_pos_motion - relative_pos_robot).pow(2).mean(-1).mean(-1)


def motion_relative_body_position_error_exp(
        env: ManagerBasedRLEnv, std: float, command_name: str, body_names: Optional[list[str]] = None
) -> torch.Tensor:
    return torch.exp(-motion_relative_body_position_error(env, command_name, body_names) * std)


def motion_relative_body_orientation_error(env: ManagerBasedRLEnv, command_name: str,
                                           body_names: Optional[list[str]] = None) -> torch.Tensor:  # TODO doesn't work
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if
                    body_name in body_names]  # TODO hurt performance?
    _, relative_ori_motion = subtract_frame_transforms(
        command.motion_ref_pose_w[:, :3], command.motion_ref_pose_w[:, 3:7], None,
        command.motion_body_pose_w[:, body_indexes, 3:7]
    )
    _, relative_ori_robot = subtract_frame_transforms(
        command.robot_ref_pose_w[:, :3], command.robot_ref_pose_w[:, 3:7], None,
        command.robot_body_pose_w[:, body_indexes, 3:7]
    )
    return quat_error_magnitude(relative_ori_motion, relative_ori_robot).pow(2).mean(-1).mean(-1)

def motion_relative_body_lin_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: Optional[list[str]] = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    relative_vel_motion, _ = subtract_frame_transforms(
        command.motion_ref_pose_w[:, :3], command.motion_ref_pose_w[:, 3:7],
        command.robot_body_vel_w[:, body_indexes, 7:10]
    )
    relative_vel_robot, _ = subtract_frame_transforms(
        command.robot_ref_pose_w[:, :3], command.robot_ref_pose_w[:, 3:7],
        command.robot_body_vel_w[:, body_indexes, :3]
    )
    return torch.exp(-((relative_vel_motion - relative_vel_robot).pow(2).mean(-1).mean(-1)) * std)


def motion_relative_body_ang_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float, body_names: Optional[list[str]] = None) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if body_name in body_names]
    relative_ang_vel_motion, _ = subtract_frame_transforms(
        command.motion_ref_pose_w[:, :3], command.motion_ref_pose_w[:, 3:7],
        command.motion_body_vel_w[:, body_indexes, 3:6]
    )
    relative_ang_vel_robot, _ = subtract_frame_transforms(
        command.robot_ref_pose_w[:, :3], command.robot_ref_pose_w[:, 3:7],
        command.robot_body_vel_w[:, body_indexes, 3:6]
    )
    return torch.exp(-((relative_ang_vel_motion - relative_ang_vel_robot).pow(2).mean(-1).mean(-1)) * std)


def motion_joint_pos_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_joint_pos - command.robot_joint_pos, dim=1)


def motion_joint_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_joint_pos_error(env, command_name) / std ** 2)


def motion_joint_vel_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_joint_vel - command.robot_joint_vel, dim=1)


def motion_joint_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-motion_joint_vel_error(env, command_name) / std ** 2)
