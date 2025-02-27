from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms
from whole_body_tracking.tasks.locomotion.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_ref_position_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return torch.norm(command.motion_ref_pose_w[:, :3] - command.robot_ref_pose_w[:, :3], dim=1)


def motion_ref_position_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    return 1 - torch.tanh(motion_ref_position_error(env, command_name) / std)


def motion_ref_orientation_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return quat_error_magnitude(command.motion_ref_pose_w[:, 3:7], command.robot_ref_pose_w[:, 3:7])


def motion_body_position_error(env: ManagerBasedRLEnv, command_name: str,
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
        command.motion_ref_pose_w[:, :3], command.motion_ref_pose_w[:, 3:7],
        command.robot_body_pose_w[:, body_indexes, :3]
    )
    return torch.norm(relative_pos_motion - relative_pos_robot, dim=1)


def motion_body_position_error_tanh(
        env: ManagerBasedRLEnv, std: float, command_name: str, body_names: Optional[list[str]] = None
) -> torch.Tensor:
    return 1 - torch.tanh(motion_body_position_error(env, command_name, body_names) / std)


def motion_body_orientation_error(env: ManagerBasedRLEnv, command_name: str,
                                  body_names: Optional[list[str]] = None) -> torch.Tensor:  # TODO doesn't work
    command: MotionCommand = env.command_manager.get_term(command_name)

    if body_names is None:
        body_names = command.cfg.body_names
    body_indexes = [i for i, body_name in enumerate(command.cfg.body_names) if
                    body_name in body_names]  # TODO hurt performance?
    return quat_error_magnitude(
        command.motion_ref_pose_w[:, 3:7], command.motion_body_pose_w[:, body_indexes, 3:7]
    )


def motion_joint_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_joint_pos - command.robot_joint_pos, dim=1)

# TODO velocity relative
