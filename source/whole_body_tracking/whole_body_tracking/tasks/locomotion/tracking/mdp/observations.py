from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import subtract_frame_transforms
from whole_body_tracking.tasks.locomotion.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_ref_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_pose_w[:, 3:7].view(env.num_envs, -1)


def robot_ref_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_vel_w[:, :3].view(env.num_envs, -1)


def robot_ref_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_ref_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_ref_pose_w[:, None, :3].repeat(1, num_bodies, 1),
        command.robot_ref_pose_w[:, None, 3:7].repeat(1, num_bodies, 1),
        command.robot_body_pose_w[:, :, :3], command.robot_body_pose_w[:, :, 3:7],
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_ref_pose_w[:, None, :3].repeat(1, num_bodies, 1),
        command.robot_ref_pose_w[:, None, 3:7].repeat(1, num_bodies, 1),
        command.robot_body_pose_w[:, :, :3], command.robot_body_pose_w[:, :, 3:7],
    )

    return ori_b.view(env.num_envs, -1)
