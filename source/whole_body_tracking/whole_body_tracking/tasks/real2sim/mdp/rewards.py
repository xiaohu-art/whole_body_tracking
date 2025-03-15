from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude
from whole_body_tracking.tasks.real2sim.mdp.commands import RealTrajCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_pos_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_root_pos"]


def root_pos_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    error = root_pos_error(env, command_name)
    return torch.exp(-error ** 2 / std ** 2)


def root_ori_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_root_rot"]


def root_ori_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    error = root_ori_error(env, command_name)
    return torch.exp(-error ** 2 / std ** 2)


def root_lin_vel_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_root_lin_vel"]


def root_lin_vel_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    error = root_lin_vel_error(env, command_name)
    return torch.exp(-error ** 2 / std ** 2)


def root_ang_vel_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)
    return command.metrics["error_root_ang_vel"]


def root_ang_vel_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    error = root_ang_vel_error(env, command_name)
    return torch.exp(-error ** 2 / std ** 2)


def joint_pos_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_joint_pos"]


def joint_pos_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    error = joint_pos_error(env, command_name)
    return torch.exp(-error ** 2 / std ** 2)


def joint_vel_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_joint_vel"]


def joint_vel_exp(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    error = joint_vel_error(env, command_name)
    return torch.exp(-error ** 2 / std ** 2)
