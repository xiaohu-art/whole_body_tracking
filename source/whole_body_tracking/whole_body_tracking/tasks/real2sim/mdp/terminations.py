from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.real2sim.mdp.commands import RealTrajCommand


def traj_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)
    return command.end_mask


def bad_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_root_pos"] > threshold


def bad_ori(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.metrics["error_root_rot"] > threshold
