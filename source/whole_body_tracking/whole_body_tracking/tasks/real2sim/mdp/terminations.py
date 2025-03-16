from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.real2sim.mdp.commands import RealTrajCommand
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def traj_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)
    return command.end_mask


def bad_ref_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)
    return command.metrics["error_root_pos"] > threshold


def bad_ref_ori(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: RealTrajCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.root_quat_w,
                                                                asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(asset.data.root_quat_w,
                                                               asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
