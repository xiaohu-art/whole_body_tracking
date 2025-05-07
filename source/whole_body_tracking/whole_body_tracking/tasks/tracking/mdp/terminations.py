from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def bad_ref_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.motion_ref_pose_w[:, :3] - command.robot_ref_pose_w[:, :3], dim=1) > threshold


def bad_ref_ori(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.motion_ref_pose_w[:, 3:7],
                                                                asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_ref_pose_w[:, 3:7],
                                                               asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bh_joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)
