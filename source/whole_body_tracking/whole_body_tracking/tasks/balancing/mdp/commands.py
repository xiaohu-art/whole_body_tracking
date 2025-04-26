from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg

from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error, quat_apply_yaw, quat_mul, subtract_frame_transforms, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class InvariantUniformPoseCommand(UniformPoseCommand):
    """A pose command generator that generates a fixed pose command."""

    cfg: InvariantUniformPoseCommandCfg

    def __init__(self, cfg: InvariantUniformPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # Define the command sample from super in new frame:
        # only pos_x and pos_y and yaw rotation same as the base frame
        # other are aligned with the world frame
        self.pose_command_w[env_ids, :3] = self.robot.data.root_state_w[env_ids, :3] + quat_apply_yaw(
            self.robot.data.root_state_w[env_ids, 3:7], self.pose_command_b[env_ids, :3]
        )
        self.pose_command_w[env_ids, 2] = self.pose_command_b[env_ids, 2]
        self.pose_command_w[env_ids, 3:7] = quat_mul(
            yaw_quat(self.robot.data.root_state_w[env_ids, 3:7]), self.pose_command_b[env_ids, 3:7]
        )

    def _update_command(self):
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )
        # Clip the position
        self.pose_command_b[:, :3] = torch.clamp(
            self.pose_command_b[:, :3], -self.cfg.position_clip, self.cfg.position_clip
        )


@configclass
class InvariantUniformPoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = InvariantUniformPoseCommand
    position_clip: float = 1.0
