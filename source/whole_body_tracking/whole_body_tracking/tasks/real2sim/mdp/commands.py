from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass, math
import isaaclab.utils.math as math_utils
from whole_body_tracking.tasks.real2sim.mdp.real_traj_loader import RealTrajLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RealTrajCommand(CommandTerm):
    cfg: RealTrajCommandCfg

    def __init__(self, cfg: RealTrajCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.traj_loader = RealTrajLoader(cfg.traj_path, self.device)

        self.traj_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_offsets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.q = torch.zeros(self.num_envs, self.traj_loader.num_joints + 7, device=self.device)
        self.v = torch.zeros(self.num_envs, self.traj_loader.num_joints + 6, device=self.device)
        self.a = torch.zeros(self.num_envs, self.traj_loader.num_joints, device=self.device)

        self.joint_indexes = []  # Joint order in real traj is different than in IssacLab, but action order is correct
        for name in self.robot.joint_names:
            assert name in self.traj_loader.joint_names, f"Joint {name} not found in the trajectory"
            self.joint_indexes.append(self.traj_loader.joint_names.index(name))

        self.metrics["error_root_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_root_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_root_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_root_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.q[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return math.convert_quat(self.q[:, 3:7], "wxyz")

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.v[:, :3]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        # angular velocity of v is in the local frame
        return math_utils.quat_rotate(self.root_quat_w, self.v[:, 3:6])

    @property
    def joint_pos(self) -> torch.Tensor:
        pos = self.q[:, 7:]
        return pos[:, self.joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        vel = self.v[:, 6:]
        return vel[:, self.joint_indexes]

    @property
    def action(self) -> torch.Tensor:
        return self.a

    def _update_metrics(self):
        self.metrics["error_root_pos"] = torch.norm(self.root_pos_w - self.robot.data.root_pos_w, dim=-1)
        self.metrics["error_root_rot"] = math.quat_error_magnitude(self.root_quat_w, self.robot.data.root_quat_w)
        self.metrics["error_root_lin_vel"] = torch.norm(self.root_lin_vel_w - self.robot.data.root_lin_vel_w, dim=-1)
        self.metrics["error_root_ang_vel"] = torch.norm(self.root_ang_vel_w - self.robot.data.root_ang_vel_w, dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot.data.joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot.data.joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        self.traj_ids[env_ids], self.frame_offsets[env_ids] = self.traj_loader.sample_indexes(len(env_ids))

        self.q[env_ids], self.v[env_ids], self.a[env_ids] = (
            self.traj_loader.sample_frame(self.traj_ids[env_ids], self.frame_offsets[env_ids]))

        self.robot.write_root_pose_to_sim(torch.cat([self.root_pos_w, self.root_quat_w], dim=-1)[env_ids], env_ids)
        self.robot.write_root_velocity_to_sim(torch.cat([self.root_lin_vel_w, self.root_ang_vel_w], dim=-1)[env_ids],
                                              env_ids)
        self.robot.write_joint_state_to_sim(self.joint_pos[env_ids], self.joint_vel[env_ids], env_ids=env_ids)

    def _update_command(self):
        self.frame_offsets += 1
        mask = self.traj_loader.get_resample_mask(self.traj_ids, self.frame_offsets)
        self._resample_command(torch.where(mask)[0])
        env_ids = torch.where(~mask)[0]
        self.q[env_ids], self.v[env_ids], self.a[env_ids] = (
            self.traj_loader.sample_frame(self.traj_ids[env_ids], self.frame_offsets[env_ids]))


@configclass
class RealTrajCommandCfg(CommandTermCfg):
    class_type: type = RealTrajCommand

    asset_name: str = MISSING

    traj_path: str = MISSING
