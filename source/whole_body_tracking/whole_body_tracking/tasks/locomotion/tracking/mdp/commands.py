from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.motion = MotionLoader(motion_file=cfg.motion_file, device=self.device)

        self.motion_ref_body_index = self.motion.get_body_index([cfg.reference_body])[0]
        self.motion_body_indexes = self.motion.get_body_index(cfg.body_names)
        self.motion_joint_indexes = self.motion.get_dof_index(cfg.joint_names)

        self.robot_ref_body_index = self.robot.body_names.index(self.cfg.reference_body)
        self.robot_body_indexes = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self.robot_joint_indexes = self.robot.find_joints(cfg.joint_names, preserve_order=True)[0]

        self.motion_times = np.zeros(self.num_envs)
        self.motion_offset_pos = torch.zeros(self.num_envs, 2, device=self.device)

        self.motion_ref_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.motion_ref_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
        self.motion_body_pose_w = torch.zeros(self.num_envs, len(cfg.body_names), 7, device=self.device)
        self.motion_body_vel_w = torch.zeros(self.num_envs, len(cfg.body_names), 6, device=self.device)
        self.motion_joint_pos = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)
        self.motion_joint_vel = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)

        self.metrics["error_ref_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def robot_ref_pose_w(self) -> torch.Tensor:
        return self.robot.data.body_link_state_w[:, self.robot_ref_body_index, :7]

    @property
    def robot_ref_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_vel_w[:, self.robot_ref_body_index]

    @property
    def robot_body_pose_w(self) -> torch.Tensor:
        return self.robot.data.body_link_state_w[:, self.robot_body_indexes, :7]

    @property
    def robot_body_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_vel_w[:, self.robot_body_indexes]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.robot_joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.robot_joint_indexes]

    def _update_metrics(self):
        self.metrics["error_ref_pos"] = torch.norm(
            self.motion_ref_pose_w[:, :3] - self.robot_ref_pose_w[:, :3], dim=1)
        self.metrics["error_ref_rot"] = torch.norm(
            self.motion_ref_pose_w[:, 3:7] - self.robot_ref_pose_w[:, 3:7], dim=1)
        self.metrics["error_ref_lin_vel"] = torch.norm(
            self.motion_ref_vel_w[:, :3] - self.robot_ref_vel_w[:, :3], dim=1)
        self.metrics["error_ref_ang_vel"] = torch.norm(
            self.motion_ref_vel_w[:, 3:] - self.robot_ref_vel_w[:, 3:], dim=1)

        self.metrics["error_body_pos"] = torch.norm(
            self.motion_body_pose_w[:, :, :3] - self.robot_body_pose_w[:, :, :3], dim=2).mean(dim=1)
        self.metrics["error_body_rot"] = torch.norm(
            self.motion_body_pose_w[:, :, 3:7] - self.robot_body_pose_w[:, :, 3:7], dim=2).mean(dim=1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.motion_joint_pos - self.robot_joint_pos, dim=1)
        self.metrics["error_joint_vel"] = torch.norm(
            self.motion_joint_vel - self.robot_joint_vel, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        self.motion_times[env_ids.cpu()] = self.motion.sample_times(num_samples=len(env_ids))

        (
            _,
            _,
            body_pos,
            _,
            _,
            _,
        ) = self.motion.sample(num_samples=self.num_envs, times=self.motion_times)
        self.motion_offset_pos[env_ids] = (self.robot_ref_pose_w[env_ids, :2]
                                           - body_pos[env_ids, self.motion_ref_body_index, :2])

    def _update_command(self):
        self.motion_times += self._env.step_dt
        (
            joint_pos,
            joint_vel,
            body_pos,
            body_rot,
            body_lin_vel,
            body_ang_vel,
        ) = self.motion.sample(num_samples=self.num_envs, times=self.motion_times)

        self.motion_ref_pose_w[:, :3] = body_pos[:, self.motion_ref_body_index]
        self.motion_ref_pose_w[:, 3:7] = body_rot[:, self.motion_ref_body_index]
        self.motion_ref_vel_w[:, :3] = body_lin_vel[:, self.motion_ref_body_index]
        self.motion_ref_vel_w[:, 3:] = body_ang_vel[:, self.motion_ref_body_index]
        self.motion_ref_pose_w[:, :2] += self.motion_offset_pos[:, :2]

        self.motion_body_pose_w[:, :, :3] = body_pos[:, self.motion_body_indexes]
        self.motion_body_pose_w[:, :, 3:7] = body_rot[:, self.motion_body_indexes]
        self.motion_body_vel_w[:, :, :3] = body_lin_vel[:, self.motion_body_indexes]
        self.motion_body_vel_w[:, :, 3:] = body_ang_vel[:, self.motion_body_indexes]
        self.motion_body_pose_w[:, :, :2] += self.motion_offset_pos[:, None, :]

        self.motion_joint_pos = joint_pos[:, self.motion_joint_indexes]
        self.motion_joint_vel = joint_vel[:, self.motion_joint_indexes]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_ref_visualizer"):
                self.current_ref_visualizer = VisualizationMarkers(
                    self.cfg.ref_visualizer_cfg.replace(prim_path="/Visuals/Command/current/ref"))
                self.goal_ref_visualizer = VisualizationMarkers(
                    self.cfg.ref_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/ref"))

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/" + name)))
                    self.goal_body_visualizers.append(VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/" + name)))

            self.current_ref_visualizer.set_visibility(True)
            self.goal_ref_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            self.current_ref_visualizer.set_visibility(False)
            self.goal_ref_visualizer.set_visibility(False)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(False)
                self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_ref_visualizer.visualize(self.robot_ref_pose_w[:, :3], self.robot_ref_pose_w[:, 3:7])
        self.goal_ref_visualizer.visualize(self.motion_ref_pose_w[:, :3], self.motion_ref_pose_w[:, 3:7])

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pose_w[:, i, :3],
                                                       self.robot_body_pose_w[:, i, 3:7])
            self.goal_body_visualizers[i].visualize(self.motion_body_pose_w[:, i, :3],
                                                    self.motion_body_pose_w[:, i, 3:7])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""
    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    reference_body: str = MISSING
    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    ref_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    ref_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
