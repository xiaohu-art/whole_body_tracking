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
        self.robot_joint_indexes = self.robot.find_joints(cfg.joint_names, preserve_order=True)

        self.motion_times = np.zeros(self.num_envs)
        self.motion_offset_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.motion_body_pos_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.motion_body_rot_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.motion_body_lin_vel_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.motion_body_ang_vel_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.motion_joint_pos = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)
        self.motion_joint_vel = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _update_metrics(self):
        pass

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
        self.motion_offset_pos[env_ids] = (self.robot.data.body_state_w[env_ids, self.motion_ref_body_index, :2]
                                           - body_pos[env_ids, self.robot_ref_body_index, :2])

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

        self.motion_joint_pos = joint_pos[:, self.motion_joint_indexes]
        self.motion_joint_vel = joint_vel[:, self.motion_joint_indexes]

        self.motion_body_pos_w = body_pos[:, self.motion_body_indexes]
        self.motion_body_rot_w = body_rot[:, self.motion_body_indexes]
        self.motion_body_lin_vel_w = body_lin_vel[:, self.motion_body_indexes]
        self.motion_body_ang_vel_w = body_ang_vel[:, self.motion_body_indexes]

        self.motion_body_pos_w[:, :, :2] += self.motion_offset_pos[:, None, :]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_pose_visualizers"):
                self.current_pose_visualizers = []
                self.goal_pose_visualizers = []
                for name in self.cfg.body_names:
                    self.current_pose_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.pose_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/" + name + "_current_pose")))
                    self.goal_pose_visualizers.append(VisualizationMarkers(self.cfg.pose_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/" + name + "_goal_pose")))
            for i in range(len(self.cfg.body_names)):
                self.current_pose_visualizers[i].set_visibility(True)
                self.goal_pose_visualizers[i].set_visibility(True)
        else:
            for i in range(len(self.cfg.body_names)):
                self.current_pose_visualizers[i].set_visibility(False)
                self.goal_pose_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        for i in range(len(self.cfg.body_names)):
            self.goal_pose_visualizers[i].visualize(self.motion_body_pos_w[:, i], self.motion_body_rot_w[:, i])
            self.current_pose_visualizers[i].visualize(self.robot.data.body_state_w[:, self.robot_body_indexes[i], :3],
                                                       self.robot.data.body_state_w[:, self.robot_body_indexes[i], 3:7])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""
    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    reference_body: str = MISSING
    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
