from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate
from .commands import RealTrajCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



class OpenLoopAction(JointPositionAction):
    cfg: OpenLoopActionCfg
    _asset: Articulation

    def __init__(self, cfg: OpenLoopActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._command_term: RealTrajCommand = env.command_manager.get_term("real_traj")

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor):
        self._processed_actions = self._command_term.action * self._scale + self._offset

@configclass
class OpenLoopActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = OpenLoopAction


class WrenchAction(ActionTerm):
    cfg: WrenchActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: WrenchActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self._asset: Articulation
        self._body_idx = self._asset.find_bodies(cfg.body_name)[0][0]
        self._wrench_b = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._wrench_b

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._wrench_b

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._wrench_b = actions
        self._wrench_b[:, :3] *= self.cfg.force_scale
        self._wrench_b[:, 3:] *= self.cfg.torque_scale
        return actions

    def apply_actions(self) -> None:
        self._asset: Articulation
        self._asset.set_external_force_and_torque(self._wrench_b[:, None, :3], self._wrench_b[:, None, 3:], body_ids=[
            self._body_idx])

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "force_visualizer"):
                self.force_visualizer = VisualizationMarkers(self.cfg.force_visualizer_cfg)
                self.torque_visualizer = VisualizationMarkers(self.cfg.torque_visualizer_cfg)
            # set their visibility to true
            self.force_visualizer.set_visibility(True)
            self.torque_visualizer.set_visibility(True)
        else:
            if hasattr(self, "torque_visualizer"):
                self.force_visualizer.set_visibility(False)
                self.torque_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._asset.is_initialized:
            return
        force_arrow_scale, force_arrow_quat = self._resolve_3d_vector_to_arrow(
            quat_rotate(self._asset.data.body_state_w[:, self._body_idx, 3:7],
                        self._wrench_b[:, :3]) / self.cfg.force_scale)
        torque_arrow_scale, torque_arrow_quat = self._resolve_3d_vector_to_arrow(
            quat_rotate(self._asset.data.body_state_w[:, self._body_idx, 3:7],
                        self._wrench_b[:, 3:]) / self.cfg.torque_scale)
        pos = self._asset.data.body_state_w[:, self._body_idx, :3].clone()
        self.force_visualizer.visualize(
            pos, force_arrow_quat, force_arrow_scale
        )
        self.torque_visualizer.visualize(
            pos, torque_arrow_quat, torque_arrow_scale
        )

    def _resolve_3d_vector_to_arrow(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obtain default scale of the marker
        default_scale = GREEN_ARROW_X_MARKER_CFG.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(vec.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(vec, dim=1)
        # arrow-direction
        heading_angle = torch.atan2(vec[:, 1], vec[:, 0])
        pitch_angle = torch.atan2(-vec[:, 2], torch.norm(vec[:, :2], dim=-1))
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, pitch_angle, heading_angle)

        return arrow_scale, arrow_quat


@configclass
class WrenchActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = WrenchAction

    body_name: str = MISSING

    force_scale: float = MISSING

    torque_scale: float = MISSING

    debug_vis: bool = True

    force_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Action/force"
    )
    torque_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Action/torque"
    )
