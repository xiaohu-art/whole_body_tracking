from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedReal2SimAction(ActionTerm):
    cfg: PreTrainedReal2SimActionCfg

    def __init__(self, cfg: PreTrainedReal2SimActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation
        self._env = env

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._real2sim_action_term: ActionTerm = cfg.real2sim_actions.class_type(cfg.real2sim_actions, env)
        self.real2sim_actions = torch.zeros(self.num_envs, self._real2sim_action_term.action_dim, device=self.device)

        # Change the action term to use the last action from the control policy.
        # Unable to use mdp.last_action since during the initialization of the action term (call func to get the shape
        # of the action) the 'ManagerBasedRLEnv' object has no attribute 'action_manager'.
        self.control_actions = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        cfg.real2sim_observations.actions.func = lambda dummy_env: self.control_actions
        cfg.real2sim_observations.actions.params = dict()
        self._real2sim_obs_manager = ObservationManager({"real2sim_policy": cfg.real2sim_observations}, env)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        pass

    def apply_actions(self):
        self.control_actions = self._env.action_manager.action
        real2sim_obs = self._real2sim_obs_manager.compute_group("real2sim_policy")
        self.real2sim_actions[:] = self.policy(real2sim_obs)
        self._real2sim_action_term.process_actions(self.real2sim_actions)
        self._real2sim_action_term.apply_actions()


@configclass
class PreTrainedReal2SimActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PreTrainedReal2SimAction

    policy_path: str = MISSING

    real2sim_actions: ActionTermCfg = MISSING

    real2sim_observations: ObservationGroupCfg = MISSING
