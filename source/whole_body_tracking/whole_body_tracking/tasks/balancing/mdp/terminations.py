from __future__ import annotations

import pdb
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def bad_body_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids].squeeze(1)
    return torch.norm(body_vel, dim=1) > threshold
