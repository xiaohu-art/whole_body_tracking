from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from whole_body_tracking.tasks.real2sim.mdp.commands import RealTrajCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def open_loop_action(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: RealTrajCommand = env.command_manager.get_term(command_name)

    return command.action
