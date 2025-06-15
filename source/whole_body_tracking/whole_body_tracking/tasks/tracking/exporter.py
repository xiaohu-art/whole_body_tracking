# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_motion_policy_as_onnx(
        motions: dict, actor_critic: object, path: str, normalizer: object | None = None, filename="policy.onnx",
        verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(motions, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):

    def __init__(self, motions, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        self.joint_pos = motions["joint_pos"].to("cpu")
        self.joint_vel = motions["joint_vel"].to("cpu")
        self.body_pos_w = motions["body_pos_w"].to("cpu")
        self.body_quat_w = motions["body_quat_w"].to("cpu")
        self.body_lin_vel_w = motions["body_lin_vel_w"].to("cpu")
        self.body_ang_vel_w = motions["body_ang_vel_w"].to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return self.actor(self.normalizer(x)), self.joint_pos[time_step_clamped], self.joint_vel[time_step_clamped], \
        self.body_pos_w[
            time_step_clamped], self.body_quat_w[time_step_clamped], self.body_lin_vel_w[time_step_clamped], \
        self.body_ang_vel_w[time_step_clamped]

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                          "body_ang_vel_w"],
            dynamic_axes={},
        )
