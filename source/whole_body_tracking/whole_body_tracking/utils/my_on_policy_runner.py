import os

import wandb
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from whole_body_tracking.utils.exporter import export_motion_policy_as_onnx, attach_onnx_metadata


class MyOnPolicyRunner(OnPolicyRunner):

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split('model')[0]
            export_motion_policy_as_onnx(self.env.unwrapped, self.alg.actor_critic, normalizer=self.obs_normalizer,
                                         path=policy_path)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path)
            wandb.save(policy_path + "policy.onnx", base_path=os.path.dirname(policy_path))
