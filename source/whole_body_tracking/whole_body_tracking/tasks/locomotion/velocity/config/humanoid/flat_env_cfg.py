from isaaclab.utils import configclass

from whole_body_tracking.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets import HUMANOID_28_CFG
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class HumanoidFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot
        self.scene.robot = HUMANOID_28_CFG.replace(
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    velocity_limit=100.0,
                    stiffness=None,
                    damping=None,
                ),
            },
        )
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.2)

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.rewards.dof_torques_l2 = None
