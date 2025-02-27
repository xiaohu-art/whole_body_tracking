import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab_assets import HUMANOID_28_CFG
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env_cfg import MOTIONS_DIR
from whole_body_tracking.tasks.locomotion.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class HumanoidFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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

        self.commands.motion.reference_body = "torso"
        self.commands.motion.joint_names = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z',
                                            'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                                            'right_elbow', 'left_shoulder_x', 'left_shoulder_y',
                                            'left_shoulder_z', 'left_elbow', 'right_hip_x', 'right_hip_y',
                                            'right_hip_z', 'right_knee', 'right_ankle_x', 'right_ankle_y',
                                            'right_ankle_z', 'left_hip_x', 'left_hip_y', 'left_hip_z',
                                            'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        self.commands.motion.body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]

@configclass
class HumanoidFlatDanceEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidFlatWalkEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
