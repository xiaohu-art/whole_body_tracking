from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

from isaaclab.utils import configclass


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.body_names = ['pelvis',
                                           'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
                                           'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
                                           'torso_link',
                                           'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
                                           'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']

        self.commands.motion.motion_file = f"{ASSET_DIR}/g1/motions/lafan_walk_short.npz"
