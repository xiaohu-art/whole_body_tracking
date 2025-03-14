from isaaclab.utils import configclass
from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class HumanoidFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.commands.motion.reference_body = "torso_link"
        self.commands.motion.joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
                                            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                                            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                                            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
                                            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
                                            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                                            'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
                                            'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                                            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
                                            'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
                                            'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        # self.commands.motion.body_names = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link',
        #                                    'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link',
        #                                    'left_ankle_roll_link',
        #                                    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
        #                                    'right_knee_link',
        #                                    'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link',
        #                                    'waist_roll_link', 'torso_link',
        #                                    'left_shoulder_pitch_link', 'left_shoulder_roll_link',
        #                                    'left_shoulder_yaw_link', 'left_elbow_link',
        #                                    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
        #                                    'right_shoulder_pitch_link',
        #                                    'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',
        #                                    'right_wrist_roll_link',
        #                                    'right_wrist_pitch_link', 'right_wrist_yaw_link']

        self.commands.motion.body_names = ['pelvis',
                                           'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
                                           'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
                                           'torso_link',
                                           'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
                                           'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']


@configclass
class G1FlatWalkEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/g1/motions/lafan_walk_short.npz"


@configclass
class G1FlatDanceEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/g1/motions/lafan_dance_short.npz"


@configclass
class G1FlatJumpEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/g1/motions/lafan_jump_short.npz"


@configclass
class G1FlatMoonWalkEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/g1/motions/lafan_moonwalk.npz"
