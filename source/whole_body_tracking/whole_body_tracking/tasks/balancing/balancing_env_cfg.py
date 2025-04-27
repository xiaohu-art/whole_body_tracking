from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import whole_body_tracking.tasks.balancing.mdp as mdp


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        )
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    left_foot_pose = mdp.InvariantUniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_ankle_roll_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.InvariantUniformPoseCommandCfg.Ranges(
            pos_x=(-0.3, 0.3),
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 0.6),
            roll=(0, 0),
            pitch=(0, 0),
            yaw=(0, 0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_foot_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    left_foot_position = RewTerm(func=mdp.position_command_error, weight=-1.0,
                                 params={"command_name": "left_foot_pose",
                                         "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link")})
    left_foot_air_time = RewTerm(func=mdp.feet_air_time, weight=0.1, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
        "threshold": 0.5,
    })

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!left_(?:hip|knee|ankle)).*_joint$"])}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3,
    #                        params={"asset_cfg": SceneEntityCfg("robot",
    #                                                            joint_names=["^(?!left_(?:hip|knee|ankle)).*_joint$"])})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3,
                           params={"asset_cfg": SceneEntityCfg("robot")})

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    bad_body_vel = DoneTerm(func=mdp.bad_body_vel,
                            params={"threshold": 1.0,
                                    "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link")})

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.52,
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    bad_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                             body_names=["torso_link", "left_hip_roll_link", "right_hip_roll_link",
                                                         "left_wrist_yaw_link", "right_wrist_yaw_link"]),
                "threshold": 1.0},
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class BalancingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
