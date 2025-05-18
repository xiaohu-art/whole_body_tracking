from __future__ import annotations

from dataclasses import MISSING
import math

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import whole_body_tracking.tasks.tracking.mdp as mdp


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


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(asset_name="robot",
                                  resampling_time_range=(1.0e9, 1.0e9), debug_vis=True,
                                  pose_range={"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.05, 0.05),
                                              "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.78, 0.78)},
                                  velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                                                  "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52),
                                                  "yaw": (-0.78, 0.78)})


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_pos_b = ObsTerm(func=mdp.motion_ref_pos_b, params={"command_name": "motion"},
                               noise=Unoise(n_min=-0.05, n_max=0.05))
        robot_ori_w = ObsTerm(func=mdp.robot_ref_ori_w, params={"command_name": "motion"},
                            noise=Unoise(n_min=-0.05, n_max=0.05))
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"},
                           noise=Unoise(n_min=-0.05, n_max=0.05))
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"},
                           noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_pos_b = ObsTerm(func=mdp.motion_ref_pos_b, params={"command_name": "motion"})
        motion_ori_b = ObsTerm(func=mdp.motion_ref_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.5, 1.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                                   "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52),
                                   "yaw": (-0.78, 0.78)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    motion_global_root_pos = RewTerm(
        func=mdp.motion_global_ref_position_error_exp, weight=0.5,
        params={"command_name": "motion", "std": math.sqrt(0.25)},
    )
    motion_global_root_ori = RewTerm(
        func=mdp.motion_global_ref_orientation_error_exp, weight=0.3,
        params={"command_name": "motion", "std": math.sqrt(0.5)},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp, weight=0.5,
        params={"command_name": "motion", "std": math.sqrt(0.25)},
    )
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_pos_error, weight=-1e-0, params={"command_name": "motion"},
    )
    motion_joint_vel = RewTerm(
        func=mdp.motion_joint_vel_error, weight=-1e-1, params={"command_name": "motion"},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*_joint", ".*waist_.*_joint"])},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ref_pos = DoneTerm(
        func=mdp.bad_ref_pos,
        params={"command_name": "motion", "threshold": 0.5},
    )
    ref_ori = DoneTerm(
        func=mdp.bad_ref_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
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
