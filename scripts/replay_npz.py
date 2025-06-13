"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the interactive scene interface."
)
parser.add_argument(
    "--motion_file",
    type=str,
    required=True,
    help="The path to the motion file.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Load motion
    motion = MotionLoader(motion_file=args_cli.motion_file, device=sim.device)
    motion_joint_indexes = motion.get_dof_index(robot.joint_names)
    motion_times = np.zeros(scene.num_envs)

    # Simulation loop
    while simulation_app.is_running():
        motion_times += sim_dt
        reset_ids = motion_times > motion.duration
        motion_times[reset_ids] = 0.0

        (
            motion_joint_pos,
            motion_joint_vel,
            motion_body_pos,
            motion_body_rot,
            motion_body_lin_vel,
            motion_body_ang_vel,
        ) = motion.sample(num_samples=scene.num_envs, times=motion_times)

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_body_pos[:, 0]
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_body_rot[:, 0]
        root_states[:, 7:10] = motion_body_lin_vel[:, 0]
        root_states[:, 10:] = motion_body_ang_vel[:, 0]

        joint_pos = motion_joint_pos[:, motion_joint_indexes]
        joint_vel = motion_joint_vel[:, motion_joint_indexes]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
