"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import numpy as np
import torch

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
    assert os.path.isfile(args_cli.motion_file), f"Invalid file path: {args_cli.motion_file}"
    data = np.load(args_cli.motion_file)

    joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=sim.device)
    joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=sim.device)
    body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=sim.device)
    body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=sim.device)
    body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=sim.device)
    body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=sim.device)
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    time_step_total = joint_pos.shape[0]

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos[time_steps], joint_vel[time_steps])
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
