"""This script replays specific motion keys from a processed pkl file.

.. code-block:: bash

    # Usage
    python replay_pkl.py --input_file motions.pkl --motion_key dance1_subject1
    python replay_pkl.py --input_file motions.pkl --motion_key dance1_subject1 --loop
    python replay_pkl.py --input_file motions.pkl --list_motions
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
import torch
import joblib

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay specific motion keys from processed pkl file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the processed motion pkl file.")
parser.add_argument("--motion_key", type=str, help="Specific motion key to replay. If not provided, will list available motions.")
parser.add_argument("--loop", action="store_true", help="Loop the motion continuously.")
parser.add_argument("--list_motions", action="store_true", help="List all available motion keys in the pkl file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
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

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class PKLMotionLoader:
    """Motion loader for processed pkl files."""
    
    def __init__(self, motion_data: dict, device: torch.device):
        """Initialize the motion loader with processed motion data.
        
        Args:
            motion_data: Dictionary containing processed motion data from pkl file
            device: Device to load tensors on
        """
        self.device = device
        self.fps = motion_data["fps"][0]  # fps is stored as a list with one element
        
        # Load motion data as tensors
        self.joint_pos = torch.tensor(motion_data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(motion_data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(motion_data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(motion_data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(motion_data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(motion_data["body_ang_vel_w"], dtype=torch.float32, device=device)
        
        self.time_step_total = self.joint_pos.shape[0]
        self.current_step = 0
        
        print(f"Motion loaded: {self.time_step_total} frames at {self.fps} fps")


def list_available_motions(pkl_file: str):
    """List all available motion keys in the pkl file."""
    if not os.path.isfile(pkl_file):
        print(f"Error: File {pkl_file} not found.")
        return
    
    try:
        all_motions = joblib.load(pkl_file)
        print(f"\nAvailable motion keys in {pkl_file}:")
        print("-" * 50)
        for i, key in enumerate(sorted(all_motions.keys()), 1):
            motion_data = all_motions[key]
            fps = motion_data.get("fps", ["unknown"])[0]
            frames = motion_data.get("joint_pos", np.array([])).shape[0] if "joint_pos" in motion_data else "unknown"
            print(f"{i:2d}. {key} (fps: {fps}, frames: {frames})")
        print("-" * 50)
        print(f"Total: {len(all_motions)} motions")
    except Exception as e:
        print(f"Error loading pkl file: {e}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_key: str):
    """Run the simulation loop for the specified motion."""
    # Load the pkl file
    print(f"[INFO]: Loading motions from {args_cli.input_file}")
    all_motions = joblib.load(args_cli.input_file)
    
    if motion_key not in all_motions:
        print(f"[ERROR]: Motion key '{motion_key}' not found in pkl file.")
        print("Available motion keys:")
        for key in sorted(all_motions.keys()):
            print(f"  - {key}")
        return
    
    # Load the specific motion
    motion_data = all_motions[motion_key]
    motion = PKLMotionLoader(motion_data, sim.device)
    
    # Extract scene entities
    robot: Articulation = scene["robot"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    print(f"[INFO]: Replaying motion '{motion_key}' ({motion.time_step_total} frames)")
    if args_cli.loop:
        print("[INFO]: Motion will loop continuously")
    
    # Simulation loop
    while simulation_app.is_running():
        # Check if we need to reset/loop
        if motion.current_step >= motion.time_step_total:
            if args_cli.loop:
                motion.current_step = 0
                print(f"[INFO]: Looping motion '{motion_key}'")
            else:
                print(f"[INFO]: Motion '{motion_key}' completed")
                break
        
        # Get current motion state
        current_step = motion.current_step
        
        # Set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[current_step][0:1] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[current_step][0:1]
        root_states[:, 7:10] = motion.body_lin_vel_w[current_step][0:1]
        root_states[:, 10:] = motion.body_ang_vel_w[current_step][0:1]
        
        robot.write_root_state_to_sim(root_states)
        
        # Set joint state
        robot.write_joint_state_to_sim(
            motion.joint_pos[current_step:current_step+1], 
            motion.joint_vel[current_step:current_step+1]
        )
        
        # Update scene and render
        scene.write_data_to_sim()
        sim.render()  # We don't want physics (sim.step())
        scene.update(sim_dt)
        
        # Update camera view
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        
        # Advance to next frame
        motion.current_step += 1


def main():
    """Main function."""
    # Check if we should list motions
    if args_cli.list_motions:
        list_available_motions(args_cli.input_file)
        return
    
    # Check if motion key is provided
    if not args_cli.motion_key:
        print("Error: --motion_key is required when not using --list_motions")
        print("Use --list_motions to see available motion keys")
        return
    
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02  # 50 fps
    sim = SimulationContext(sim_cfg)
    
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene, args_cli.motion_key)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
