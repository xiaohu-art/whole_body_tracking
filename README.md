# BeyondMimic Motion Tracking Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing

Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
      ```bash
      hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --local-dir {local_path}
      ```
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
# example
python scripts/csv_to_npz.py --input_file LAFAN1/g1/dance1_subject1.csv --input_fps 30 --output_dir LAFAN1/g1/output --output_name dance1_subject1 --headless

python scripts/replay_npz.py --motion_file LAFAN1/g1/output/dance1_subject1.npz
```

### Policy Training

- Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_file LAFAN1/g1/output/dance1_subject1.npz \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --motion_file LAFAN1/g1/output/dance1_subject1.npz
```

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.
