#!/usr/bin/env python3
"""Download k evenly spaced W&B checkpoints, load each into Isaac‑Lab/RSL‑RL,
and export the policies to ONNX.

You must launch Isaac Sim before running this script.
"""

import argparse
import sys
from pathlib import Path

import cli_args  # isort: skip
from isaaclab.app import AppLauncher

# ───── CLI ─────
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_ckpts", type=int, default=5)
parser.add_argument("--export_dir", type=Path, default=Path("./logs/exported_models"),
                    help="Root directory for ONNX exports")
cli_args.add_rsl_rl_args(parser)  # adds --wandb_path (required)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.num_envs = True, 1


# ───── W&B helpers ─────

def fetch_run(spec: str):
    import wandb
    api = wandb.Api()
    if spec.endswith(".pt"):
        spec = "/".join(spec.split("/")[:-1])
    try:
        return api.run(spec)
    except wandb.errors.CommError as e:
        sys.exit(f"wandb: {e}")


def list_sorted_ckpts(run):
    files = [f.name for f in run.files()
             if f.name.startswith("model_") and f.name.endswith(".pt")]
    return sorted(files, key=lambda n: int(n.split("_")[1].split(".")[0]))


def pick_evenly_spaced(names, k):  # keep user logic unchanged
    n = len(names)
    if k >= n:
        return names
    return [names[round(i * (n - 1) / k)] for i in range(k + 1)][1:]


def download(run, names, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        print(f"↓ {n}")
        run.file(n).download(str(dest), replace=True)


# ───── main ─────

def main():
    if not args.wandb_path:
        sys.exit("--wandb_path required (entity/project/run_id)")

    run = fetch_run(args.wandb_path)
    ckpts = list_sorted_ckpts(run)
    if not ckpts:
        sys.exit("No checkpoints found.")
    sel = pick_evenly_spaced(ckpts, args.num_ckpts)

    ckpt_root = Path("/tmp/wandb_ckpts") / run.id
    download(run, sel, ckpt_root)

    # launch Isaac Sim
    AppLauncher(args)

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx)
    from rsl_rl.runners import OnPolicyRunner
    import whole_body_tracking.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)

    # create export destination root once
    run_export_root = (args.export_dir / run.id).resolve()
    run_export_root.mkdir(parents=True, exist_ok=True)

    for ckpt in sel:
        ckpt_path = ckpt_root / ckpt
        print(f"Loading {ckpt_path}")
        runner.load(str(ckpt_path))

        onnx_name = ckpt.replace(".pt", ".onnx")
        export_policy_as_onnx(
            runner.alg.actor_critic,
            normalizer=runner.obs_normalizer,
            path=str(run_export_root),
            filename=onnx_name,
        )
        print(f"→ exported to {run_export_root / onnx_name}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
