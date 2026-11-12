"""Portable adapters for the original training, evaluation, and video commands.

No scheduler jobs are submitted. Use --dry-run to inspect commands without running
training or rollouts. Activate the intended research environment first.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robomimic-root", type=Path, required=True,
                        help="Source checkout containing robomimic/scripts/")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("train", help="Explicitly reproduce training; not run for this submission")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    for mode in ["evaluate", "video"]:
        p = sub.add_parser(mode)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--seed", type=int, default=123)
        if mode == "video":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.robomimic_root.resolve()
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    if args.mode == "train":
        config_path = Path(__file__).resolve().parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["train"]["data"] = [{"path": str(args.dataset.resolve())}]
        config["train"]["output_dir"] = str(args.output_dir.resolve())
        script = root / "robomimic/scripts/train.py"
        command = [sys.executable, "-u", str(script), "--config", "<temporary-portable-config.json>"]
        if args.dry_run:
            print(json.dumps({"cwd": str(root), "command": command, "path_overrides": config["train"]["data"],
                              "output_dir": config["train"]["output_dir"]}, indent=2))
            return
        if not args.dataset.is_file():
            parser.error("Dataset file does not exist")
        if not script.is_file():
            parser.error("Training entry point does not exist")
        with tempfile.TemporaryDirectory(prefix="lift-early-config-") as directory:
            temporary = Path(directory) / "config.json"
            temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            command[-1] = str(temporary)
            subprocess.run(command, cwd=root, env=env, check=True)
        return
    script = root / "robomimic/scripts/run_trained_agent.py"
    command = [sys.executable, "-u", str(script), "--agent", str(args.checkpoint.resolve()),
               "--n_rollouts", "50" if args.mode == "evaluate" else "5", "--horizon", "400",
               "--seed", str(args.seed)]
    if args.mode == "video":
        command.extend(["--video_path", str(args.output.resolve()), "--video_skip", "5",
                        "--camera_names", "agentview"])
    if args.dry_run:
        print(json.dumps({"cwd": str(root), "command": command}, indent=2))
        return
    if not args.checkpoint.is_file() or not script.is_file():
        parser.error("Checkpoint or evaluation entry point does not exist")
    if args.mode == "video":
        if args.output.exists():
            parser.error("Video output already exists; choose another path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
