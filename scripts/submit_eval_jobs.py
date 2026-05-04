"""Translate legacy --flag_name arguments into `olmo-eval beaker launch`
flags and delegate to scripts/submit_eval_jobs.sh.

For new code, prefer calling submit_eval_jobs.sh directly with olmo-eval flags.
This wrapper exists to keep existing call sites working.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default=None, help="Used as the experiment name (-n).")
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Model path (-m). Host path / HF repo / s3:// / gs://. Beaker dataset ids are not supported.",
    )
    parser.add_argument("--tasks", type=str, default="aime_2025:pass_at_32")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--cluster", nargs="+", default=["h100"])
    parser.add_argument("--priority", type=str, default="normal")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--workspace", type=str, default="ai2/tulu-3-results")
    parser.add_argument("--budget", type=str, default="ai2/oe-adapt")
    parser.add_argument("--beaker_image", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None, help="HF revision (git sha/tag).")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--sampling_max_tokens", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--olmo_eval_ref", type=str, default="main")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_launch_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    name = args.experiment_name or args.model_name
    if name:
        out += ["-n", name]
    out += ["-m", args.location]
    if args.revision:
        out += ["-o", f"provider.revision={args.revision}"]
    out += ["-o", f"provider.max_model_len={args.max_length}"]
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        out += ["-t", task]
        if args.sampling_max_tokens is not None:
            out += ["-o", f"max_tokens={args.sampling_max_tokens}"]
    out += ["--gpus", str(args.num_gpus)]
    for cluster in args.cluster:
        out += ["-c", cluster]
    out += ["-p", args.priority]
    if args.preemptible:
        out += ["--preemptible"]
    out += ["-w", args.workspace, "-B", args.budget]
    if args.beaker_image:
        out += ["-I", args.beaker_image]
    if args.dry_run:
        out += ["-d"]
    return out


def main() -> None:
    args = parse_args()
    launch_args = build_launch_args(args)
    script = Path(__file__).resolve().parent / "submit_eval_jobs.sh"
    env = os.environ.copy()
    env["OLMO_EVAL_REF"] = args.olmo_eval_ref
    cmd = [str(script), *launch_args]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
