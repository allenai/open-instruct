"""Submit evaluation jobs using allenai/olmo-eval-internal.

Submits a Beaker v2 experiment that runs `olmo-eval run` against a model. The
Beaker image ships with CUDA and PyTorch; olmo-eval-internal, vllm, and
transformers are installed at job start via INSTALL_SCRIPT to allow testing the
latest code. When `--location` is a Beaker dataset, the model is mounted at
`/model`.

Example:
    uv run python scripts/submit_eval_jobs.py \\
        --model_name qwen3_4b_base_dapo_20260422_083224 \\
        --location 01KPTSPMHGEZVYCDNR0XBVJCGZ \\
        --tasks aime_2025:pass_at_32 \\
        --max_length 8192 \\
        --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale \\
        --priority urgent \\
        --preemptible \\
        --workspace ai2/open-instruct-dev
"""

import argparse
import re
import shlex
import subprocess
from datetime import date

import yaml

from open_instruct import launch_utils


BEAKER_ID_RE = re.compile(r"^[0-9A-Z]{26}$")
DEFAULT_CLUSTERS = ("ai2/jupiter",)
MAX_EXPERIMENT_NAME_LEN = 128
EXPERIMENT_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

DEFAULT_OLMO_EVAL_REF = "main"
GIT_REF_SAFE_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def build_install_script(ref: str) -> str:
    if not GIT_REF_SAFE_RE.match(ref):
        raise ValueError(f"Invalid git ref {ref!r}; expected characters [A-Za-z0-9._/-].")
    return (
        "set -euo pipefail && "
        "git clone "
        "https://x-access-token:${GITHUB_TOKEN}@github.com/allenai/olmo-eval-internal.git "
        "/opt/olmo-eval-internal && "
        f"cd /opt/olmo-eval-internal && git checkout {shlex.quote(ref)} && "
        "uv pip install --cache-dir /weka/oe-eval-default/olmo-eval-pypi-cache -e '.[vllm]' && "
        "uv pip install --cache-dir /weka/oe-eval-default/olmo-eval-pypi-cache "
        "--upgrade 'vllm[runai]>=0.19.0' 'transformers>=5.4.0' && "
        "cd /workspace"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, required=True, help="Human-readable run name.")
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help=(
            "Model location. Accepts: a bare Beaker dataset id (26 uppercase alphanumerics), "
            "'beaker://<id>', an HF repo id (e.g. allenai/OLMo-2-1124-7B-Instruct), "
            "an absolute Weka/NFS path, or a gs:// URL."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="aime_2025:pass_at_32",
        help="Comma-separated olmo-eval task specs. See `olmo-eval tasks`/`olmo-eval suites`.",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--cluster", nargs="+", default=list(DEFAULT_CLUSTERS))
    parser.add_argument("--priority", type=str, default="normal")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--workspace", type=str, default="ai2/tulu-3-results")
    parser.add_argument("--budget", type=str, default="ai2/oe-adapt")
    parser.add_argument(
        "--beaker_image",
        type=str,
        default="ai2-tylerm/olmo-eval-cu1281-trc290-amd64",
        help="Beaker image with olmo-eval installed.",
    )
    parser.add_argument("--revision", type=str, default=None, help="HF revision (git sha/tag).")
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Provider max_model_len. Sampling max_tokens comes from the task definition.",
    )
    parser.add_argument(
        "--sampling_max_tokens",
        type=int,
        default=None,
        help="Override per-task sampling max_tokens (applied via -o max_tokens=N after each -t).",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--olmo_eval_ref",
        type=str,
        default=DEFAULT_OLMO_EVAL_REF,
        help="Git ref (branch/tag/sha) of allenai/olmo-eval-internal to install at job start.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the spec and beaker command, but do not write or submit."
    )
    return parser.parse_args()


def resolve_model_mount(location: str) -> tuple[str, str | None]:
    """Resolve --location into (model_path_in_container, beaker_dataset_id_or_None)."""
    if location.startswith("beaker://"):
        return "/model", location[len("beaker://") :]
    if BEAKER_ID_RE.match(location):
        return "/model", location
    return location, None


def build_inner_cmd(args: argparse.Namespace, model_path: str) -> list[str]:
    cmd = [
        "olmo-eval",
        "run",
        "-m",
        model_path,
        "--harness",
        "default",
        "-o",
        "provider.kind=vllm_server",
        "-o",
        f"provider.max_model_len={args.max_length}",
        "-o",
        "provider.trust_remote_code=true",
    ]
    if args.revision:
        cmd += ["-o", f"provider.revision={args.revision}"]
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        cmd += ["-t", task]
        if args.sampling_max_tokens is not None:
            cmd += ["-o", f"max_tokens={args.sampling_max_tokens}"]
    cmd += ["--num-gpus", str(args.num_gpus)]
    cmd += ["--output-dir", "/results"]
    return cmd


def build_spec(args: argparse.Namespace, inner_cmd: list[str], dataset_id: str | None, experiment_name: str) -> dict:
    non_weka_clusters = [c for c in args.cluster if c not in launch_utils.WEKA_CLUSTERS]
    if non_weka_clusters:
        raise ValueError(
            f"Clusters {non_weka_clusters} do not support Weka mounts required by this script. "
            f"Use one of {launch_utils.WEKA_CLUSTERS}."
        )
    datasets: list[dict] = [
        {"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}},
        {"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}},
        {"mountPath": "/weka/oe-eval-default", "source": {"weka": "oe-eval-default"}},
    ]
    if dataset_id:
        datasets.append({"mountPath": "/model", "source": {"beaker": dataset_id}})

    full_command = f"{build_install_script(args.olmo_eval_ref)} && {shlex.join(inner_cmd)}"

    return {
        "version": "v2",
        "description": experiment_name,
        "budget": args.budget,
        "retry": {"allowedTaskRetries": 2},
        "tasks": [
            {
                "name": experiment_name,
                "image": {"beaker": args.beaker_image},
                "command": ["/bin/bash", "-c"],
                "arguments": [full_command],
                "envVars": [
                    {"name": "HF_TOKEN", "secret": "HF_TOKEN"},
                    {"name": "OPENAI_API_KEY", "secret": "openai_api_key"},
                    {"name": "GITHUB_TOKEN", "secret": "GITHUB_TOKEN"},
                    {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
                ],
                "datasets": datasets,
                "result": {"path": "/results"},
                "resources": {"gpuCount": args.num_gpus},
                "constraints": {"cluster": list(args.cluster)},
                "context": {"priority": args.priority, "preemptible": args.preemptible},
            }
        ],
    }


def main() -> None:
    args = parse_args()
    launch_utils.validate_beaker_workspace(args.workspace)

    model_path, dataset_id = resolve_model_mount(args.location)
    inner_cmd = build_inner_cmd(args, model_path)

    today = date.today().strftime("%m%d%Y")
    raw_name = args.experiment_name or f"olmo_eval_{args.model_name}_{today}"
    experiment_name = EXPERIMENT_NAME_SAFE_RE.sub("_", raw_name)[:MAX_EXPERIMENT_NAME_LEN]
    spec = build_spec(args, inner_cmd, dataset_id, experiment_name)

    print("Inner command:", shlex.join(inner_cmd))

    if args.dry_run:
        print("Dry run; spec:")
        print(yaml.safe_dump(spec, default_flow_style=False, sort_keys=False))
        return

    spec_path = launch_utils.auto_created_spec_path(experiment_name)
    with open(spec_path, "w") as f:
        yaml.safe_dump(spec, f, default_flow_style=False, sort_keys=False)
    print("Spec written to:", spec_path)

    beaker_cmd = ["beaker", "experiment", "create", spec_path, "--workspace", args.workspace]
    print("Running:", shlex.join(beaker_cmd))
    subprocess.run(beaker_cmd, check=True)


if __name__ == "__main__":
    main()
