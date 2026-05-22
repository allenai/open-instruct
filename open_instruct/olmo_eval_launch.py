"""Launch Beaker evaluation jobs via allenai/olmo-eval-internal."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Literal

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

BEAKER_DATASET_ID_RE = re.compile(r"^[0-9A-Z]{26}$")


@dataclass
class OlmoEvalLaunchConfig:
    """Configuration for `olmo-eval beaker launch` jobs triggered from training."""

    try_launch_olmo_eval_jobs_on_weka: bool = False
    """Whether to launch olmo-eval Beaker jobs after saving checkpoints on Weka."""

    olmo_eval_tasks: list[str] | None = None
    """Olmo-eval task specs (e.g. math:posttrain:dev). Required when launching evals."""

    olmo_eval_cluster: str = "h100"
    """Beaker cluster alias or full name."""

    olmo_eval_groups: list[str] | None = None
    """Beaker group(s) for olmo-eval jobs. Defaults to the training experiment name."""

    olmo_eval_priority: Literal["low", "normal", "high", "urgent"] = "urgent"
    """Priority for auto-launched olmo-eval jobs."""

    olmo_eval_workspace: str = "ai2/open-instruct-dev"
    """Beaker workspace for olmo-eval jobs."""

    olmo_eval_budget: str | None = None
    """Beaker budget. Required when the workspace has no bound budget."""

    olmo_eval_name: str | None = None
    """Experiment name passed to olmo-eval. Auto-generated when unset."""

    olmo_eval_harness: str | None = None
    """Optional harness preset for olmo-eval."""

    olmo_eval_harness_overrides: list[str] | None = None
    """Dotlist overrides for the harness preset."""

    olmo_eval_gpus: int | None = None
    """GPU count for eval jobs. None lets olmo-eval auto-detect."""

    olmo_eval_preemptible: bool = True
    """Whether olmo-eval jobs are preemptible."""

    olmo_eval_timeout: str | None = None
    """Job timeout (e.g. 24h). None uses olmo-eval defaults."""

    olmo_eval_beaker_image: str | None = None
    """Optional Beaker image override for olmo-eval jobs."""

    olmo_eval_dry_run: bool = False
    """Print the olmo-eval launch command without submitting."""


def resolve_olmo_eval_model_path(checkpoint_path: str) -> str:
    """Return the model path to pass to `olmo-eval beaker launch -m`.

    Olmo-eval expects a Weka/HF path for checkpoints saved during training. Beaker
    dataset ids from prior jobs are not supported as model specs.
    """
    if BEAKER_DATASET_ID_RE.match(checkpoint_path):
        raise ValueError(
            "Olmo-eval launch requires a checkpoint path on Weka, not a Beaker dataset id. "
            f"Got dataset id {checkpoint_path!r}."
        )
    return checkpoint_path.rstrip("/")


def effective_olmo_eval_groups(config: OlmoEvalLaunchConfig, exp_name: str) -> list[str]:
    """Return Beaker groups for olmo-eval launch, defaulting to the training experiment name."""
    if config.olmo_eval_groups is not None:
        return config.olmo_eval_groups
    return [exp_name]


def build_olmo_eval_beaker_launch_command(
    model_path: str, config: OlmoEvalLaunchConfig, *, exp_name: str, experiment_name: str | None = None
) -> list[str]:
    """Build the `olmo-eval beaker launch` argv for a saved checkpoint."""
    if not config.olmo_eval_tasks:
        raise ValueError("olmo_eval_tasks must be set when launching olmo-eval jobs.")

    resolved_model_path = resolve_olmo_eval_model_path(model_path)
    cmd = [
        "olmo-eval",
        "beaker",
        "launch",
        "-m",
        resolved_model_path,
        "-c",
        config.olmo_eval_cluster,
        "-w",
        config.olmo_eval_workspace,
        "-p",
        config.olmo_eval_priority,
        "--yes",
        "--no-follow",
    ]

    if experiment_name is not None:
        cmd.extend(["-n", experiment_name])
    elif config.olmo_eval_name is not None:
        cmd.extend(["-n", config.olmo_eval_name])

    for task in config.olmo_eval_tasks:
        cmd.extend(["-t", task])

    for group in effective_olmo_eval_groups(config, exp_name):
        cmd.extend(["-g", group])

    if config.olmo_eval_budget is not None:
        cmd.extend(["-B", config.olmo_eval_budget])

    if config.olmo_eval_beaker_image is not None:
        cmd.extend(["-I", config.olmo_eval_beaker_image])

    if config.olmo_eval_harness is not None:
        cmd.extend(["-H", config.olmo_eval_harness])

    for override in config.olmo_eval_harness_overrides or []:
        cmd.extend(["-o", override])

    if config.olmo_eval_gpus is not None:
        cmd.extend(["-G", str(config.olmo_eval_gpus)])

    if config.olmo_eval_timeout is not None:
        cmd.extend(["-T", config.olmo_eval_timeout])

    if config.olmo_eval_preemptible:
        cmd.append("--preemptible")
    else:
        cmd.append("--no-preemptible")

    if config.olmo_eval_dry_run:
        cmd.append("--dry-run")

    return cmd


def launch_olmo_evals_on_weka(
    model_path: str, config: OlmoEvalLaunchConfig, *, exp_name: str, experiment_name: str | None = None
) -> None:
    """Launch olmo-eval Beaker jobs for a checkpoint saved during training."""
    command = build_olmo_eval_beaker_launch_command(
        model_path=model_path, config=config, exp_name=exp_name, experiment_name=experiment_name
    )
    logger.info("Launching olmo-eval jobs with command: %s", shlex.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logger.info(
        "Olmo-eval launch finished (return code %s)\nStdout:\n%s\nStderr:\n%s",
        process.returncode,
        stdout.decode(),
        stderr.decode(),
    )


def default_olmo_eval_experiment_name(leaderboard_name: str, training_step: int | None = None) -> str:
    """Build a short experiment name when one is not configured explicitly."""
    if training_step is not None:
        return f"{leaderboard_name}_step_{training_step}"
    return leaderboard_name
