import functools
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import beaker
import requests
from dateutil import parser
from rich.pretty import pprint

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

WEKA_CLUSTERS = ["ai2/jupiter", "ai2/saturn", "ai2/titan", "ai2/neptune", "ai2/ceres", "ai2/triton", "ai2/rhea"]
GCP_CLUSTERS = ["ai2/augusta"]
INTERCONNECT_CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/titan", "ai2/augusta"]


def live_subprocess_output(cmd: list[str]) -> str:
    output_lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(line.strip())
            output_lines.append(line.strip())
    process.wait()
    if process.returncode != 0:
        process_error = process.stderr.read() if process.stderr else "No error message available"
        error_message = f"gsutil command failed with return code {process.returncode}: {process_error}"
        print(error_message)
        raise Exception(error_message)

    return "\n".join(output_lines)


def download_from_gs_bucket(src_paths: list[str], dest_path: str) -> None:
    os.makedirs(dest_path, exist_ok=True)
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_thread_count=1",
        "-o",
        "GSUtil:sliced_object_download_threshold=150",
        "-m",
        "cp",
        "-r",
    ]
    cmd.extend(src_paths)
    cmd.append(dest_path)
    print(f"Downloading from GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def download_from_hf(model_name_or_path: str, revision: str) -> None:
    cmd = ["huggingface-cli", "download", model_name_or_path, "--revision", revision]
    print(f"Downloading from HF with command: {cmd}")
    output = live_subprocess_output(cmd)
    if "\n" in output:
        output = output.split("\n")[-1].strip()
    return output


def gs_folder_exists(path: str) -> bool:
    cmd = ["gsutil", "ls", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode == 0


def upload_to_gs_bucket(src_path: str, dest_path: str) -> None:
    cmd = ["gsutil", "-o", "GSUtil:parallel_composite_upload_threshold=150M", "cp", "-r", src_path, dest_path]
    print(f"Copying model to GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def sync_gs_bucket(src_path: str, dest_path: str) -> None:
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_composite_upload_threshold=150M",
        "-m",
        "rsync",
        "-r",
        "-d",
        src_path,
        dest_path,
    ]
    print(f"Copying model to GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def download_latest_checkpoint_from_gs(gs_checkpoint_state_dir: str, checkpoint_state_dir: str) -> None:
    """Download the latest checkpoint from GCS and update the latest file."""
    if gs_folder_exists(gs_checkpoint_state_dir):
        os.makedirs(checkpoint_state_dir, exist_ok=True)
        print(f"Downloading model checkpoint from GCS to {checkpoint_state_dir}")
        sync_gs_bucket(gs_checkpoint_state_dir, checkpoint_state_dir)


@dataclass
class BeakerRuntimeConfig:
    beaker_workload_id: str
    beaker_node_hostname: list[str] | None = None
    beaker_experiment_url: list[str] | None = None
    beaker_dataset_ids: list[str] | None = None
    beaker_dataset_id_urls: list[str] | None = None


def is_beaker_job() -> bool:
    return "BEAKER_JOB_ID" in os.environ


def get_beaker_experiment_info(experiment_id: str) -> dict | None:
    get_experiment_command = f"beaker experiment get {experiment_id} --format json"
    process = subprocess.Popen(["bash", "-c", get_experiment_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Failed to get Beaker experiment: {stderr}")
        return None
    return json.loads(stdout)[0]


def beaker_experiment_succeeded(experiment_id: str) -> bool:
    experiment = get_beaker_experiment_info(experiment_id)
    num_replicas = experiment["jobs"][0]["execution"]["spec"].get("replicas", 1)
    if not experiment:
        return False
    pprint(experiment)
    finalizeds = [
        "finalized" in job["status"] and "exitCode" in job["status"] and job["status"]["exitCode"] == 0
        for job in experiment["jobs"]
    ]
    pprint(finalizeds)
    return sum(finalizeds) == num_replicas


@dataclass
class DatasetInfo:
    id: str
    committed: Any
    non_empty: bool


def get_beaker_dataset_ids(experiment_id: str, sort=False) -> list[str] | None:
    """if sort is True, the non-empty latest dataset will be availble at the end of the list"""
    experiment = get_beaker_experiment_info(experiment_id)
    if not experiment:
        return None
    result_ids = [job["result"]["beaker"] for job in experiment["jobs"]]
    dataset_infos = []
    for result_id in result_ids:
        get_dataset_command = f"beaker dataset get {result_id} --format json"
        process = subprocess.Popen(["bash", "-c", get_dataset_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Failed to get Beaker dataset: {stderr}")
            return None
        datasets = json.loads(stdout)
        dataset_infos.extend(
            [
                DatasetInfo(
                    id=dataset["id"],
                    committed=dataset["committed"],
                    non_empty=(
                        False if dataset["storage"]["totalSize"] is None else dataset["storage"]["totalSize"] > 0
                    ),
                )
                for dataset in datasets
            ]
        )
    if sort:
        dataset_infos.sort(key=lambda x: (x.non_empty, parser.parse(x.committed)))
    pprint(dataset_infos)
    return [dataset.id for dataset in dataset_infos]


@functools.lru_cache(maxsize=1)
def get_beaker_whoami() -> str | None:
    get_beaker_whoami_command = "beaker account whoami --format json"
    process = subprocess.Popen(
        ["bash", "-c", get_beaker_whoami_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Failed to get Beaker account: {stderr}")
        return None
    accounts = json.loads(stdout)
    return accounts[0]["name"]


def maybe_get_beaker_config():
    beaker_dataset_ids = get_beaker_dataset_ids(os.environ["BEAKER_WORKLOAD_ID"])
    if beaker_dataset_ids is None:
        beaker_dataset_id_urls = []
    else:
        beaker_dataset_id_urls = [f"https://beaker.org/ds/{dataset_id}" for dataset_id in beaker_dataset_ids]
    return BeakerRuntimeConfig(
        beaker_workload_id=os.environ["BEAKER_WORKLOAD_ID"],
        beaker_node_hostname=os.environ["BEAKER_NODE_HOSTNAME"],
        beaker_experiment_url=f"https://beaker.org/ex/{os.environ['BEAKER_WORKLOAD_ID']}/",
        beaker_dataset_ids=get_beaker_dataset_ids(os.environ["BEAKER_WORKLOAD_ID"]),
        beaker_dataset_id_urls=beaker_dataset_id_urls,
    )


def format_eta(seconds: float) -> str:
    """Format ETA in a human-readable format."""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def maybe_update_beaker_description(
    current_step: int | None = None,
    total_steps: int | None = None,
    start_time: float | None = None,
    wandb_url: str | None = None,
    original_descriptions: dict[str, str] = {},  # noqa: B006
) -> None:
    """Update Beaker experiment description with training progress and/or wandb URL.

    Args:
        current_step: Current training step (for progress tracking)
        total_steps: Total number of training steps (for progress tracking)
        start_time: Training start time (from time.time()) (for progress tracking)
        wandb_url: Optional wandb URL to include
        original_descriptions: Cache of original descriptions for progress updates
    """
    if not is_beaker_job():
        return

    experiment_id = os.environ.get("BEAKER_WORKLOAD_ID")
    if not experiment_id:
        logger.warning(
            f"BEAKER_WORKLOAD_ID not found in environment. Available env vars: {', '.join(sorted([k for k in os.environ if 'BEAKER' in k]))}"
        )
        return

    try:
        client = beaker.Beaker.from_env()
    except beaker.exceptions.BeakerConfigurationError as e:
        logger.warning(f"Failed to initialize Beaker client: {e}")
        return

    try:
        workload = client.workload.get(experiment_id)
        spec = client.experiment.get_spec(workload)
    except (beaker.exceptions.BeakerExperimentNotFound, ValueError):
        logger.warning(
            f"Failed to get Beaker experiment with ID: {experiment_id}"
            "This might be fine if you are e.g. running in an interactive job."
        )
        return

    if experiment_id not in original_descriptions:
        raw_description = spec.description or ""
        if "git_commit:" in raw_description:
            raw_description = raw_description.split("git_commit:")[0].strip()
        original_descriptions[experiment_id] = raw_description

    description_components = [
        original_descriptions[experiment_id],
        f"git_commit: {os.environ.get('GIT_COMMIT', 'unknown')}",
        f"git_branch: {os.environ.get('GIT_BRANCH', 'unknown')}",
    ]

    if wandb_url:
        description_components.append(wandb_url)

    if current_step is not None:
        progress_pct = (current_step / total_steps) * 100
        elapsed_time = time.perf_counter() - start_time

        if current_step >= total_steps:
            time_str = format_eta(elapsed_time)
            time_label = "finished in"
        else:
            if current_step > 0:
                time_per_step = elapsed_time / current_step
                remaining_steps = total_steps - current_step
                eta_seconds = time_per_step * remaining_steps
                time_str = format_eta(eta_seconds)
            else:
                time_str = "calculating..."
            time_label = "eta"

        progress_bar = f"[{progress_pct:.1f}% complete (step {current_step}/{total_steps}), {time_label} {time_str}]"
        description_components.append(progress_bar)
    new_description = " ".join(description_components)
    try:
        client.workload.update(workload, description=new_description)
    except requests.exceptions.HTTPError as e:
        logger.warning(
            f"Failed to update Beaker description due to HTTP error: {e}"
            "Continuing without updating description - this is likely a temporary Beaker service issue"
        )


def get_beaker_experiment_url() -> str | None:
    """If the env var BEAKER_WORKLOAD_ID is set, gets the current experiment URL."""
    try:
        beaker_client = beaker.Beaker.from_env()
        workload = beaker_client.workload.get(os.environ["BEAKER_WORKLOAD_ID"])
        url = beaker_client.experiment.url(workload.experiment)
        return url
    except Exception:
        return None


def wandb_url_to_run_path(url: str) -> str:
    """
    Convert a wandb URL to a wandb run path.

    Args:
        url (str): wandb URL in format https://wandb.ai/entity/project/runs/run_id

    Returns:
        str: wandb run path in format entity/project/run_id

    >>> wandb_url_to_run_path("https://wandb.ai/org/project/runs/runid")
    org/project/runid

    >>> wandb_url_to_run_path("https://wandb.ai/ai2-llm/open_instruct_internal/runs/5nigq0mz")
    ai2-llm/open_instruct_internal/5nigq0mz
    """
    path_parts = url.replace("https://wandb.ai/", "").split("/")
    entity = path_parts[0]
    project = path_parts[1]
    run_id = path_parts[3]  # Skip 'runs' at index 2

    return f"{entity}/{project}/{run_id}"


def launch_ai2_evals_on_weka(
    path: str,
    leaderboard_name: str,
    oe_eval_max_length: int | None = None,
    wandb_url: str | None = None,
    training_step: int | None = None,
    oe_eval_tasks: list[str] | None = None,
    stop_strings: list[str] | None = None,
    gs_bucket_path: str | None = None,
    eval_priority: str | None = "normal",
    eval_workspace: str | None = "ai2/tulu-3-results",
    beaker_image: str | None = None,
    oe_eval_gpu_multiplier: int | None = None,
) -> None:
    beaker_users = get_beaker_whoami()

    if gs_bucket_path is not None:
        cluster_str = f"--cluster {' '.join(GCP_CLUSTERS)}"
        if beaker_users is not None:
            gs_saved_path = f"{gs_bucket_path}/{beaker_users}/{path}"
        else:
            gs_saved_path = f"{gs_bucket_path}/{path}"
        gs_command = f"""gsutil \\
            -o "GSUtil:parallel_composite_upload_threshold=150M" \\
            cp -r {path} \\
            {gs_saved_path}"""
        print(f"Copying model to GS bucket with command: {gs_command}")
        process = subprocess.Popen(["bash", "-c", gs_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"GS bucket copy stdout:\n{stdout.decode()}")
        print(f"GS bucket copy stderr:\n{stderr.decode()}")
        print(f"GS bucket copy process return code: {process.returncode}")

        path = gs_saved_path
    else:
        cluster_str = ""
    command = f"""\
python scripts/submit_eval_jobs.py \
--model_name {leaderboard_name} \
--location {path} {cluster_str} \
--is_tuned \
--workspace {eval_workspace} \
--priority {eval_priority} \
--preemptible \
--use_hf_tokenizer_template \
--run_oe_eval_experiments \
--skip_oi_evals"""
    if wandb_url is not None:
        command += f" --run_id {wandb_url}"
        wandb_run_path = wandb_url_to_run_path(wandb_url)
        command += f" --wandb_run_path {wandb_run_path}"
    if oe_eval_max_length is not None:
        command += f" --oe_eval_max_length {oe_eval_max_length}"
    if training_step is not None:
        command += f" --step {training_step}"
    if gs_bucket_path is None:
        command += " --evaluate_on_weka"
    if oe_eval_tasks is not None:
        command += f" --oe_eval_tasks {','.join(oe_eval_tasks)}"
    if stop_strings is not None:
        command += f" --oe_eval_stop_sequences '{','.join(stop_strings)}'"
    if beaker_image is not None:
        command += f" --beaker_image {beaker_image}"
    if oe_eval_gpu_multiplier is not None:
        command += f" --gpu_multiplier {oe_eval_gpu_multiplier}"
    print(f"Launching eval jobs with command: {command}")
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
    print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
    print(f"Submit jobs after model training is finished - process return code: {process.returncode}")
