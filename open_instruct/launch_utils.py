import os
import subprocess

AUTO_CREATED_BEAKER_CONFIG_DIR = "configs/beaker_configs/auto_created"

WEKA_CLUSTERS = [
    "ai2/jupiter",
    "ai2/saturn",
    "ai2/titan",
    "ai2/neptune",
    "ai2/ceres",
    "ai2/triton",
    "ai2/rhea",
    "ai2/prometheus",
    "ai2/holmes",
]

INTERCONNECT_CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/titan", "ai2/holmes"]


def live_subprocess_output(cmd: list[str]) -> str:
    output_lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(line.strip())
            output_lines.append(line.strip())
    process.wait()
    if process.returncode != 0:
        full_output = "\n".join(output_lines)
        error_message = f"Command `{' '.join(cmd)}` failed with return code {process.returncode}:\n{full_output}"
        raise Exception(error_message)

    return "\n".join(output_lines)


def download_from_hf(model_name_or_path: str, revision: str) -> str:
    cmd = ["huggingface-cli", "download", model_name_or_path, "--revision", revision]
    print(f"Downloading from HF with command: {cmd}")
    output = live_subprocess_output(cmd)
    if "\n" in output:
        output = output.split("\n")[-1].strip()
    return output


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


def gs_folder_exists(path: str) -> bool:
    cmd = ["gsutil", "ls", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode == 0


def upload_to_gs_bucket(src_path: str, dest_path: str) -> None:
    cmd = ["gsutil", "-o", "GSUtil:parallel_composite_upload_threshold=150M", "cp", "-r", src_path, dest_path]
    print(f"Copying model to GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def validate_beaker_workspace(workspace: str) -> None:
    parts = workspace.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"--workspace must be fully qualified as '<org>/<workspace>' (e.g., 'ai2/oe-adapt-general'). Received: '{workspace}'"
        )


def auto_created_spec_path(experiment_name: str) -> str:
    os.makedirs(AUTO_CREATED_BEAKER_CONFIG_DIR, exist_ok=True)
    return os.path.join(AUTO_CREATED_BEAKER_CONFIG_DIR, f"{experiment_name}.yaml")
