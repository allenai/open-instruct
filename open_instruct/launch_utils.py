import os
import subprocess

from transformers.utils import hub as transformers_hub

WEKA_CLUSTERS = ["ai2/jupiter", "ai2/saturn", "ai2/titan", "ai2/neptune", "ai2/ceres", "ai2/triton", "ai2/rhea"]


def custom_cached_file(model_name_or_path: str, filename: str, revision: str | None = None, repo_type: str = "model"):
    if os.path.isdir(model_name_or_path):
        resolved_file = os.path.join(model_name_or_path, filename)
        if os.path.isfile(resolved_file):
            return resolved_file
        else:
            return None
    else:
        resolved_file = transformers_hub.try_to_load_from_cache(
            model_name_or_path,
            filename,
            cache_dir=transformers_hub.TRANSFORMERS_CACHE,
            revision=revision,
            repo_type=repo_type,
        )
        if not isinstance(resolved_file, str):
            return None
        return resolved_file


def get_commit_hash(
    model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"
) -> str:
    file = custom_cached_file(model_name_or_path, filename, revision=revision, repo_type=repo_type)
    commit_hash = transformers_hub.extract_commit_hash(file, None)
    return commit_hash


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
