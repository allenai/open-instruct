import os
import subprocess

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
