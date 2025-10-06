import argparse
import functools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def is_checkpoint_folder(directory: str, folder_name: str) -> bool:
    """
    Checks if a given folder name corresponds to a valid checkpoint directory.
    """
    is_dir = os.path.isdir(os.path.join(directory, folder_name))
    is_step = folder_name.startswith("step_") and folder_name.split("_")[-1].isdigit()
    is_epoch = folder_name.startswith("epoch_") and folder_name.split("_")[-1].isdigit()
    return is_dir and (is_step or is_epoch)


def retry_on_exception(max_attempts=4, delay=1, backoff=2):
    """
    Retry a function on exception. Useful for HF API calls that may fail due to
    network issues.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            local_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed with error: {e}. Retrying in {local_delay} seconds...")
                    time.sleep(local_delay)
                    local_delay *= backoff
            return None

        return wrapper

    return decorator


@retry_on_exception()
def push_folder_to_hub(
    output_dir: str,
    hf_repo_id: str,
    hf_repo_revision: str,
    private: bool = True,
):
    """
    Uploads a folder to the Hugging Face Hub, creating a new revision.
    """
    hf_repo_url = f"https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}"
    api = HfApi()
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    if hf_repo_revision is not None:
        api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=hf_repo_revision,
        folder_path=output_dir,
        commit_message=f"upload checkpoint {hf_repo_revision}",
        run_as_future=False,
    )
    print(f"🔥 Pushed {output_dir} to {hf_repo_url}")


def _maybe_strip_module_prefix(state_dict: dict) -> dict:
    """
    Strip a leading 'module.' prefix introduced by DistributedDataParallel saves.
    """
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _find_or_consolidate_weights(checkpoint_path: str) -> str | None:
    """
    Return a path to an existing consolidated weights file inside `checkpoint_path`.

    - Prefer pytorch_model.bin
    - Else pytorch_model.safetensors
    - Else return None (do not attempt consolidation here)
    """
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return bin_path

    sft_path = os.path.join(checkpoint_path, "pytorch_model.safetensors")
    if os.path.exists(sft_path):
        return sft_path

    return None


def _consolidate_with_zero_to_fp32(checkpoint_path: str) -> str | None:
    """
    Attempt to run zero_to_fp32.py to produce a pytorch_model.bin file in place.
    Handles variants that expect file vs directory outputs.
    Returns the path to the produced pytorch_model.bin if successful, else None.
    """
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    pytorch_model_dir = os.path.join(checkpoint_path, "pytorch_model")
    consolidate_script = os.path.join(checkpoint_path, "zero_to_fp32.py")
    if not (os.path.isdir(pytorch_model_dir) and os.path.exists(consolidate_script)):
        return None

    print(f" -> Found DeepSpeed checkpoint in {checkpoint_path}. Consolidating weights...")
    try:
        # Patch zero_to_fp32.py for torch 2.6+ compatibility
        print(" -> Patching zero_to_fp32.py for torch.load compatibility...")
        with open(consolidate_script, "r") as f:
            script_content = f.read()

        original_line = "state_dict = torch.load(f, map_location=device)"
        patched_line = "state_dict = torch.load(f, map_location=device, weights_only=False)"

        if original_line in script_content:
            script_content = script_content.replace(original_line, patched_line)
            with open(consolidate_script, "w") as f:
                f.write(script_content)
            print(" -> Patch applied successfully.")
        else:
            print(" -> WARNING: Could not find the line to patch in zero_to_fp32.py. The script might still fail.")

        # First attempt: pass output as file
        try:
            result = subprocess.run(
                ["python", "zero_to_fp32.py", ".", "pytorch_model.bin"],
                cwd=checkpoint_path,
                check=True,
                capture_output=True,
                text=True,
            )
            print(" -> Consolidation successful (file mode).")
            if result.stdout:
                print(" -> Consolidation script stdout:")
                print(result.stdout)
            if os.path.exists(bin_path):
                return bin_path
        except subprocess.CalledProcessError as e:
            # Retry: some variants expect an output directory; create and pass a dir
            out_dir = os.path.join(checkpoint_path, "pytorch_model_fp32")
            os.makedirs(out_dir, exist_ok=True)
            print(" -> File-mode consolidation failed. Retrying with directory output...")
            try:
                result = subprocess.run(
                    ["python", "zero_to_fp32.py", ".", "pytorch_model_fp32"],
                    cwd=checkpoint_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(" -> Consolidation successful (directory mode).")
                if result.stdout:
                    print(" -> Consolidation script stdout:")
                    print(result.stdout)
                # If the script created a single bin file inside the directory, use it
                candidate = os.path.join(out_dir, "pytorch_model.bin")
                if os.path.exists(candidate):
                    return candidate
                # If shards + index exist, return the directory path so caller can package shards
                index_json = os.path.join(out_dir, "pytorch_model.bin.index.json")
                if os.path.exists(index_json):
                    return out_dir
                # Otherwise, leave None and fall back to accelerator.load_state
            except subprocess.CalledProcessError as e2:
                print(f" -> ERROR: Failed to consolidate checkpoint {checkpoint_path} in directory mode.")
                print(" -> stdout:")
                print(e2.stdout)
                print(" -> stderr:")
                print(e2.stderr)
                return None
    except Exception as e:
        print(f" -> ERROR: Unexpected error during consolidation: {e}")
        return None

    return None


def _copy_weights_to_dir(weights_src_path: str, dst_dir: str) -> None:
    """
    Copy single-file weights or sharded weights (with index) into dst_dir.
    """
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.isdir(weights_src_path):
        for fname in os.listdir(weights_src_path):
            src = os.path.join(weights_src_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, fname))
    else:
        shutil.copy2(weights_src_path, os.path.join(dst_dir, os.path.basename(weights_src_path)))


def _load_state_dict_from_file(weights_path: str) -> dict:
    """
    Load a state_dict from a .bin or .safetensors file, handling common formats.
    """
    if weights_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except Exception as e:
            raise RuntimeError(
                "safetensors is required to load '.safetensors' checkpoints; please install it"
            ) from e
        state_dict = safetensors_load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    return _maybe_strip_module_prefix(state_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Convert and upload raw training checkpoints to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path of the base model used for training (e.g., 'Qwen/Qwen3-8B').",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="The local directory containing the raw training checkpoint folders (e.g., 'output/my_run/').",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="The Hugging Face repository ID to upload to (e.g., 'username/my-model').",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set this flag if the model requires trusting remote code.",
    )

    args = parser.parse_args()

    # --- 1. Find checkpoint folders ---
    if not os.path.isdir(args.checkpoints_dir):
        print(f"Error: The specified checkpoints directory does not exist: {args.checkpoints_dir}")
        return

    checkpoint_folders = [f for f in os.listdir(args.checkpoints_dir) if is_checkpoint_folder(args.checkpoints_dir, f)]

    if not checkpoint_folders:
        print(f"No checkpoint folders (e.g., 'step_500', 'epoch_1') found in {args.checkpoints_dir}.")
        return

    print(f"Found {len(checkpoint_folders)} checkpoints to convert and upload from {args.checkpoints_dir}.")

    # --- 2. Load base model and tokenizer ---
    print(f"\nLoading base model '{args.model_name_or_path}' to use as the architecture...")
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,  # Assuming bf16, adjust if necessary
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False
    )

    # --- 3. Initialize Accelerator ---
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    print("Model prepared with Accelerate.")

    # --- 4. Iterate, Convert, and Upload ---
    sorted_checkpoints = sorted(checkpoint_folders, key=lambda x: int(x.split("_")[-1]))

    for folder in sorted_checkpoints:
        checkpoint_path = os.path.join(args.checkpoints_dir, folder)
        temp_save_dir = Path(args.checkpoints_dir) / f"{folder}_converted_for_upload"
        revision_name = folder

        print(f"\nProcessing checkpoint: {checkpoint_path}")

        try:
            # Prefer direct weight files (pytorch_model.bin / .safetensors) if present.
            # If DeepSpeed shards are detected, consolidate them. Otherwise, fall back to accelerator.load_state.
            weights_path = _find_or_consolidate_weights(checkpoint_path)

            if weights_path is not None:
                print(f" -> Found consolidated weights: {weights_path}")
                state_dict = _load_state_dict_from_file(weights_path)
            else:
                # Try loading with accelerate first (works for many DS layouts)
                try:
                    print(f" -> No consolidated weights found. Trying accelerator.load_state from {checkpoint_path}...")
                    accelerator.load_state(checkpoint_path)
                    print(" -> Accelerator state loaded successfully.")
                    state_dict = accelerator.get_state_dict(model)
                except Exception as e:
                    print(f" -> accelerator.load_state failed: {e}")
                    print(" -> Attempting consolidation with zero_to_fp32.py...")
                    consolidated = _consolidate_with_zero_to_fp32(checkpoint_path)
                    if consolidated is not None:
                        print(f" -> Consolidated weights created: {consolidated}")
                        if os.path.isdir(consolidated):
                            # Directory mode: copy shards and index into temp_save_dir and skip state_dict assembly
                            print(" -> Detected sharded fp32 weights. Packaging shards and index for upload...")
                            _copy_weights_to_dir(consolidated, str(temp_save_dir))
                            state_dict = None
                        else:
                            state_dict = _load_state_dict_from_file(consolidated)
                    else:
                        print(" -> ERROR: Could not load or consolidate weights for this checkpoint. Skipping.")
                        continue

            # Unwrap the model and save it in the final format using the resolved state_dict
            print(f" -> Converting and saving to temporary directory: {temp_save_dir}...")
            unwrapped_model = accelerator.unwrap_model(model)

            # If we already copied shards (directory mode), only ensure config is saved.
            if state_dict is None:
                # Save config to the same directory to make a valid HF folder
                config.save_pretrained(str(temp_save_dir))
            else:
                unwrapped_model.save_pretrained(
                    str(temp_save_dir),
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state_dict,
                    safe_serialization=False,  # Set to False to get .bin files
                )

            if accelerator.is_main_process:
                # Save tokenizer and fix chat template
                chat_template = tokenizer.chat_template
                tokenizer.chat_template = None
                tokenizer.save_pretrained(str(temp_save_dir))
                tokenizer.chat_template = chat_template

                if chat_template:
                    with open(temp_save_dir / "tokenizer_config.json", "r", encoding="utf-8") as f:
                        tokenizer_config = json.load(f)
                    tokenizer_config["chat_template"] = chat_template
                    with open(temp_save_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
                        json.dump(tokenizer_config, f, indent=2, sort_keys=True)
                        f.write("\n")

            accelerator.wait_for_everyone()
            print(" -> Conversion complete.")

            # Upload the converted model
            print(f" -> Uploading to Hugging Face Hub as revision '{revision_name}'...")
            push_folder_to_hub(
                output_dir=str(temp_save_dir),
                hf_repo_id=args.hf_repo_id,
                hf_repo_revision=revision_name,
            )

        except Exception as e:
            print(f"An error occurred while processing {checkpoint_path}: {e}")
        finally:
            # Clean up the temporary directory
            if accelerator.is_main_process and os.path.exists(temp_save_dir):
                print(f" -> Cleaning up temporary directory: {temp_save_dir}")
                shutil.rmtree(temp_save_dir)
            accelerator.wait_for_everyone()

    print("\n\nAll checkpoints processed.")


if __name__ == "__main__":
    main()
