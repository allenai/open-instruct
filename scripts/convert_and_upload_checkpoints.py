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
            # Check if this is a DeepSpeed checkpoint that needs consolidation.
            # This is indicated by the presence of a `pytorch_model` directory
            # (for sharded weights) and the consolidation script, but no `pytorch_model.bin`.
            pytorch_model_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
            pytorch_model_dir = os.path.join(checkpoint_path, "pytorch_model")
            consolidate_script = os.path.join(checkpoint_path, "zero_to_fp32.py")

            if (
                not os.path.exists(pytorch_model_bin)
                and os.path.isdir(pytorch_model_dir)
                and os.path.exists(consolidate_script)
            ):
                print(f" -> Found DeepSpeed checkpoint in {checkpoint_path}. Consolidating weights...")
                try:
                    # Patch `zero_to_fp32.py` to be compatible with PyTorch 2.6+ `torch.load`
                    # by setting `weights_only=False`. This is safe as we're using our own checkpoints.
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
                        print(" -> WARNING: Could not find the line to patch in zero_to_fp32.py. The script might fail.")
                    
                    # The script `zero_to_fp32.py` is usually run from within the checkpoint directory.
                    # It takes the current directory '.' and the output file name 'pytorch_model.bin' as arguments.
                    result = subprocess.run(
                        ["python", "zero_to_fp32.py", ".", "pytorch_model.bin"],
                        cwd=checkpoint_path,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(" -> Consolidation successful.")
                    if result.stdout:
                        print(" -> Consolidation script stdout:")
                        print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f" -> ERROR: Failed to consolidate checkpoint {checkpoint_path}.")
                    print(" -> stdout:")
                    print(e.stdout)
                    print(" -> stderr:")
                    print(e.stderr)
                    continue  # Skip to the next checkpoint

            # Load the raw checkpoint state into the model
            print(f" -> Loading raw checkpoint state from {checkpoint_path}...")
            accelerator.load_state(checkpoint_path)
            print(" -> State loaded successfully.")

            # Unwrap the model and save it in the final format
            print(f" -> Converting and saving to temporary directory: {temp_save_dir}...")
            unwrapped_model = accelerator.unwrap_model(model)

            # Use get_state_dict to ensure all weights are gathered
            state_dict = accelerator.get_state_dict(model)

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
