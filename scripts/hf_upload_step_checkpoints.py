#!/usr/bin/env python3
import argparse
import functools
import os
import re
import tempfile
import time

from huggingface_hub import HfApi, revision_exists, snapshot_download


def retry_on_exception(max_attempts: int = 4, delay: int = 1, backoff: int = 2):
    """
    Retry a function on exception. Helpful for transient network errors.
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


def find_step_folders(base_dir: str):
    """
    Recursively find all subfolders named 'step_N' where N is a number.
    Returns list of tuples: (step_number, folder_path)
    """
    step_folders = []
    pattern = re.compile(r'^step_(\d+)$')
    
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            match = pattern.match(dir_name)
            if match:
                step_num = int(match.group(1))
                folder_path = os.path.join(root, dir_name)
                step_folders.append((step_num, folder_path))
    
    return sorted(step_folders, key=lambda x: x[0])


@retry_on_exception()
def upload_folder_if_not_exists(api: HfApi, folder_path: str, hf_repo_id: str, revision: str, private: bool = False):
    """
    Upload folder to HF repo as revision if revision doesn't exist.
    """
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    
    if revision_exists(hf_repo_id, revision=revision):
        print(f"  ⏭️  Revision '{revision}' already exists, skipping")
        return False
    
    api.create_branch(repo_id=hf_repo_id, branch=revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=revision,
        folder_path=folder_path,
        commit_message=f"Upload checkpoint {revision}",
        run_as_future=False,
    )
    print(f"  ✅ Uploaded {revision}: https://huggingface.co/{hf_repo_id}/tree/{revision}")
    return True


@retry_on_exception()
def copy_revision_to_main(api: HfApi, hf_repo_id: str, source_revision: str):
    """
    Copy files from a specific revision to the main branch (overwriting main).
    """
    print(f"Downloading files from revision: {source_revision}")
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_download(
            repo_id=hf_repo_id,
            revision=source_revision,
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
        )
        
        print(f"Uploading files to main branch (overwriting)")
        api.upload_folder(
            repo_id=hf_repo_id,
            revision="main",
            folder_path=tmpdir,
            commit_message=f"Copy contents from revision {source_revision} to main",
            run_as_future=False,
        )
    
    print(f"✅ Copied {source_revision} to main: https://huggingface.co/{hf_repo_id}/tree/main")


def main():
    parser = argparse.ArgumentParser(
        description="Find step_N folders, upload as revisions if they don't exist, then copy largest step to main."
    )
    parser.add_argument("hf_repo_id", type=str, help="HF repo id, e.g. 'username/my-model'")
    parser.add_argument("base_dir", type=str, help="Base directory to recursively search for step_N folders")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist (always True).",
    )

    args = parser.parse_args()
    # Always use private=True
    args.private = True

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Error: Base directory does not exist: {base_dir}")

    print(f"Searching for step_N folders in: {base_dir}")
    step_folders = find_step_folders(base_dir)
    
    if not step_folders:
        print("No step_N folders found.")
        return
    
    print(f"Found {len(step_folders)} step_N folder(s)")
    
    api = HfApi()
    if not api.repo_exists(args.hf_repo_id):
        api.create_repo(args.hf_repo_id, exist_ok=True, private=args.private)
        print(f"Created repository: {args.hf_repo_id}")
    
    max_step = None
    max_step_path = None
    
    for step_num, folder_path in step_folders:
        revision = f"step_{step_num}"
        print(f"\nProcessing step_{step_num}...")
        upload_folder_if_not_exists(api, folder_path, args.hf_repo_id, revision, args.private)
        
        if max_step is None or step_num > max_step:
            max_step = step_num
            max_step_path = folder_path
    
    if max_step is not None:
        print(f"\n{'='*60}")
        print(f"Copying step_{max_step} to main branch...")
        copy_revision_to_main(api, args.hf_repo_id, f"step_{max_step}")
    else:
        print("No steps found to copy to main.")


if __name__ == "__main__":
    main()

