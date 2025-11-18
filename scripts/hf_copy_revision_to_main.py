#!/usr/bin/env python3
import argparse
import functools
import os
import tempfile
import time

from huggingface_hub import HfApi, snapshot_download


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


@retry_on_exception()
def copy_revision_to_main(hf_repo_id: str, source_revision: str):
    """
    Download files from a specific revision and upload them to the main branch.
    """
    api = HfApi()
    
    if not api.repo_exists(hf_repo_id):
        raise SystemExit(f"Error: Repository does not exist: {hf_repo_id}")
    
    print(f"Downloading files from revision: {source_revision}")
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_download(
            repo_id=hf_repo_id,
            revision=source_revision,
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
        )
        
        print(f"Uploading files to main branch")
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
        description="Copy files from a specific revision to the main branch of a Hugging Face repo."
    )
    parser.add_argument("hf_repo_id", type=str, help="HF repo id, e.g. 'username/my-model'")
    parser.add_argument("source_revision", type=str, help="Source revision/branch name to copy from")

    args = parser.parse_args()

    print(f"Repo: {args.hf_repo_id}")
    print(f"Source revision: {args.source_revision}")
    print(f"Target: main")

    copy_revision_to_main(
        hf_repo_id=args.hf_repo_id,
        source_revision=args.source_revision,
    )


if __name__ == "__main__":
    main()

