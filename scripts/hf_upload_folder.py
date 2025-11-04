#!/usr/bin/env python3
import argparse
import functools
import os
import time

from huggingface_hub import HfApi


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
def push_folder_to_hub(folder_path: str, hf_repo_id: str, hf_repo_revision: str, private: bool = False):
    api = HfApi()
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    if hf_repo_revision is not None:
        api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=hf_repo_revision,
        folder_path=folder_path,
        commit_message=f"upload folder {hf_repo_revision}",
        run_as_future=False,
    )
    print(f"🔥 Pushed {folder_path} to https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local folder to a Hugging Face repo/revision (branch)."
    )
    parser.add_argument("path", type=str, help="Local folder path to upload")
    parser.add_argument("hf_repo_id", type=str, help="Target HF repo id, e.g. 'username/my-model'")
    parser.add_argument("revision", type=str, help="Target revision/branch name")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist (default public).",
    )

    args = parser.parse_args()

    folder_path = os.path.abspath(args.path)
    if not os.path.isdir(folder_path):
        raise SystemExit(f"Error: Path does not exist or is not a directory: {folder_path}")

    print(f"Uploading folder: {folder_path}")
    print(f" -> Repo: {args.hf_repo_id}")
    print(f" -> Revision: {args.revision}")
    if args.private:
        print(" -> Visibility: private")

    push_folder_to_hub(
        folder_path=folder_path,
        hf_repo_id=args.hf_repo_id,
        hf_repo_revision=args.revision,
        private=args.private,
    )


if __name__ == "__main__":
    main()


