import argparse
import functools
import os
import time

from huggingface_hub import HfApi


def retry_on_exception(max_attempts=4, delay=1, backoff=2):
    """
    Retry a function on exception. Useful for HF API calls that may fail due to
    transient network issues.
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
    folder_path: str,
    hf_repo_id: str,
    hf_repo_revision: str,
    private: bool = True,
):
    """
    Upload a local folder to the Hugging Face Hub at the specified revision (branch).
    Creates the repo/branch if they don't exist.
    """
    api = HfApi()
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    if hf_repo_revision is not None:
        api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=hf_repo_revision,
        folder_path=folder_path,
        commit_message=f"upload checkpoint {hf_repo_revision}",
        run_as_future=False,
    )
    print(f"🔥 Pushed {folder_path} to https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}")


def is_hf_checkpoint_folder(parent_dir: str, name: str) -> bool:
    """Return True if `name` is a directory under `parent_dir` and ends with `_hf`."""
    full_path = os.path.join(parent_dir, name)
    return os.path.isdir(full_path) and name.endswith("_hf")


def folder_to_revision(folder_name: str) -> str:
    """
    Convert a folder name like `epoch_0_hf` or `step_500_hf` to a revision name
    by stripping the trailing `_hf`.
    """
    return folder_name[:-3] if folder_name.endswith("_hf") else folder_name


def sort_key_for_revision(revision: str):
    """
    Provide a stable sort key that prefers numeric suffix after an underscore, e.g.:
    - epoch_0 -> ("epoch", 0)
    - step_500 -> ("step", 500)
    - other -> (revision, -1)
    """
    parts = revision.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return ("_".join(parts[:-1]), int(parts[-1]))
    return (revision, -1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Upload existing Hugging Face-format checkpoints (folders ending with _hf) "
            "as separate revisions to a single HF repo."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoints_dir",
        type=str,
        help=(
            "Path to a directory containing multiple checkpoint subfolders like epoch_0_hf, step_500_hf, etc."
        ),
    )
    group.add_argument(
        "--checkpoint_path",
        type=str,
        help=(
            "Path to a single HF checkpoint folder to upload (e.g., epoch_3_hf)."
        ),
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="Target Hugging Face repo id, e.g. 'username/my-model'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private if it doesn't exist (default public).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Revision (branch/tag) to use when uploading a single checkpoint via --checkpoint_path. "
            "If omitted, it is derived from the folder name by stripping a trailing '_hf'."
        ),
    )

    args = parser.parse_args()

    if args.checkpoints_dir:
        if not os.path.isdir(args.checkpoints_dir):
            print(f"Error: The specified checkpoints directory does not exist: {args.checkpoints_dir}")
            return

        # Find *_hf folders
        entries = os.listdir(args.checkpoints_dir)
        hf_folders = [name for name in entries if is_hf_checkpoint_folder(args.checkpoints_dir, name)]

        if not hf_folders:
            print(
                f"No '*_hf' checkpoint folders found in {args.checkpoints_dir}. Nothing to upload."
            )
            return

        # Derive revisions and sort them
        revisions_and_paths = []
        for name in hf_folders:
            revision = folder_to_revision(name)
            full_path = os.path.join(args.checkpoints_dir, name)
            revisions_and_paths.append((revision, full_path))

        revisions_and_paths.sort(key=lambda x: sort_key_for_revision(x[0]))

        print(
            f"Found {len(revisions_and_paths)} HF checkpoints to upload from {args.checkpoints_dir}."
        )

        for revision, folder_path in revisions_and_paths:
            print(f"\nUploading folder: {folder_path}")
            print(f" -> Using revision: {revision}")
            try:
                push_folder_to_hub(
                    folder_path=folder_path,
                    hf_repo_id=args.hf_repo_id,
                    hf_repo_revision=revision,
                    private=args.private,
                )
            except Exception as e:
                print(f"ERROR: Failed to upload {folder_path} as revision '{revision}': {e}")

        print("\nAll *_hf checkpoints processed.")
    else:
        # Single checkpoint upload path
        folder_path = args.checkpoint_path
        if not os.path.isdir(folder_path):
            print(f"Error: The specified checkpoint path does not exist or is not a directory: {folder_path}")
            return

        # Determine revision: use --revision if provided, else derive from folder name
        if args.revision is not None:
            revision = args.revision
        else:
            # Derive from leaf folder name, stripping trailing _hf if present
            folder_name = os.path.basename(os.path.normpath(folder_path))
            revision = folder_to_revision(folder_name)

        print(f"\nUploading folder: {folder_path}")
        print(f" -> Using revision: {revision}")
        try:
            push_folder_to_hub(
                folder_path=folder_path,
                hf_repo_id=args.hf_repo_id,
                hf_repo_revision=revision,
                private=args.private,
            )
        except Exception as e:
            print(f"ERROR: Failed to upload {folder_path} as revision '{revision}': {e}")
            return
        print("\nSingle checkpoint processed.")


if __name__ == "__main__":
    main()


