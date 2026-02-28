#!/usr/bin/env python3
"""Create train/eval datasets for the AppWorld RL environment."""

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset


def _load_appworld_symbols() -> tuple[type, Any, Any]:
    try:
        module = importlib.import_module("appworld")
    except ImportError as e:
        raise ImportError(
            "This script requires `appworld`. Install with `pip install -e '.[appworld]'` "
            "(or `pip install appworld`), "
            "then run `appworld install` and `appworld download data`."
        ) from e

    appworld_cls = getattr(module, "AppWorld", None)
    load_task_ids = getattr(module, "load_task_ids", None)
    update_root = getattr(module, "update_root", None)
    if appworld_cls is None or load_task_ids is None:
        raise RuntimeError("Could not find `AppWorld` and `load_task_ids` in the installed appworld package.")
    return appworld_cls, load_task_ids, update_root


def _format_supervisor(supervisor: Any) -> str:
    if not isinstance(supervisor, dict):
        return "Unknown"
    first = str(supervisor.get("first_name", "")).strip()
    last = str(supervisor.get("last_name", "")).strip()
    email = str(supervisor.get("email", "")).strip()
    name = " ".join(part for part in (first, last) if part).strip()
    if name and email:
        return f"{name} ({email})"
    return name or email or "Unknown"


def _build_user_prompt(
    task_id: str, instruction: str, supervisor: Any, app_descriptions: dict[str, str] | None, nothink: bool
) -> str:
    lines = [
        f"AppWorld task_id: {task_id}",
        f"Supervisor: {_format_supervisor(supervisor)}",
        f"Instruction: {instruction}",
        "",
        "Important: this prompt must include all task context; env reset output is not injected.",
        "",
        "Use the `appworld_execute` tool to run Python code in the AppWorld shell.",
        "The shell is stateful across calls.",
        "Use `apis.<app>.<api>(...)` for API calls and call `apis.supervisor.complete_task()` when done.",
    ]
    if app_descriptions:
        lines.append("")
        lines.append("App descriptions:")
        for app_name in sorted(app_descriptions.keys()):
            lines.append(f"- {app_name}: {app_descriptions[app_name]}")
    prompt = "\n".join(lines)
    if nothink:
        prompt += " /nothink"
    return prompt


def create_appworld_samples(
    split_name: str,
    max_samples: int | None,
    nothink: bool,
    include_app_descriptions: bool,
    inspection_experiment_name: str,
) -> list[dict]:
    appworld_cls, load_task_ids, _ = _load_appworld_symbols()
    task_ids = list(load_task_ids(split_name))
    if max_samples is not None:
        task_ids = task_ids[:max_samples]

    samples: list[dict] = []
    system_prompt = (
        "You are an autonomous coding agent solving AppWorld tasks. "
        "Write concise Python code that interacts with AppWorld APIs to complete the task."
    )

    for idx, task_id in enumerate(task_ids, start=1):
        if idx % 50 == 0:
            print(f"[{split_name}] Processed {idx}/{len(task_ids)} tasks...")

        with appworld_cls(task_id=task_id, experiment_name=inspection_experiment_name) as world:
            task = world.task
            instruction = str(getattr(task, "instruction", "")).strip()
            supervisor = getattr(task, "supervisor", {})
            app_descriptions = getattr(task, "app_descriptions", None) if include_app_descriptions else None

        user_prompt = _build_user_prompt(task_id, instruction, supervisor, app_descriptions, nothink=nothink)
        samples.append(
            {
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "ground_truth": "",
                "dataset": "passthrough",
                "env_config": {"env_name": "appworld", "task_id": task_id},
            }
        )
    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def upload_to_huggingface(samples: list[dict], repo_id: str):
    dataset = Dataset.from_list(samples)
    dataset.push_to_hub(repo_id, private=False)
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Create train/eval datasets for AppWorld RL env")
    parser.add_argument("--namespace", default="hamishivi", help="HuggingFace namespace")
    parser.add_argument("--train-split", default="train", help="AppWorld split for training examples")
    parser.add_argument("--eval-split", default="dev", help="AppWorld split for eval examples")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train sample count")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap for eval sample count")
    parser.add_argument("--appworld-root", default=None, help="Optional APPWORLD_ROOT path")
    parser.add_argument(
        "--inspection-experiment-name",
        default="open_instruct_dataset_build",
        help="Experiment name used while loading tasks to extract metadata",
    )
    parser.add_argument("--include-app-descriptions", action="store_true", help="Include app descriptions in prompts")
    parser.add_argument("--nothink", action="store_true", help="Append /nothink to user prompts")
    parser.add_argument("--local-only", action="store_true", help="Only write local jsonl files (no HF upload)")
    args = parser.parse_args()

    _, _, update_root = _load_appworld_symbols()
    if args.appworld_root:
        if update_root is not None:
            update_root(args.appworld_root)
        else:
            os.environ["APPWORLD_ROOT"] = args.appworld_root
        print(f"Using APPWORLD_ROOT={args.appworld_root}")

    suffix = "-nothink" if args.nothink else ""
    data_dir = Path(__file__).parent.parent.parent / "data" / "envs"
    specs = [
        ("train", args.train_split, args.max_train_samples, f"rlenv-appworld-train{suffix}"),
        ("eval", args.eval_split, args.max_eval_samples, f"rlenv-appworld-eval{suffix}"),
    ]

    local_paths: dict[str, Path] = {}
    repo_ids: dict[str, str] = {}
    for label, split_name, max_samples, repo_name in specs:
        print(f"Building {label} dataset from AppWorld split '{split_name}'...")
        samples = create_appworld_samples(
            split_name=split_name,
            max_samples=max_samples,
            nothink=args.nothink,
            include_app_descriptions=args.include_app_descriptions,
            inspection_experiment_name=args.inspection_experiment_name,
        )
        local_path = data_dir / f"{repo_name.replace('-', '_')}.jsonl"
        save_jsonl(samples, local_path)
        local_paths[label] = local_path

        if not args.local_only:
            repo_id = f"{args.namespace}/{repo_name}"
            upload_to_huggingface(samples, repo_id)
            repo_ids[label] = repo_id

    print("\nSuggested GRPO args (train on train split, eval on dev split):")
    if args.local_only:
        print(f"  --dataset_mixer_list {local_paths['train']} 1.0")
        print(f"  --dataset_mixer_eval_list {local_paths['eval']} 1.0")
    else:
        print(f"  --dataset_mixer_list {repo_ids['train']} 1.0")
        print(f"  --dataset_mixer_eval_list {repo_ids['eval']} 1.0")
    print("  --dataset_mixer_list_splits train")
    print("  --dataset_mixer_eval_list_splits train")
    print("  --tools appworld")
    print("  --reward_aggregator last")


if __name__ == "__main__":
    main()
