"""Legacy workflow for downloading eval results from Beaker and running length analysis."""

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

from beaker import Beaker

from download_results import gather_experiments
from length_analysis import (
    calculate_statistics,
    extract_model_responses,
    load_jsonl_file,
    print_statistics,
    tokenize_responses,
)


StatsDict = Dict[str, Union[int, float]]


def sanitize_name(value: str) -> str:
    """Return a filesystem-friendly name derived from value."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "experiment"


def download_results(beaker_client: Beaker, experiment, output_root: Path, overwrite: bool = False) -> Path:
    """Download the Beaker dataset for the experiment into output_root."""
    safe_name = sanitize_name(getattr(experiment, "name", "experiment"))
    target_dir = output_root / f"{safe_name}-{experiment.id}"

    if target_dir.exists():
        if target_dir.is_file():
            if overwrite:
                target_dir.unlink()
            else:
                raise RuntimeError(
                    f"Target path {target_dir} exists and is not a directory. Use --overwrite to replace it."
                )
        elif any(target_dir.iterdir()):
            if overwrite:
                shutil.rmtree(target_dir)
            else:
                print(f"  Results already exist at {target_dir}; skipping download.")
                return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = experiment.jobs[0].result.beaker
    print(f"  Fetching dataset {dataset_id} to {target_dir}...")
    beaker_client.dataset.fetch(dataset_id, target=str(target_dir), quiet=True)
    return target_dir


def download_experiment_task(experiment, output_root: Path, overwrite: bool = False):
    """Download a single experiment's dataset, intended for use in parallel workers."""
    label = f"{getattr(experiment, 'name', 'experiment')} ({experiment.id})"
    try:
        beaker_client = Beaker.from_env()
        print(f"\n=== Scheduling download: {label} ===")
        target_dir = download_results(beaker_client, experiment, output_root, overwrite)
        return label, target_dir
    except Exception as exc:  # pragma: no cover - legacy behavior
        print(f"  Download failed for {label}: {exc}")
        return label, None


def find_matching_files(root: Path, pattern: str) -> List[Path]:
    """Return all files under root matching pattern."""
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def normalize_stats(stats: StatsDict) -> StatsDict:
    """Convert numpy scalar values to native Python scalars for serialization."""
    return {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in stats.items()
    }


def analyze_experiment(experiment_label: str, files: Iterable[Path], tokenizer) -> Tuple[StatsDict, List[int]]:
    """Run length analysis for a single experiment using provided tokenizer."""
    all_token_counts: List[int] = []

    for file_path in files:
        print(f"\nProcessing file: {file_path}")
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} entries")
        responses = extract_model_responses(data)
        print(f"  Found {len(responses)} model responses")

        if not responses:
            continue

        token_counts = tokenize_responses(responses, tokenizer)
        all_token_counts.extend(token_counts)
        print(f"  Accumulated {len(token_counts)} responses for experiment {experiment_label}")

    if not all_token_counts:
        print(f"  No model responses found for experiment {experiment_label}.")
        return {}, []

    stats = normalize_stats(calculate_statistics(all_token_counts))
    print_statistics(f"Experiment {experiment_label}", stats)
    return stats, all_token_counts


def load_tokenizer():
    from transformers import AutoTokenizer

    print("Loading dolma2 tokenizer...")
    return AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download eval results matching authors/match string and run length analysis.",
    )
    parser.add_argument(
        "--authors",
        nargs="+",
        required=False,
        default=None,
        help="List of Beaker author usernames to include. If omitted, include all authors.",
    )
    parser.add_argument(
        "--search_string",
        required=False,
        default=None,
        help=(
            "Substring to match in experiment names e.g. a leaderboard string. "
            "Optional when providing --dataset-paths."
        ),
    )
    parser.add_argument(
        "--workspace",
        default="ai2/tulu-3-results",
        help="Beaker workspace to search (default: ai2/tulu-3-results).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of experiments to examine (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where experiment datasets will be stored (default: downloads).",
    )
    parser.add_argument(
        "--pattern",
        default="*predictions.jsonl",
        help="Glob pattern for files used in length analysis (default: *predictions.jsonl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download datasets even if they already exist locally.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel download workers (default: CPU count).",
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        default=None,
        help=(
            "Explicit local directories containing Beaker dataset contents to analyze. "
            "When provided, the script skips Beaker lookup/download and analyzes these directories directly."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Optional labels for the provided --dataset-paths, in the same order. "
            "If omitted, labels default to the sanitized absolute path of each directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    elif args.dataset_paths:
        output_dir = "manual_datasets_analysis"
    else:
        if not args.search_string:
            raise SystemExit("--search_string is required unless --dataset-paths is provided.")
        output_dir = args.search_string

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    downloaded_dirs: Dict[str, Path] = {}

    if args.dataset_paths:
        print("Using provided dataset paths. Skipping Beaker downloads...")
        if args.labels is not None and len(args.labels) != len(args.dataset_paths):
            raise SystemExit(
                f"--labels count ({len(args.labels)}) must match --dataset-paths count ({len(args.dataset_paths)})."
            )

        candidate_pairs = []
        for idx, raw_path in enumerate(args.dataset_paths):
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                print(f"  Warning: Provided path does not exist and will be skipped: {path}")
                continue
            if not path.is_dir():
                print(f"  Warning: Provided path is not a directory and will be skipped: {path}")
                continue
            label = args.labels[idx] if args.labels is not None else sanitize_name(str(path))
            candidate_pairs.append((label, path))

        labels_only = [label for label, _ in candidate_pairs]
        if len(set(labels_only)) != len(labels_only):
            seen = set()
            dups = sorted({name for name in labels_only if (name in seen) or seen.add(name)})
            raise SystemExit(f"Duplicate labels detected: {dups}. Please ensure labels are unique.")

        for label, path in candidate_pairs:
            downloaded_dirs[label] = path
        if not downloaded_dirs:
            print("No valid dataset directories provided.")
            return
        print("\nBeginning analysis of provided datasets...")
    else:
        beaker_client = Beaker.from_env()

        experiments = gather_experiments(
            author_list=args.authors,
            workspace_name=args.workspace,
            relevant_match=args.search_string,
            limit=args.limit,
        )

        if not experiments:
            print("No experiments matched the provided criteria.")
            return

        workers = args.workers or os.cpu_count() or 4
        print(f"\nStarting parallel downloads with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_exp = {
                executor.submit(download_experiment_task, experiment, output_root, args.overwrite): experiment
                for experiment in experiments
            }
            for future in as_completed(future_to_exp):
                label, target_dir = future.result()
                if target_dir is not None:
                    downloaded_dirs[label] = target_dir

        if not downloaded_dirs:
            print("No experiment datasets were downloaded successfully.")
            return

        print("\nDownloads complete. Beginning analysis...")

    tokenizer = load_tokenizer()

    per_experiment_stats: Dict[str, StatsDict] = {}
    aggregate_token_counts: List[int] = []

    for experiment_label, target_dir in downloaded_dirs.items():
        print(f"\n=== Analyzing: {experiment_label} ===")
        matched_files = find_matching_files(target_dir, args.pattern)

        if not matched_files:
            print(f"  No files matching '{args.pattern}' found in {target_dir}")
            continue

        experiment_stats, token_counts = analyze_experiment(experiment_label, matched_files, tokenizer)
        per_experiment_stats[experiment_label] = experiment_stats
        aggregate_token_counts.extend(token_counts)

    overall_stats = (
        normalize_stats(calculate_statistics(aggregate_token_counts))
        if aggregate_token_counts
        else {}
    )

    if overall_stats:
        print_statistics("All Experiments", overall_stats)
    else:
        print("\nNo model responses found across experiments.")

    summary = {
        "per_experiment": per_experiment_stats,
        "overall": overall_stats,
    }

    print("\nAll statistics across processed experiments (also saved to statistics.json):")
    print(json.dumps(summary, indent=2))

    stats_path = output_root / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nAggregate statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
