#!/usr/bin/env python3
"""Aggregate response length medians and primary scores from statistics.json files with preferred column merging."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Define column preferences: (preferred_column, alternative_column, merged_name)
COLUMN_PREFERENCES = [
    # Prefer non-lite livecodebench
    ('livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags',
     'livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite',
     'livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags'),
    
    # Prefer deepseek_v2 for BBH
    ('bbh:cot::hamish_zs_reasoning_deepseek_v2',
     'bbh:cot::hamish_zs_reasoning',
     'bbh:cot::hamish_zs_reasoning_deepseek_v2'),
    
    # Prefer full minerva_math over 500 subset
    ('minerva_math::hamish_zs_reasoning_deepseek',
     'minerva_math_500::hamish_zs_reasoning_deepseek',
     'minerva_math::hamish_zs_reasoning_deepseek'),
    
    # Prefer qwen3_instruct GPQA over hamish version
    ('gpqa:0shot_cot::qwen3_instruct',
     'gpqa:0shot_cot::hamish_zs_reasoning_deepseek',
     'gpqa:0shot_cot::qwen3_instruct'),
]


def discover_statistics_files(paths: Iterable[str]) -> List[Path]:
    """Return all statistics.json files found under the provided paths."""
    discovered: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if path.name == "statistics.json":
                discovered.append(path)
            else:
                raise ValueError(f"Expected statistics.json file, got: {path}")
        elif path.is_dir():
            discovered.extend(sorted(path.rglob("statistics.json")))
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    unique_files = list(dict.fromkeys(discovered))
    if not unique_files:
        raise SystemExit("No statistics.json files found in provided paths.")
    return unique_files


def load_alias_metrics(stats_path: Path) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Load the model name with per-alias medians and primary scores."""
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    model_name = payload.get("model_name") or stats_path.parent.name
    alias_entries = payload.get("aliases", {})

    medians: Dict[str, float] = {}
    primary_scores: Dict[str, float] = {}
    for alias, info in alias_entries.items():
        stats = info.get("stats") or {}
        median_value = stats.get("median")
        if median_value is None:
            continue
        try:
            medians[alias] = float(median_value)
        except (TypeError, ValueError):
            continue

        scores = info.get("scores") or {}
        primary = scores.get("primary_score")
        if primary is None:
            continue
        try:
            score_value = float(primary)
            if 0 < score_value < 1:
                score_value *= 100
            primary_scores[alias] = score_value
        except (TypeError, ValueError):
            continue

    if not medians and not primary_scores:
        raise ValueError(
            f"No usable alias metrics found in {stats_path}. "
            "Ensure the file was produced by download_and_analyze.py."
        )

    return model_name, medians, primary_scores


def merge_preferred_columns(
    matrix: Dict[str, Dict[str, float]],
    aliases: List[str]
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Merge duplicate columns using preferred versions.
    
    Returns: (merged_matrix, merged_aliases)
    """
    # Build a mapping from any alias to its preferred/merged name
    alias_mapping: Dict[str, str] = {}
    columns_to_skip: set[str] = set()
    
    for preferred, alternative, merged in COLUMN_PREFERENCES:
        # Mark both columns for processing
        if preferred in aliases or alternative in aliases:
            alias_mapping[preferred] = merged
            alias_mapping[alternative] = merged
            columns_to_skip.add(preferred)
            columns_to_skip.add(alternative)
    
    # Build new alias list
    new_aliases = []
    for alias in aliases:
        if alias in columns_to_skip:
            merged_name = alias_mapping[alias]
            if merged_name not in new_aliases:
                new_aliases.append(merged_name)
        else:
            new_aliases.append(alias)
    
    # Build new matrix with merged columns
    new_matrix: Dict[str, Dict[str, float]] = {}
    
    for model_name, model_data in matrix.items():
        new_matrix[model_name] = {}
        
        for alias in aliases:
            if alias in alias_mapping:
                # This is a column to be merged
                merged_name = alias_mapping[alias]
                if merged_name not in new_matrix[model_name]:
                    # Determine which value to use
                    for preferred, alternative, merged in COLUMN_PREFERENCES:
                        if merged == merged_name:
                            # Prefer the preferred column if it exists, otherwise use alternative
                            if preferred in model_data:
                                new_matrix[model_name][merged_name] = model_data[preferred]
                            elif alternative in model_data:
                                new_matrix[model_name][merged_name] = model_data[alternative]
                            break
            else:
                # Regular column, copy as-is
                if alias in model_data:
                    new_matrix[model_name][alias] = model_data[alias]
    
    return new_matrix, new_aliases


def write_tsv(
    output_path: Path,
    matrix: Dict[str, Dict[str, float]],
    aliases: List[str],
    round_digits: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model"] + aliases)

        for model_name in sorted(matrix.keys()):
            row = [model_name]
            for alias in aliases:
                value = matrix[model_name].get(alias)
                row.append(f"{value:.{round_digits}f}" if value is not None else "")
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-alias response length medians and primary scores from one or more "
            "statistics.json files into TSV matrices, with preferred column merging."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more paths to statistics.json files or directories containing them."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where output TSV files will be written "
            "(default: the first input path, or its parent directory if it's a file)."
        ),
    )
    parser.add_argument(
        "--output",
        default="median_lengths.tsv",
        help="Output TSV filename (default: median_lengths.tsv).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Number of decimal places to include for medians (default: 1).",
    )
    parser.add_argument(
        "--scores-output",
        default="primary_scores.tsv",
        help="Optional TSV filename for primary scores (default: primary_scores.tsv).",
    )
    parser.add_argument(
        "--score-round",
        type=int,
        default=4,
        help="Decimal places for primary scores (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_files = discover_statistics_files(args.paths)
    print(f"Discovered {len(stats_files)} statistics.json file(s)")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        first_path = Path(args.paths[0]).expanduser().resolve()
        if first_path.is_file():
            output_dir = first_path.parent
        else:
            output_dir = first_path
    output_dir.mkdir(parents=True, exist_ok=True)

    length_matrix: Dict[str, Dict[str, float]] = {}
    score_matrix: Dict[str, Dict[str, float]] = {}
    length_aliases: List[str] = []
    score_aliases: List[str] = []

    for stats_path in stats_files:
        model_name, medians, primary_scores = load_alias_metrics(stats_path)

        if medians:
            length_matrix.setdefault(model_name, {})
            length_matrix[model_name].update(medians)
            for alias in medians:
                if alias not in length_aliases:
                    length_aliases.append(alias)

        if primary_scores:
            score_matrix.setdefault(model_name, {})
            score_matrix[model_name].update(primary_scores)
            for alias in primary_scores:
                if alias not in score_aliases:
                    score_aliases.append(alias)

    # Merge preferred columns
    if length_matrix:
        print("\nMerging duplicate columns in length data...")
        length_matrix, length_aliases = merge_preferred_columns(length_matrix, length_aliases)
        length_columns = sorted(length_aliases)
        length_output = output_dir / args.output
        write_tsv(length_output, length_matrix, length_columns, args.round)
        print(
            f"Wrote length medians TSV with {len(length_matrix)} models and "
            f"{len(length_columns)} aliases to {length_output}"
        )
    else:
        print("No length medians found; skipping length TSV generation.")

    if score_matrix:
        print("\nMerging duplicate columns in score data...")
        score_matrix, score_aliases = merge_preferred_columns(score_matrix, score_aliases)
        score_columns = sorted(score_aliases)
        score_output = output_dir / args.scores_output
        write_tsv(score_output, score_matrix, score_columns, args.score_round)
        print(
            f"Wrote primary scores TSV with {len(score_matrix)} models and "
            f"{len(score_columns)} aliases to {score_output}"
        )
    else:
        print("No primary scores found; skipping score TSV generation.")


if __name__ == "__main__":
    main()

