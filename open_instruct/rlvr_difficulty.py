"""
Build a per-instance difficulty map from open-instruct rollout traces or
Hugging Face datasets with pass-rate aggregates.

The script accepts one or more local rollout directories, metadata ``.jsonl``
files, rollout shard ``.jsonl`` files written by ``open_instruct.rl_utils``,
or a Hugging Face dataset that already contains per-row pass counts. For each
prompt instance it:

1. loads rollout shards written by ``save_rollouts_to_disk()``, including compact score-only shards,
   or loads per-row pass counts from a Hub dataset,
2. groups attempts by source dataset identity when available, otherwise by a
   deterministic fingerprint over task name, prompt tokens, and ground truth,
3. normalizes binary verifiable rewards from ``{0, C}`` back to ``{0, 1}``
   when possible,
4. fits a Beta prior across binary outcomes and estimates per-item success
   rates, and
5. writes a JSONL difficulty file and schema/metadata sidecars.

Examples:
    uv run scripts/data/difficulty_sampling/create_bucketed_difficulty.py \
      --source /tmp/qwen_math_rollouts \
      --task math \
      --output /tmp/qwen_math_difficulty

    uv run scripts/data/difficulty_sampling/create_bucketed_difficulty.py \
      --source /tmp/qwen_math_rollouts/qwen_math_metadata.jsonl \
      --output /tmp/difficulty_map

    uv run scripts/data/difficulty_sampling/create_bucketed_difficulty.py \
      --hf-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
      --hf-split train \
      --output /tmp/dapo_math_qwen3_difficulty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta as beta_distribution

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


EPS = 1e-8
EXPERIMENT_METADATA_KEYS = ("source_root", "model_name", "experiment_id", "experiment_name")
JEFFREYS_PRIOR_ALPHA = 0.5
JEFFREYS_PRIOR_BETA = 0.5
DEFAULT_DIFFICULTY_BUCKETS = 5
POSTERIOR_QUANTILE_GRID_SIZE = 512
POSTERIOR_QUANTILE_BATCH_SIZE = 256
DIFFICULTY_GENERATION_METHOD = "beta_binomial_posterior_quantiles"
DIFFICULTY_METHOD_FILENAME_ALIASES = {DIFFICULTY_GENERATION_METHOD: "bbq"}
PRIOR_SOURCE_FILENAME_ALIASES = {"empirical_bayes": "eb", "jeffreys": "j", "jeffreys_fallback": "jf"}
ROLLOUT_SOURCE_FORMAT_KIND = "open_instruct_rollout_traces"
HF_SOURCE_FORMAT_KIND = "hugging_face_dataset_passrate_rows"
ROLLOUT_INSTANCE_ID_DEFINITION = (
    "source_dataset::source_dataset_id when available; otherwise sha1(task_name,prompt_tokens,ground_truth)"
)
HF_INSTANCE_ID_DEFINITION = (
    "dataset_repo_id::row_id_field when a stable row id is available; otherwise dataset_repo_id::row_index"
)
HF_SOURCE_ROW_INDEX_FIELD = "_source_row_index"
HF_OUTPUT_COLUMNS = ("difficulty",)


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float
    source: str


@dataclass(frozen=True)
class RolloutSource:
    input_arg: str
    root_path: Path
    metadata_path: Path
    rollout_paths: tuple[Path, ...]
    run_name: str


@dataclass(frozen=True)
class DifficultyPosteriorRow:
    row: dict[str, Any]
    difficulty_alpha: float
    difficulty_beta: float


@dataclass(frozen=True)
class InputRowsBundle:
    rows: list[dict[str, Any]]
    malformed_records: int
    source_format: dict[str, Any]
    source_dataset: Dataset | None = None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a per-instance difficulty map from open-instruct rollout traces or HF pass-rate datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        nargs="+",
        help="One or more local rollout dirs, *_metadata.jsonl files, or *_rollouts_*.jsonl shards.",
    )
    source_group.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Hugging Face dataset repo id containing per-row pass-rate aggregates.",
    )
    parser.add_argument("--hf-config", type=str, default=None, help="Optional dataset config for --hf-dataset.")
    parser.add_argument("--hf-split", type=str, default="train", help="Input split to load from --hf-dataset.")
    parser.add_argument(
        "--hf-row-id-field",
        type=str,
        default="extra_info.index",
        help="Dot-path to the stable per-row id field inside --hf-dataset.",
    )
    parser.add_argument(
        "--hf-task-field", type=str, default="dataset", help="Dot-path to the task/verifier field in --hf-dataset."
    )
    parser.add_argument(
        "--hf-model-field",
        type=str,
        default="generator_model",
        help="Dot-path to the generator model field in --hf-dataset.",
    )
    parser.add_argument(
        "--hf-pass-count-field",
        type=str,
        default="pass_count",
        help="Dot-path to the integer pass-count field in --hf-dataset.",
    )
    parser.add_argument(
        "--hf-attempt-count-field",
        type=str,
        default="num_samples",
        help="Dot-path to the total-attempt-count field in --hf-dataset.",
    )
    parser.add_argument(
        "--hf-pass-rate-field",
        type=str,
        default="pass_rate",
        help="Optional dot-path to a pass-rate or fraction field used for validation/fallback in --hf-dataset.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Optional task filter. Matches the rollout trace dataset/verifier source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output directory or path-like root. The script writes one file per task/model inside it as "
            "<task>__<model>__<tag>.jsonl plus matching .schema.json and .metadata.json sidecars."
        ),
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None, help="Optional dataset repo id to push the validated rows to."
    )
    parser.add_argument("--split", type=str, default="train", help="Split to use with --push-to-hub.")
    parser.add_argument(
        "--strict", action="store_true", help="Fail if a rollout record is malformed or required files are missing."
    )
    parser.add_argument(
        "--allow-nonunit-scores",
        action="store_true",
        help="Keep rows whose rewards cannot be normalized to binary correctness. Difficulty will be null for them.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap for the number of resolved instances written (useful for smoke tests).",
    )
    parser.add_argument(
        "--beta-prior",
        choices=["empirical-bayes", "jeffreys"],
        default="empirical-bayes",
        help="Global Beta prior to use for smoothing binary solve rates.",
    )
    parser.add_argument(
        "--posterior-lower-quantile",
        type=float,
        default=0.1,
        help="Lower posterior quantile used to define difficulty as 1 - quantile.",
    )
    parser.add_argument(
        "--difficulty-buckets",
        type=int,
        default=DEFAULT_DIFFICULTY_BUCKETS,
        help=(
            "Number of posterior-aware quantile buckets to assign for stratification. "
            "Set to 0 to skip discrete bucket assignment."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    validate_args(args)
    task_filters = set(args.task)
    output_root = resolve_output_root(args.output)

    input_rows = load_input_rows(args, task_filters=task_filters)

    if not input_rows.rows:
        raise ValueError("No resolved per-instance rows were produced.")

    rows = sorted(
        input_rows.rows,
        key=lambda row: (
            stable_string(row.get("task_name")),
            stable_string((row.get("experiment_metadata") or {}).get("model_name")),
            stable_string(row.get("instance_id")),
        ),
    )
    if args.max_instances is not None:
        rows = rows[: args.max_instances]

    rows_by_group = group_rows_by_task_and_model(rows)
    if args.push_to_hub is not None and len(rows_by_group) != 1:
        raise ValueError(
            "--push-to-hub requires a single task/model output. Filter with --task or use a source with one task."
        )

    skipped_nonunit = 0
    written_outputs: list[tuple[str, str | None, int, Path, Path, Path]] = []

    for (task_name, model_name), group_rows in sorted(
        rows_by_group.items(), key=lambda item: (item[0][0], stable_string(item[0][1]))
    ):
        group_rows, score_processing, group_skipped_nonunit = normalize_attempt_scores_for_group(
            group_rows, allow_nonunit_scores=args.allow_nonunit_scores
        )
        if input_rows.source_format["kind"] == HF_SOURCE_FORMAT_KIND:
            score_processing["source_field"] = ",".join(
                field_name
                for field_name in (
                    input_rows.source_format.get("pass_count_field"),
                    input_rows.source_format.get("attempt_count_field"),
                    input_rows.source_format.get("pass_rate_field"),
                )
                if field_name
            )
        skipped_nonunit += group_skipped_nonunit

        if not group_rows:
            logger.warning(
                "Skipping task=%s model=%s because no rows remained after reward normalization.", task_name, model_name
            )
            continue

        prior, binary_row_count = estimate_beta_prior(group_rows, prior_mode=args.beta_prior)
        group_rows = apply_beta_binomial_difficulty(
            group_rows, prior=prior, lower_quantile=args.posterior_lower_quantile, num_buckets=args.difficulty_buckets
        )
        if input_rows.source_dataset is None:
            ordered_group_rows = sorted(group_rows, key=lambda row: row["instance_id"])
            output_rows = strip_output_only_rollout_fields(ordered_group_rows)
            dataset = Dataset.from_list(output_rows)
        else:
            ordered_group_rows = sort_hf_group_rows(group_rows)
            output_rows = strip_internal_fields(ordered_group_rows)
            dataset = build_hf_output_dataset(input_rows.source_dataset, ordered_group_rows)

        dataset_metadata = build_dataset_metadata(
            rows=output_rows,
            task_name=task_name,
            model_name=model_name,
            requested_prior_mode=args.beta_prior,
            requested_bucket_count=args.difficulty_buckets,
            lower_quantile=args.posterior_lower_quantile,
            prior=prior,
            binary_row_count=binary_row_count,
            score_processing=score_processing,
            source_format=input_rows.source_format,
        )

        if prior is not None:
            logger.info(
                "Using %s Beta prior alpha=%.4f beta=%.4f across %s binary instances for task=%s model=%s.",
                prior.source,
                prior.alpha,
                prior.beta,
                binary_row_count,
                task_name,
                model_name,
            )
        else:
            logger.warning(
                "No binary instances were available for Beta-Binomial difficulty estimation for task=%s model=%s.",
                task_name,
                model_name,
            )

        annotate_dataset_metadata(dataset, dataset_metadata)
        output_jsonl, schema_json, metadata_json = build_output_paths(
            output_root, task_name=task_name, model_name=model_name, dataset_metadata=dataset_metadata
        )
        write_output_files(
            output_jsonl=output_jsonl,
            schema_json=schema_json,
            metadata_json=metadata_json,
            dataset=dataset,
            dataset_metadata=dataset_metadata,
        )

        if args.push_to_hub is not None:
            dataset.push_to_hub(args.push_to_hub, split=args.split, private=True)

        written_outputs.append((task_name, model_name, len(output_rows), output_jsonl, schema_json, metadata_json))
        logger.info(
            "Wrote %s rows for task=%s model=%s to %s, %s, and %s.",
            len(output_rows),
            task_name,
            model_name,
            output_jsonl,
            schema_json,
            metadata_json,
        )

    logger.info(
        "Finished writing %s output file groups (%s malformed rollout records, %s skipped due to unsupported scores).",
        len(written_outputs),
        input_rows.malformed_records,
        skipped_nonunit,
    )


def load_input_rows(args: argparse.Namespace, *, task_filters: set[str]) -> InputRowsBundle:
    if args.hf_dataset is not None:
        return load_hf_dataset_rows(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            split=args.hf_split,
            task_filters=task_filters,
            strict=args.strict,
            row_id_field=args.hf_row_id_field,
            task_field=args.hf_task_field,
            model_field=args.hf_model_field,
            pass_count_field=args.hf_pass_count_field,
            attempt_count_field=args.hf_attempt_count_field,
            pass_rate_field=args.hf_pass_rate_field,
        )

    if not args.source:
        raise ValueError("Expected --source when --hf-dataset is not provided.")

    source_runs = discover_rollout_sources(args.source)
    if not source_runs:
        raise ValueError("No rollout trace sources were found.")

    contributions: list[dict[str, Any]] = []
    malformed_records = 0

    for source_run in source_runs:
        logger.info(
            "Loading %s (run=%s, metadata=%s, shards=%s)",
            source_run.input_arg,
            source_run.run_name,
            source_run.metadata_path,
            len(source_run.rollout_paths),
        )
        run_contributions, run_malformed = build_contributions_for_source(
            source_run=source_run, task_filters=task_filters, strict=args.strict
        )
        contributions.extend(run_contributions)
        malformed_records += run_malformed

    return InputRowsBundle(
        rows=aggregate_contributions(contributions),
        malformed_records=malformed_records,
        source_format=build_rollout_source_format_metadata(),
    )


def load_hf_dataset_rows(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    task_filters: set[str],
    strict: bool,
    row_id_field: str,
    task_field: str,
    model_field: str,
    pass_count_field: str,
    attempt_count_field: str,
    pass_rate_field: str | None,
) -> InputRowsBundle:
    logger.info(
        "Loading Hugging Face dataset %s (config=%s, split=%s).", dataset_name, config_name or "default", split
    )

    if config_name:
        source_dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        source_dataset = load_dataset(dataset_name, split=split)

    rows: list[dict[str, Any]] = []
    malformed_records = 0

    for row_index, source_row in enumerate(source_dataset):
        try:
            row = build_hf_dataset_row(
                source_row=source_row,
                source_row_index=row_index,
                dataset_name=dataset_name,
                config_name=config_name,
                split=split,
                row_id_field=row_id_field,
                task_field=task_field,
                model_field=model_field,
                pass_count_field=pass_count_field,
                attempt_count_field=attempt_count_field,
                pass_rate_field=pass_rate_field,
            )
        except Exception as exc:
            malformed_records += 1
            message = f"Malformed HF dataset row {dataset_name}[{split}][{row_index}]: {exc}"
            if strict:
                raise ValueError(message) from exc
            logger.warning(message)
            continue

        task_name = stable_string(row.get("task_name"))
        if task_filters and task_name not in task_filters and get_base_task_name(task_name) not in task_filters:
            continue
        rows.append(row)

    return InputRowsBundle(
        rows=rows,
        malformed_records=malformed_records,
        source_format=build_hf_source_format_metadata(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            row_id_field=row_id_field,
            task_field=task_field,
            model_field=model_field,
            pass_count_field=pass_count_field,
            attempt_count_field=attempt_count_field,
            pass_rate_field=pass_rate_field,
        ),
        source_dataset=source_dataset,
    )


def build_hf_dataset_row(
    *,
    source_row: dict[str, Any],
    source_row_index: int,
    dataset_name: str,
    config_name: str | None,
    split: str,
    row_id_field: str,
    task_field: str,
    model_field: str,
    pass_count_field: str,
    attempt_count_field: str,
    pass_rate_field: str | None,
) -> dict[str, Any]:
    task_name = normalize_task_name(get_nested_field(source_row, task_field))
    if task_name is None:
        raise ValueError(f"missing task field {task_field!r}")

    source_row_id = normalize_identifier(get_nested_field(source_row, row_id_field)) or str(source_row_index)
    pass_count, attempt_count = extract_hf_attempt_summary(
        row=source_row,
        pass_count_field=pass_count_field,
        attempt_count_field=attempt_count_field,
        pass_rate_field=pass_rate_field,
    )
    model_name = optional_string(get_nested_field(source_row, model_field))

    return {
        HF_SOURCE_ROW_INDEX_FIELD: source_row_index,
        "instance_id": make_hf_instance_id(dataset_name=dataset_name, source_row_id=source_row_id),
        "task_name": task_name,
        "base_task_name": get_base_task_name(task_name),
        "source_dataset": dataset_name,
        "source_row_id": source_row_id,
        "attempt_scores": expand_binary_attempt_scores(pass_count=pass_count, attempt_count=attempt_count),
        "finish_reasons": [],
        "experiment_metadata": {
            "source_root": format_hf_source_locator(dataset_name=dataset_name, config_name=config_name, split=split),
            "model_name": model_name,
            "experiment_id": None,
            "experiment_name": dataset_name,
        },
        "score_sources": [task_name],
        "warnings": [],
    }


def build_rollout_source_format_metadata() -> dict[str, Any]:
    return {
        "kind": ROLLOUT_SOURCE_FORMAT_KIND,
        "task_field": "dataset",
        "score_field": "reward",
        "source_dataset_field": "source_dataset",
        "source_dataset_id_field": "source_dataset_id",
        "source_row_id_field": "source_row_id",
        "instance_id_definition": ROLLOUT_INSTANCE_ID_DEFINITION,
    }


def build_hf_source_format_metadata(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    row_id_field: str,
    task_field: str,
    model_field: str,
    pass_count_field: str,
    attempt_count_field: str,
    pass_rate_field: str | None,
) -> dict[str, Any]:
    return {
        "kind": HF_SOURCE_FORMAT_KIND,
        "dataset_repo_id": dataset_name,
        "config_name": config_name,
        "split": split,
        "row_id_field": row_id_field,
        "task_field": task_field,
        "model_field": model_field,
        "pass_count_field": pass_count_field,
        "attempt_count_field": attempt_count_field,
        "pass_rate_field": pass_rate_field,
        "instance_id_definition": HF_INSTANCE_ID_DEFINITION,
    }


def format_hf_source_locator(*, dataset_name: str, config_name: str | None, split: str) -> str:
    config_token = config_name or "default"
    return f"hf://{dataset_name}/{config_token}/{split}"


def make_hf_instance_id(*, dataset_name: str, source_row_id: str) -> str:
    return f"{dataset_name}::{source_row_id}"


def sort_hf_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row[HF_SOURCE_ROW_INDEX_FIELD])


def build_hf_output_dataset(source_dataset: Dataset, rows: list[dict[str, Any]]) -> Dataset:
    ordered_rows = sort_hf_group_rows(rows)
    dataset = source_dataset.select([row[HF_SOURCE_ROW_INDEX_FIELD] for row in ordered_rows])

    for column_name in HF_OUTPUT_COLUMNS:
        values = [make_jsonable(row.get(column_name)) for row in ordered_rows]
        if column_name in dataset.column_names:
            dataset = dataset.remove_columns(column_name)
        dataset = dataset.add_column(column_name, values)

    return dataset


def strip_internal_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != HF_SOURCE_ROW_INDEX_FIELD} for row in rows]


def get_nested_field(value: Any, field_path: str) -> Any:
    if not field_path:
        return value

    current = value
    for field_name in field_path.split("."):
        if not isinstance(current, dict) or field_name not in current:
            return None
        current = current[field_name]
    return current


def normalize_identifier(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = stable_string(value).strip()
    return text or None


def normalize_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def parse_pass_rate_value(value: Any) -> tuple[int | None, int | None, float | None]:
    if value is None:
        return None, None, None
    if is_number(value):
        rate = float(value)
        if 0.0 <= rate <= 1.0:
            return None, None, rate
        raise ValueError(f"expected pass-rate value in [0, 1], received {value!r}")
    if not isinstance(value, str):
        raise ValueError(f"unsupported pass-rate value {value!r}")

    stripped = value.strip()
    if not stripped:
        return None, None, None

    if "/" in stripped:
        numerator_text, denominator_text = stripped.split("/", 1)
        numerator = normalize_nonnegative_int(numerator_text)
        denominator = normalize_nonnegative_int(denominator_text)
        if numerator is None or denominator is None or numerator > denominator:
            raise ValueError(f"invalid pass-rate fraction {value!r}")
        rate = 0.0 if denominator == 0 else numerator / denominator
        return numerator, denominator, rate

    try:
        rate = float(stripped)
    except ValueError as exc:
        raise ValueError(f"invalid pass-rate value {value!r}") from exc
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError(f"expected pass-rate value in [0, 1], received {value!r}")
    return None, None, rate


def extract_hf_attempt_summary(
    *, row: dict[str, Any], pass_count_field: str, attempt_count_field: str, pass_rate_field: str | None
) -> tuple[int, int]:
    pass_count = normalize_nonnegative_int(get_nested_field(row, pass_count_field))
    attempt_count = normalize_nonnegative_int(get_nested_field(row, attempt_count_field))

    parsed_pass_count = None
    parsed_attempt_count = None
    parsed_pass_rate = None
    if pass_rate_field:
        parsed_pass_count, parsed_attempt_count, parsed_pass_rate = parse_pass_rate_value(
            get_nested_field(row, pass_rate_field)
        )

    if pass_count is None and parsed_pass_count is not None:
        pass_count = parsed_pass_count
    if attempt_count is None and parsed_attempt_count is not None:
        attempt_count = parsed_attempt_count

    if pass_count is None or attempt_count is None:
        raise ValueError(
            f"missing pass-count summary fields {pass_count_field!r}/{attempt_count_field!r}"
            f"{f' or parseable {pass_rate_field!r}' if pass_rate_field else ''}"
        )
    if attempt_count <= 0:
        raise ValueError(f"attempt count must be positive, received {attempt_count}")
    if pass_count > attempt_count:
        raise ValueError(f"pass count {pass_count} exceeds attempt count {attempt_count}")

    if parsed_pass_count is not None and parsed_pass_count != pass_count:
        raise ValueError(f"pass-count field {pass_count_field!r} disagrees with {pass_rate_field!r}")
    if parsed_attempt_count is not None and parsed_attempt_count != attempt_count:
        raise ValueError(f"attempt-count field {attempt_count_field!r} disagrees with {pass_rate_field!r}")
    if parsed_pass_rate is not None and not is_close(pass_count / attempt_count, parsed_pass_rate):
        raise ValueError(
            f"pass-count fields {pass_count_field!r}/{attempt_count_field!r} disagree with {pass_rate_field!r}"
        )

    return pass_count, attempt_count


def expand_binary_attempt_scores(*, pass_count: int, attempt_count: int) -> list[float]:
    return [1.0] * pass_count + [0.0] * (attempt_count - pass_count)


def discover_rollout_sources(sources: list[str]) -> list[RolloutSource]:
    discovered: dict[Path, RolloutSource] = {}

    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Could not find source path {source}")

        if source_path.is_dir():
            metadata_paths = sorted(source_path.rglob("*_metadata.jsonl"))
            if not metadata_paths:
                raise FileNotFoundError(f"Could not find *_metadata.jsonl under {source}")
            for metadata_path in metadata_paths:
                rollout_source = build_rollout_source_from_metadata(metadata_path, input_arg=source)
                discovered[rollout_source.metadata_path] = rollout_source
            continue

        if source_path.name.endswith("_metadata.jsonl"):
            rollout_source = build_rollout_source_from_metadata(source_path, input_arg=source)
            discovered[rollout_source.metadata_path] = rollout_source
            continue

        if source_path.suffix == ".jsonl" and "_rollouts_" in source_path.name:
            rollout_source = build_rollout_source_from_rollout(source_path, input_arg=source)
            discovered[rollout_source.metadata_path] = rollout_source
            continue

        raise ValueError(
            f"Unsupported source path {source}. Expected a directory, *_metadata.jsonl, or *_rollouts_*.jsonl."
        )

    return sorted(discovered.values(), key=lambda source_run: (str(source_run.root_path), source_run.run_name))


def build_rollout_source_from_metadata(metadata_path: Path, *, input_arg: str) -> RolloutSource:
    run_name = parse_run_name_from_metadata_path(metadata_path)
    rollout_paths = tuple(sorted(metadata_path.parent.glob(f"{run_name}_rollouts_*.jsonl")))
    if not rollout_paths:
        raise FileNotFoundError(f"Could not find rollout shards for run {run_name} next to {metadata_path}")
    return RolloutSource(
        input_arg=input_arg,
        root_path=metadata_path.parent.absolute(),
        metadata_path=metadata_path.absolute(),
        rollout_paths=rollout_paths,
        run_name=run_name,
    )


def build_rollout_source_from_rollout(rollout_path: Path, *, input_arg: str) -> RolloutSource:
    run_name = parse_run_name_from_rollout_path(rollout_path)
    metadata_path = rollout_path.parent / f"{run_name}_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata file {metadata_path} for rollout shard {rollout_path}")
    return build_rollout_source_from_metadata(metadata_path, input_arg=input_arg)


def parse_run_name_from_metadata_path(metadata_path: Path) -> str:
    suffix = "_metadata.jsonl"
    if not metadata_path.name.endswith(suffix):
        raise ValueError(f"Metadata path must end with {suffix}: {metadata_path}")
    return metadata_path.name[: -len(suffix)]


def parse_run_name_from_rollout_path(rollout_path: Path) -> str:
    marker = "_rollouts_"
    if marker not in rollout_path.name:
        raise ValueError(f"Rollout shard filename must contain {marker}: {rollout_path}")
    return rollout_path.name.split(marker, 1)[0]


def build_contributions_for_source(
    *, source_run: RolloutSource, task_filters: set[str], strict: bool
) -> tuple[list[dict[str, Any]], int]:
    run_metadata = read_rollout_metadata(source_run.metadata_path, fallback_run_name=source_run.run_name)
    contributions: list[dict[str, Any]] = []
    malformed_records = 0

    for rollout_path in source_run.rollout_paths:
        for line_number, record in enumerate(read_jsonl(rollout_path), start=1):
            try:
                contribution = build_rollout_contribution(
                    record=record, source_run=source_run, run_metadata=run_metadata
                )
            except Exception as exc:
                malformed_records += 1
                message = f"Malformed rollout record in {rollout_path}:{line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                logger.warning(message)
                continue

            task_name = stable_string(contribution.get("task_name"))
            if task_filters and task_name not in task_filters and get_base_task_name(task_name) not in task_filters:
                continue
            contributions.append(contribution)

    return contributions, malformed_records


def read_rollout_metadata(metadata_path: Path, *, fallback_run_name: str) -> dict[str, Any]:
    rows = read_jsonl(metadata_path)
    if not rows:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    if len(rows) > 1:
        logger.warning("Expected one metadata row in %s but found %s. Using the first row.", metadata_path, len(rows))

    metadata = rows[0]
    return {
        "run_name": optional_string(metadata.get("run_name")) or fallback_run_name,
        "model_name": optional_string(metadata.get("model_name")),
        "experiment_id": optional_string(metadata.get("experiment_id")),
        "git_commit": optional_string(metadata.get("git_commit")),
        "timestamp": optional_string(metadata.get("timestamp")),
    }


def build_rollout_contribution(
    *, record: dict[str, Any], source_run: RolloutSource, run_metadata: dict[str, Any]
) -> dict[str, Any]:
    task_name = normalize_task_name(record.get("dataset"))
    if task_name is None:
        raise ValueError("missing dataset/verifier source")

    source_dataset = normalize_source_dataset(record.get("source_dataset"))
    source_dataset_id = extract_source_dataset_id(record)

    prompt_tokens = normalize_token_list(record.get("prompt_tokens"))
    if prompt_tokens is None and (source_dataset is None or source_dataset_id is None):
        raise ValueError("missing prompt_tokens and source dataset identity (source_dataset/source_row_id)")

    reward = extract_numeric_reward(record.get("reward"))
    if reward is None:
        raise ValueError("missing or invalid reward")

    ground_truth = make_jsonable(record.get("ground_truth"))
    finish_reason = optional_string(record.get("finish_reason"))

    return {
        "instance_id": make_rollout_instance_id(
            task_name=task_name,
            prompt_tokens=prompt_tokens,
            ground_truth=ground_truth,
            source_dataset=source_dataset,
            source_dataset_id=source_dataset_id,
        ),
        "task_name": task_name,
        "base_task_name": get_base_task_name(task_name),
        "prompt_tokens": prompt_tokens,
        "ground_truth": ground_truth,
        "source_dataset": source_dataset,
        "source_dataset_id": source_dataset_id,
        "score_source": task_name,
        "attempt_scores": [reward],
        "finish_reasons": [finish_reason] if finish_reason else [],
        "experiment_metadata": {
            "source_root": str(source_run.root_path),
            "model_name": run_metadata["model_name"],
            "experiment_id": run_metadata["experiment_id"],
            "experiment_name": run_metadata["run_name"],
        },
        "warnings": extract_rollout_warnings(record.get("request_info")),
    }


def normalize_task_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalize_task_name(value[0])
    serialized = serialize_value(value)
    return serialized or None


def normalize_source_dataset(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalize_source_dataset(value[0])
    serialized = serialize_value(value)
    return serialized or None


def extract_source_dataset_id(record: dict[str, Any]) -> int | None:
    for field_name in ("source_dataset_id", "source_row_id"):
        source_dataset_id = normalize_source_dataset_id(record.get(field_name))
        if source_dataset_id is not None:
            return source_dataset_id
    return None


def normalize_source_dataset_id(value: Any) -> int | None:
    return normalize_nonnegative_int(value)


def normalize_token_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None

    tokens: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        tokens.append(int(item))
    return tokens


def extract_numeric_reward(value: Any) -> float | None:
    if not is_number(value):
        return None
    return float(value)


def extract_rollout_warnings(request_info: Any) -> list[str]:
    if not isinstance(request_info, dict):
        return []

    warnings: list[str] = []
    if request_info.get("timeouts"):
        warnings.append("timeout")
    if optional_string(request_info.get("tool_errors")):
        warnings.append("tool_error")
    return warnings


def aggregate_contributions(contributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for contribution in contributions:
        instance_id = contribution["instance_id"]
        if instance_id not in grouped:
            grouped[instance_id] = {
                key: value
                for key, value in contribution.items()
                if key not in {"attempt_scores", "finish_reasons", "experiment_metadata", "warnings", "score_source"}
            }
            grouped[instance_id]["attempt_scores"] = []
            grouped[instance_id]["finish_reasons"] = []
            grouped[instance_id]["experiment_metadata"] = None
            grouped[instance_id]["score_sources"] = set()
            grouped[instance_id]["warnings"] = set()

        row = grouped[instance_id]
        row["attempt_scores"].extend(float(score) for score in contribution["attempt_scores"])
        row["finish_reasons"].extend(contribution["finish_reasons"])
        row["experiment_metadata"] = merge_experiment_metadata(
            existing=row["experiment_metadata"], incoming=contribution["experiment_metadata"], instance_id=instance_id
        )
        row["score_sources"].add(stable_string(contribution["score_source"]))
        row["warnings"].update(contribution["warnings"])

    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        row["attempt_scores"] = [float(score) for score in row["attempt_scores"]]
        row["finish_reasons"] = [stable_string(reason) for reason in row["finish_reasons"] if stable_string(reason)]
        row["experiment_metadata"] = normalize_experiment_metadata(row["experiment_metadata"])
        row["score_sources"] = sorted(value for value in row["score_sources"] if value)
        row["warnings"] = sorted(value for value in row["warnings"] if value)
        rows.append(row)

    return rows


def strip_output_only_rollout_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key not in {"prompt_tokens", "ground_truth"}} for row in rows]


def normalize_attempt_scores_for_group(
    rows: list[dict[str, Any]], *, allow_nonunit_scores: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    score_processing = infer_score_processing(rows)
    normalized_rows: list[dict[str, Any]] = []
    skipped_nonunit = 0

    for row in rows:
        normalized_scores = normalize_attempt_scores(row["attempt_scores"], score_processing)
        if normalized_scores is None:
            if allow_nonunit_scores:
                kept_row = dict(row)
                kept_row["attempt_scores"] = [float(score) for score in row["attempt_scores"]]
                kept_row["warnings"] = sorted({*kept_row["warnings"], "nonbinary_reward_scores"})
                normalized_rows.append(kept_row)
            else:
                skipped_nonunit += 1
            continue

        normalized_row = dict(row)
        normalized_row["attempt_scores"] = normalized_scores
        normalized_rows.append(normalized_row)

    return normalized_rows, score_processing, skipped_nonunit


def infer_score_processing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(score) for row in rows for score in row.get("attempt_scores", [])]
    score_processing = {
        "source_field": "reward",
        "output_field": "attempt_scores",
        "normalization": "unsupported",
        "positive_reward_value": None,
        "supports_binary_difficulty": False,
    }

    if not scores:
        return score_processing

    if all(is_close(score, 0.0) or is_close(score, 1.0) for score in scores):
        score_processing["normalization"] = "identity_binary"
        score_processing["positive_reward_value"] = 1.0
        score_processing["supports_binary_difficulty"] = True
        return score_processing

    if any(score < -EPS for score in scores):
        return score_processing

    positive_scores = [score for score in scores if score > EPS]
    if not positive_scores:
        score_processing["normalization"] = "all_zero_binary"
        score_processing["supports_binary_difficulty"] = True
        return score_processing

    positive_reward_value = max(positive_scores)
    if all(is_close(score, 0.0) or is_close(score, positive_reward_value) for score in scores):
        score_processing["normalization"] = "binary_zero_or_constant"
        score_processing["positive_reward_value"] = positive_reward_value
        score_processing["supports_binary_difficulty"] = True

    return score_processing


def normalize_attempt_scores(attempt_scores: list[float], score_processing: dict[str, Any]) -> list[float] | None:
    if not score_processing.get("supports_binary_difficulty"):
        return None

    normalization = stable_string(score_processing.get("normalization"))
    positive_reward_value = score_processing.get("positive_reward_value")
    normalized_scores: list[float] = []

    for score in attempt_scores:
        if is_close(score, 0.0):
            normalized_scores.append(0.0)
            continue

        if normalization == "identity_binary" and is_close(score, 1.0):
            normalized_scores.append(1.0)
            continue

        if (
            normalization == "binary_zero_or_constant"
            and positive_reward_value is not None
            and is_close(score, float(positive_reward_value))
        ):
            normalized_scores.append(1.0)
            continue

        if normalization == "all_zero_binary":
            return None

        return None

    return normalized_scores


def estimate_beta_prior(rows: list[dict[str, Any]], *, prior_mode: str) -> tuple[BetaPrior | None, int]:
    binary_counts = [counts for row in rows if (counts := extract_binary_counts(row["attempt_scores"])) is not None]
    if not binary_counts:
        return None, 0

    if prior_mode == "jeffreys":
        return BetaPrior(JEFFREYS_PRIOR_ALPHA, JEFFREYS_PRIOR_BETA, "jeffreys"), len(binary_counts)

    prior = fit_empirical_beta_prior(binary_counts)
    if prior is not None:
        return prior, len(binary_counts)

    logger.warning("Falling back to Jeffreys prior after empirical-Bayes fitting failed.")
    return BetaPrior(JEFFREYS_PRIOR_ALPHA, JEFFREYS_PRIOR_BETA, "jeffreys_fallback"), len(binary_counts)


def apply_beta_binomial_difficulty(
    rows: list[dict[str, Any]], *, prior: BetaPrior | None, lower_quantile: float, num_buckets: int
) -> list[dict[str, Any]]:
    posterior_rows: list[DifficultyPosteriorRow] = []

    for row in rows:
        row["difficulty"] = make_empty_difficulty_payload()

        if prior is None:
            continue

        binary_counts = extract_binary_counts(row["attempt_scores"])
        if binary_counts is None:
            continue

        success_count, attempt_count = binary_counts
        posterior_alpha = success_count + prior.alpha
        posterior_beta = attempt_count - success_count + prior.beta
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_lower_bound = float(beta_distribution.ppf(lower_quantile, posterior_alpha, posterior_beta))

        row["difficulty"] = {
            "value": max(0.0, min(1.0, 1.0 - posterior_lower_bound)),
            "posterior_mean": posterior_mean,
            "posterior_lower_bound": posterior_lower_bound,
            "expected_quantile": None,
            "bucket_index": None,
            "bucket_count": None,
        }
        posterior_rows.append(
            DifficultyPosteriorRow(row=row, difficulty_alpha=posterior_beta, difficulty_beta=posterior_alpha)
        )

    assign_posterior_difficulty_buckets(posterior_rows, num_buckets=num_buckets)
    return rows


def make_empty_difficulty_payload() -> dict[str, Any]:
    return {
        "value": None,
        "posterior_mean": None,
        "posterior_lower_bound": None,
        "expected_quantile": None,
        "bucket_index": None,
        "bucket_count": None,
    }


def assign_posterior_difficulty_buckets(posterior_rows: list[DifficultyPosteriorRow], *, num_buckets: int) -> None:
    if not posterior_rows:
        return

    expected_quantiles = estimate_expected_difficulty_quantiles(posterior_rows)
    for posterior_row, expected_quantile in zip(posterior_rows, expected_quantiles, strict=True):
        posterior_row.row["difficulty"]["expected_quantile"] = expected_quantile

    if num_buckets <= 0:
        return

    effective_bucket_count = min(num_buckets, len(posterior_rows))
    ordered_rows = sorted(
        zip(posterior_rows, expected_quantiles, strict=True),
        key=lambda item: (item[1], item[0].row["difficulty"]["value"], stable_string(item[0].row["instance_id"])),
    )
    base_bucket_size, remainder = divmod(len(ordered_rows), effective_bucket_count)

    cursor = 0
    for bucket_index in range(effective_bucket_count):
        bucket_size = base_bucket_size + (1 if bucket_index < remainder else 0)
        for posterior_row, _expected_quantile in ordered_rows[cursor : cursor + bucket_size]:
            posterior_row.row["difficulty"]["bucket_index"] = bucket_index
            posterior_row.row["difficulty"]["bucket_count"] = effective_bucket_count
        cursor += bucket_size


def estimate_expected_difficulty_quantiles(
    posterior_rows: list[DifficultyPosteriorRow],
    *,
    grid_size: int = POSTERIOR_QUANTILE_GRID_SIZE,
    batch_size: int = POSTERIOR_QUANTILE_BATCH_SIZE,
) -> list[float]:
    if not posterior_rows:
        return []
    if len(posterior_rows) == 1:
        return [0.5]

    grid = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    difficulty_alphas = np.asarray([row.difficulty_alpha for row in posterior_rows], dtype=np.float64)
    difficulty_betas = np.asarray([row.difficulty_beta for row in posterior_rows], dtype=np.float64)

    mixture_cdf = np.zeros(grid_size, dtype=np.float64)
    for start in range(0, len(posterior_rows), batch_size):
        stop = start + batch_size
        batch_cdf = beta_distribution.cdf(
            grid[None, :], difficulty_alphas[start:stop, None], difficulty_betas[start:stop, None]
        )
        mixture_cdf += np.nan_to_num(batch_cdf, nan=0.0, posinf=1.0, neginf=0.0).sum(axis=0)
    mixture_cdf /= len(posterior_rows)

    quantiles = np.zeros(len(posterior_rows), dtype=np.float64)
    dx = 1.0 / grid_size
    for start in range(0, len(posterior_rows), batch_size):
        stop = start + batch_size
        batch_pdf = beta_distribution.pdf(
            grid[None, :], difficulty_alphas[start:stop, None], difficulty_betas[start:stop, None]
        )
        quantiles[start:stop] = np.clip(
            np.nan_to_num(batch_pdf, nan=0.0, posinf=0.0, neginf=0.0).dot(mixture_cdf) * dx, 0.0, 1.0
        )

    return quantiles.tolist()


def fit_empirical_beta_prior(binary_counts: list[tuple[int, int]]) -> BetaPrior | None:
    total_successes = sum(success_count for success_count, _ in binary_counts)
    total_attempts = sum(attempt_count for _, attempt_count in binary_counts)
    if total_attempts == 0 or total_successes in {0, total_attempts}:
        return None

    mean_rate = total_successes / total_attempts
    init_alpha = max(mean_rate * 2.0, 1e-3)
    init_beta = max((1.0 - mean_rate) * 2.0, 1e-3)

    def objective(log_params: tuple[float, float]) -> float:
        alpha = math.exp(log_params[0])
        beta = math.exp(log_params[1])
        return -sum(
            betaln(success_count + alpha, attempt_count - success_count + beta) - betaln(alpha, beta)
            for success_count, attempt_count in binary_counts
        )

    result = minimize(
        objective,
        x0=(math.log(init_alpha), math.log(init_beta)),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    )
    if not result.success:
        logger.warning("Empirical-Bayes fit failed: %s", result.message)
        return None

    return BetaPrior(alpha=math.exp(result.x[0]), beta=math.exp(result.x[1]), source="empirical_bayes")


def merge_experiment_metadata(
    existing: dict[str, Any] | None, incoming: dict[str, Any], *, instance_id: str
) -> dict[str, Any]:
    normalized_incoming = normalize_experiment_metadata(incoming)
    if existing is None:
        return normalized_incoming

    merged = dict(existing)
    for key in EXPERIMENT_METADATA_KEYS:
        existing_value = merged.get(key)
        incoming_value = normalized_incoming.get(key)
        if existing_value in {None, ""}:
            merged[key] = incoming_value
        elif incoming_value in {None, ""} or incoming_value == existing_value:
            continue
        else:
            raise ValueError(
                f"Conflicting experiment metadata for instance {instance_id}: "
                f"{key}={existing_value!r} vs {incoming_value!r}"
            )
    return merged


def normalize_experiment_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {key: None for key in EXPERIMENT_METADATA_KEYS}
    return {key: metadata.get(key) for key in EXPERIMENT_METADATA_KEYS}


def resolve_output_root(output: Path) -> Path:
    output_str = str(output)
    if output_str.endswith(".schema.json"):
        return Path(output_str[: -len(".schema.json")])
    if output_str.endswith(".jsonl"):
        return Path(output_str[: -len(".jsonl")])
    if output_str.endswith(".json"):
        return Path(output_str[: -len(".json")])
    return output


def build_output_paths(
    output_root: Path, *, task_name: str, model_name: str | None, dataset_metadata: dict[str, Any]
) -> tuple[Path, Path, Path]:
    task_suffix = sanitize_name(task_name) or "unknown-task"
    model_suffix = sanitize_name(model_name or "") or "unknown-model"
    difficulty_suffix = build_difficulty_filename_suffix(dataset_metadata)
    stem = output_root / f"{task_suffix}__{model_suffix}{difficulty_suffix}"
    return Path(f"{stem}.jsonl"), Path(f"{stem}.schema.json"), Path(f"{stem}.metadata.json")


def write_output_files(
    *, output_jsonl: Path, schema_json: Path, metadata_json: Path, dataset: Dataset, dataset_metadata: dict[str, Any]
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as output_file:
        for row in dataset:
            output_file.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")

    schema_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        schema_payload: Any = dataset.features.to_dict()
    except AttributeError:
        schema_payload = str(dataset.features)
    with schema_json.open("w") as output_file:
        json.dump(schema_payload, output_file, indent=2, sort_keys=True)

    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with metadata_json.open("w") as output_file:
        json.dump(dataset_metadata, output_file, indent=2, sort_keys=True)


def build_dataset_metadata(
    *,
    rows: list[dict[str, Any]],
    task_name: str,
    model_name: str | None,
    requested_prior_mode: str,
    requested_bucket_count: int,
    lower_quantile: float,
    prior: BetaPrior | None,
    binary_row_count: int,
    score_processing: dict[str, Any],
    source_format: dict[str, Any],
) -> dict[str, Any]:
    effective_bucket_count = extract_effective_bucket_count(rows)
    difficulty_generation = {
        "method": DIFFICULTY_GENERATION_METHOD,
        "difficulty_value_field": "difficulty.value",
        "difficulty_value_definition": "1 - difficulty.posterior_lower_bound",
        "bucket_field": "difficulty.bucket_index",
        "bucket_count_field": "difficulty.bucket_count",
        "bucket_ranking_field": "difficulty.expected_quantile",
        "posterior_lower_quantile": lower_quantile,
        "bucket_count_requested": requested_bucket_count,
        "bucket_count_effective": effective_bucket_count,
        "beta_prior_requested": requested_prior_mode,
        "beta_prior_used": {
            "source": prior.source if prior is not None else None,
            "alpha": prior.alpha if prior is not None else None,
            "beta": prior.beta if prior is not None else None,
        },
        "binary_instance_count": binary_row_count,
        "nonbinary_instance_count": max(0, len(rows) - binary_row_count),
    }
    difficulty_generation["tag"] = build_difficulty_config_tag(difficulty_generation)
    return {
        "task_name": task_name,
        "model_name": model_name,
        "row_count": len(rows),
        "source_format": dict(source_format),
        "score_processing": dict(score_processing),
        "difficulty_generation": difficulty_generation,
    }


def extract_effective_bucket_count(rows: list[dict[str, Any]]) -> int:
    effective_bucket_counts = {
        difficulty.get("bucket_count")
        for row in rows
        if isinstance((difficulty := row.get("difficulty")), dict) and difficulty.get("bucket_count") is not None
    }
    if not effective_bucket_counts:
        return 0
    if len(effective_bucket_counts) != 1:
        raise ValueError(f"Expected a single effective bucket count, found {sorted(effective_bucket_counts)}")
    return next(iter(effective_bucket_counts))


def build_difficulty_filename_suffix(dataset_metadata: dict[str, Any]) -> str:
    return f"__{dataset_metadata['difficulty_generation']['tag']}"


def build_difficulty_config_tag(difficulty_generation: dict[str, Any]) -> str:
    method_token = abbreviate_filename_token(
        optional_string(difficulty_generation.get("method")),
        aliases=DIFFICULTY_METHOD_FILENAME_ALIASES,
        default="diff",
    )
    prior_source = optional_string((difficulty_generation.get("beta_prior_used") or {}).get("source"))
    prior_token = abbreviate_filename_token(prior_source, aliases=PRIOR_SOURCE_FILENAME_ALIASES, default="none")
    quantile_token = format_quantile_token(difficulty_generation["posterior_lower_quantile"])
    bucket_token = format_bucket_token(
        requested_count=difficulty_generation["bucket_count_requested"],
        effective_count=difficulty_generation["bucket_count_effective"],
    )
    return "-".join([method_token, prior_token, quantile_token, bucket_token])


def abbreviate_filename_token(value: str | None, *, aliases: dict[str, str], default: str) -> str:
    if not value:
        return default
    return aliases.get(value, sanitize_name(value))


def format_quantile_token(value: float) -> str:
    return f"q{format_filename_number(value * 100.0)}"


def format_bucket_token(*, requested_count: int, effective_count: int) -> str:
    if requested_count == effective_count:
        return f"k{requested_count}"
    return f"k{requested_count}e{effective_count}"


def annotate_dataset_metadata(dataset: Dataset, dataset_metadata: dict[str, Any]) -> None:
    if not hasattr(dataset, "info") or dataset.info is None:
        return
    dataset.info.description = json.dumps(dataset_metadata, indent=2, sort_keys=True)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.posterior_lower_quantile < 1.0:
        raise ValueError("--posterior-lower-quantile must be between 0 and 1.")
    if args.difficulty_buckets < 0:
        raise ValueError("--difficulty-buckets must be non-negative.")
    if args.max_instances is not None and args.max_instances <= 0:
        raise ValueError("--max-instances must be positive when provided.")


def group_rows_by_task_and_model(rows: list[dict[str, Any]]) -> dict[tuple[str, str | None], list[dict[str, Any]]]:
    rows_by_group: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        experiment_metadata = row.get("experiment_metadata") or {}
        task_name = stable_string(row.get("task_name"))
        model_name = optional_string(experiment_metadata.get("model_name"))
        rows_by_group[(task_name, model_name)].append(row)
    return dict(rows_by_group)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def get_base_task_name(task_name: str) -> str:
    return task_name.split("@", 1)[0].split(":", 1)[0]


def extract_binary_counts(attempt_scores: list[float]) -> tuple[int, int] | None:
    if not attempt_scores:
        return None

    success_count = 0
    for score in attempt_scores:
        if is_close(score, 0.0):
            continue
        if is_close(score, 1.0):
            success_count += 1
            continue
        return None

    return success_count, len(attempt_scores)


def make_rollout_instance_id(
    *,
    task_name: str,
    prompt_tokens: list[int] | None,
    ground_truth: Any,
    source_dataset: str | None = None,
    source_dataset_id: int | None = None,
) -> str:
    if source_dataset is not None and source_dataset_id is not None:
        return f"{source_dataset}::{source_dataset_id}"

    if prompt_tokens is None:
        raise ValueError("prompt_tokens are required when source row identity is unavailable")

    fingerprint = {"task_name": task_name, "prompt_tokens": prompt_tokens, "ground_truth": make_jsonable(ground_truth)}
    digest = hashlib.sha1(canonical_json(fingerprint).encode("utf-8")).hexdigest()[:20]
    task_prefix = sanitize_name(task_name) or "unknown"
    return f"{task_prefix}::{digest}"


def canonical_json(value: Any) -> str:
    return json.dumps(make_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {stable_string(key): make_jsonable(item) for key, item in value.items()}
    return stable_string(value)


def stable_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def optional_string(value: Any) -> str | None:
    text = stable_string(value)
    return text or None


def serialize_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(make_jsonable(value), ensure_ascii=False, sort_keys=True)


def format_filename_number(value: float) -> str:
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p")


def sanitize_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value))


def is_close(lhs: float, rhs: float) -> bool:
    tolerance = EPS * max(1.0, abs(lhs), abs(rhs))
    return abs(lhs - rhs) <= tolerance


if __name__ == "__main__":
    main()
