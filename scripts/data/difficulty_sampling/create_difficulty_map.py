#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "datasets>=4.0.0",
#   "numpy>=2",
#   "scipy>=1.14.0",
# ]
# ///

"""
Build a per-instance difficulty map from Hugging Face datasets with pass-rate
aggregates.

The script loads a Hugging Face dataset that already contains per-row pass
counts, expands those counts into binary attempt outcomes, fits a Beta prior
across binary outcomes, estimates per-item difficulty, and writes JSONL
difficulty files plus schema/metadata sidecars. When `--push-to-hub` is set, it
also uploads the validated output dataset to the requested Hugging Face repo.
Hub uploads require exactly one task/model output group, so use `--task` or a
single-group input dataset when pushing.

Examples:
    Write local difficulty files:
    uv run scripts/data/difficulty_sampling/create_difficulty_map.py \
      --hf-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
      --hf-split train \
      --output /tmp/dapo_math_qwen3_difficulty

    Write local files and push the single output group to the Hub:
    uv run scripts/data/difficulty_sampling/create_difficulty_map.py \
      --hf-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
      --hf-split train \
      --task math \
      --output /tmp/dapo_math_qwen3_difficulty \
      --push-to-hub your-org/dapo-math-qwen3-difficulty \
      --split train
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
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
JEFFREYS_PRIOR_ALPHA = 0.5
JEFFREYS_PRIOR_BETA = 0.5
DEFAULT_DIFFICULTY_BUCKETS = 5
POSTERIOR_QUANTILE_GRID_SIZE = 512
POSTERIOR_QUANTILE_BATCH_SIZE = 256
DIFFICULTY_GENERATION_METHOD = "beta_binomial_posterior_quantiles"
DIFFICULTY_METHOD_FILENAME_ALIASES = {DIFFICULTY_GENERATION_METHOD: "bbq"}
PRIOR_SOURCE_FILENAME_ALIASES = {"empirical_bayes": "eb", "jeffreys": "j", "jeffreys_fallback": "jf"}
HF_SOURCE_FORMAT_KIND = "hugging_face_dataset_passrate_rows"
HF_INSTANCE_ID_DEFINITION = (
    "dataset_repo_id::row_id_field when a stable row id is available; otherwise dataset_repo_id::row_index"
)
HF_OUTPUT_COLUMNS = ("difficulty",)


@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float
    source: str


@dataclass
class ExperimentMetadata:
    source_root: str
    model_name: str | None
    experiment_id: str | None
    experiment_name: str


@dataclass
class DifficultyPayload:
    value: float | None = None
    posterior_mean: float | None = None
    posterior_lower_bound: float | None = None
    expected_quantile: float | None = None
    bucket_index: int | None = None
    bucket_count: int | None = None


@dataclass
class DifficultyRow:
    source_row_index: int
    instance_id: str
    task_name: str
    base_task_name: str
    source_dataset: str
    source_row_id: str
    attempt_scores: list[float]
    finish_reasons: list[str]
    experiment_metadata: ExperimentMetadata
    score_sources: list[str]
    warnings: list[str]
    difficulty: DifficultyPayload = field(default_factory=DifficultyPayload)


@dataclass(frozen=True)
class SourceFormatMetadata:
    kind: str
    dataset_repo_id: str
    config_name: str | None
    split: str
    row_id_field: str
    task_field: str
    model_field: str
    pass_count_field: str
    attempt_count_field: str
    pass_rate_field: str | None
    instance_id_definition: str


@dataclass
class ScoreProcessingMetadata:
    source_field: str = "reward"
    output_field: str = "attempt_scores"
    normalization: str = "unsupported"
    positive_reward_value: float | None = None
    supports_binary_difficulty: bool = False


@dataclass(frozen=True)
class BetaPriorUsageMetadata:
    source: str | None
    alpha: float | None
    beta: float | None


@dataclass
class DifficultyGenerationMetadata:
    method: str
    difficulty_value_field: str
    difficulty_value_definition: str
    bucket_field: str
    bucket_count_field: str
    bucket_ranking_field: str
    posterior_lower_quantile: float
    bucket_count_requested: int
    bucket_count_effective: int
    beta_prior_requested: str
    beta_prior_used: BetaPriorUsageMetadata
    binary_instance_count: int
    nonbinary_instance_count: int
    tag: str | None = None


@dataclass
class DatasetMetadata:
    task_name: str
    model_name: str | None
    row_count: int
    source_format: SourceFormatMetadata
    score_processing: ScoreProcessingMetadata
    difficulty_generation: DifficultyGenerationMetadata


@dataclass(frozen=True)
class DifficultyPosteriorRow:
    row: DifficultyRow
    difficulty_alpha: float
    difficulty_beta: float


@dataclass(frozen=True)
class InputRowsBundle:
    rows: list[DifficultyRow]
    malformed_records: int
    source_format: SourceFormatMetadata
    source_dataset: Dataset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a per-instance difficulty map from HF pass-rate datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
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
        "--task", action="append", default=[], help="Optional task filter. Matches the dataset task/verifier source."
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
        "--push-to-hub",
        type=str,
        default=None,
        help=("Optional dataset repo id to push the validated rows to. Requires exactly one task/model output group."),
    )
    parser.add_argument("--split", type=str, default="train", help="Split to use with --push-to-hub.")
    parser.add_argument("--strict", action="store_true", help="Fail if an input dataset row is malformed.")
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
    output_root = args.output
    output_root_str = str(output_root)
    for suffix in (".schema.json", ".jsonl", ".json"):
        if output_root_str.endswith(suffix):
            output_root = Path(output_root_str[: -len(suffix)])
            break

    input_rows = load_hf_dataset_rows(
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

    if not input_rows.rows:
        raise ValueError("No resolved per-instance rows were produced.")

    rows = sorted(
        input_rows.rows, key=lambda row: (row.task_name, row.experiment_metadata.model_name or "", row.instance_id)
    )
    if args.max_instances is not None:
        rows = rows[: args.max_instances]

    rows_by_group = group_rows_by_task_and_model(rows)
    if args.push_to_hub is not None and len(rows_by_group) != 1:
        raise ValueError(
            "--push-to-hub requires a single task/model output. Filter with --task or use a dataset with one task."
        )

    score_source_field = ",".join(
        field_name
        for field_name in (
            input_rows.source_format.pass_count_field,
            input_rows.source_format.attempt_count_field,
            input_rows.source_format.pass_rate_field,
        )
        if field_name
    )
    skipped_nonunit = 0
    written_outputs: list[tuple[str, str | None, int, Path, Path, Path]] = []

    for (task_name, model_name), group_rows in sorted(
        rows_by_group.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        group_rows, score_processing, group_skipped_nonunit = normalize_attempt_scores_for_group(
            group_rows, allow_nonunit_scores=args.allow_nonunit_scores
        )
        score_processing.source_field = score_source_field
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
        output_rows, dataset = build_hf_output_dataset(input_rows.source_dataset, group_rows)

        dataset_metadata = build_dataset_metadata(
            rows=group_rows,
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
            len(group_rows),
            task_name,
            model_name,
            output_jsonl,
            schema_json,
            metadata_json,
        )

    logger.info(
        "Finished writing %s output file groups (%s malformed dataset rows, %s skipped due to unsupported scores).",
        len(written_outputs),
        input_rows.malformed_records,
        skipped_nonunit,
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

    rows: list[DifficultyRow] = []
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

        task_name = row.task_name
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
) -> DifficultyRow:
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
    raw_model_name = get_nested_field(source_row, model_field)
    model_name = None if raw_model_name is None else str(raw_model_name) or None

    return DifficultyRow(
        source_row_index=source_row_index,
        instance_id=f"{dataset_name}::{source_row_id}",
        task_name=task_name,
        base_task_name=get_base_task_name(task_name),
        source_dataset=dataset_name,
        source_row_id=source_row_id,
        attempt_scores=expand_binary_attempt_scores(pass_count=pass_count, attempt_count=attempt_count),
        finish_reasons=[],
        experiment_metadata=ExperimentMetadata(
            source_root=f"hf://{dataset_name}/{config_name or 'default'}/{split}",
            model_name=model_name,
            experiment_id=None,
            experiment_name=dataset_name,
        ),
        score_sources=[task_name],
        warnings=[],
    )


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
) -> SourceFormatMetadata:
    return SourceFormatMetadata(
        kind=HF_SOURCE_FORMAT_KIND,
        dataset_repo_id=dataset_name,
        config_name=config_name,
        split=split,
        row_id_field=row_id_field,
        task_field=task_field,
        model_field=model_field,
        pass_count_field=pass_count_field,
        attempt_count_field=attempt_count_field,
        pass_rate_field=pass_rate_field,
        instance_id_definition=HF_INSTANCE_ID_DEFINITION,
    )


def build_hf_output_dataset(
    source_dataset: Dataset, rows: list[DifficultyRow]
) -> tuple[list[dict[str, Any]], Dataset]:
    ordered_rows = sorted(rows, key=lambda row: row.source_row_index)
    output_rows = []
    for row in ordered_rows:
        output_row = asdict(row)
        output_row.pop("source_row_index")
        output_rows.append(output_row)

    dataset = source_dataset.select([row.source_row_index for row in ordered_rows])

    for column_name in HF_OUTPUT_COLUMNS:
        values = [make_jsonable(getattr(row, column_name)) for row in ordered_rows]
        if column_name in dataset.column_names:
            dataset = dataset.remove_columns(column_name)
        dataset = dataset.add_column(column_name, values)

    return output_rows, dataset


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
    text = (value if isinstance(value, str) else str(value)).strip()
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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rate = float(value)
        if not math.isfinite(rate):
            raise ValueError(f"expected finite pass-rate value in [0, 1], received {value!r}")
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
    if parsed_pass_rate is not None and not math.isclose(
        pass_count / attempt_count, parsed_pass_rate, rel_tol=EPS, abs_tol=EPS
    ):
        raise ValueError(
            f"pass-count fields {pass_count_field!r}/{attempt_count_field!r} disagree with {pass_rate_field!r}"
        )

    return pass_count, attempt_count


def expand_binary_attempt_scores(*, pass_count: int, attempt_count: int) -> list[float]:
    return [1.0] * pass_count + [0.0] * (attempt_count - pass_count)


def normalize_task_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return normalize_task_name(value[0])
    serialized = json.dumps(make_jsonable(value), ensure_ascii=False, sort_keys=True)
    return serialized or None


def normalize_attempt_scores_for_group(
    rows: list[DifficultyRow], *, allow_nonunit_scores: bool
) -> tuple[list[DifficultyRow], ScoreProcessingMetadata, int]:
    score_processing = infer_score_processing(rows)
    normalized_rows: list[DifficultyRow] = []
    skipped_nonunit = 0

    for row in rows:
        normalized_scores = normalize_attempt_scores(row.attempt_scores, score_processing)
        if normalized_scores is None:
            if allow_nonunit_scores:
                row.attempt_scores = [float(score) for score in row.attempt_scores]
                row.warnings = sorted({*row.warnings, "nonbinary_reward_scores"})
                normalized_rows.append(row)
            else:
                skipped_nonunit += 1
            continue

        row.attempt_scores = normalized_scores
        normalized_rows.append(row)

    return normalized_rows, score_processing, skipped_nonunit


def infer_score_processing(rows: list[DifficultyRow]) -> ScoreProcessingMetadata:
    scores = [float(score) for row in rows for score in row.attempt_scores]
    score_processing = ScoreProcessingMetadata()

    if not scores:
        return score_processing

    if all(
        math.isclose(score, 0.0, rel_tol=EPS, abs_tol=EPS) or math.isclose(score, 1.0, rel_tol=EPS, abs_tol=EPS)
        for score in scores
    ):
        score_processing.normalization = "identity_binary"
        score_processing.positive_reward_value = 1.0
        score_processing.supports_binary_difficulty = True
        return score_processing

    if any(score < -EPS for score in scores):
        return score_processing

    positive_scores = [score for score in scores if score > EPS]
    if not positive_scores:
        score_processing.normalization = "all_zero_binary"
        score_processing.supports_binary_difficulty = True
        return score_processing

    positive_reward_value = max(positive_scores)
    if all(
        math.isclose(score, 0.0, rel_tol=EPS, abs_tol=EPS)
        or math.isclose(score, positive_reward_value, rel_tol=EPS, abs_tol=EPS)
        for score in scores
    ):
        score_processing.normalization = "binary_zero_or_constant"
        score_processing.positive_reward_value = positive_reward_value
        score_processing.supports_binary_difficulty = True

    return score_processing


def normalize_attempt_scores(
    attempt_scores: list[float], score_processing: ScoreProcessingMetadata
) -> list[float] | None:
    if not score_processing.supports_binary_difficulty:
        return None

    normalization = score_processing.normalization
    positive_reward_value = score_processing.positive_reward_value
    normalized_scores: list[float] = []

    for score in attempt_scores:
        if math.isclose(score, 0.0, rel_tol=EPS, abs_tol=EPS):
            normalized_scores.append(0.0)
            continue

        if normalization == "identity_binary" and math.isclose(score, 1.0, rel_tol=EPS, abs_tol=EPS):
            normalized_scores.append(1.0)
            continue

        if (
            normalization == "binary_zero_or_constant"
            and positive_reward_value is not None
            and math.isclose(score, float(positive_reward_value), rel_tol=EPS, abs_tol=EPS)
        ):
            normalized_scores.append(1.0)
            continue

        if normalization == "all_zero_binary":
            return None

        return None

    return normalized_scores


def estimate_beta_prior(rows: list[DifficultyRow], *, prior_mode: str) -> tuple[BetaPrior | None, int]:
    binary_counts = [counts for row in rows if (counts := extract_binary_counts(row.attempt_scores)) is not None]
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
    rows: list[DifficultyRow], *, prior: BetaPrior | None, lower_quantile: float, num_buckets: int
) -> list[DifficultyRow]:
    posterior_rows: list[DifficultyPosteriorRow] = []

    for row in rows:
        row.difficulty = DifficultyPayload()

        if prior is None:
            continue

        binary_counts = extract_binary_counts(row.attempt_scores)
        if binary_counts is None:
            continue

        success_count, attempt_count = binary_counts
        posterior_alpha = success_count + prior.alpha
        posterior_beta = attempt_count - success_count + prior.beta
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_lower_bound = float(beta_distribution.ppf(lower_quantile, posterior_alpha, posterior_beta))

        row.difficulty = DifficultyPayload(
            value=max(0.0, min(1.0, 1.0 - posterior_lower_bound)),
            posterior_mean=posterior_mean,
            posterior_lower_bound=posterior_lower_bound,
        )
        posterior_rows.append(
            DifficultyPosteriorRow(row=row, difficulty_alpha=posterior_beta, difficulty_beta=posterior_alpha)
        )

    assign_posterior_difficulty_buckets(posterior_rows, num_buckets=num_buckets)
    return rows


def assign_posterior_difficulty_buckets(posterior_rows: list[DifficultyPosteriorRow], *, num_buckets: int) -> None:
    if not posterior_rows:
        return

    expected_quantiles = estimate_expected_difficulty_quantiles(posterior_rows)
    for posterior_row, expected_quantile in zip(posterior_rows, expected_quantiles, strict=True):
        posterior_row.row.difficulty.expected_quantile = expected_quantile

    if num_buckets <= 0:
        return

    effective_bucket_count = min(num_buckets, len(posterior_rows))
    ordered_rows = sorted(
        zip(posterior_rows, expected_quantiles, strict=True),
        key=lambda item: (item[1], item[0].row.difficulty.value, item[0].row.instance_id),
    )
    base_bucket_size, remainder = divmod(len(ordered_rows), effective_bucket_count)

    cursor = 0
    for bucket_index in range(effective_bucket_count):
        bucket_size = base_bucket_size + (1 if bucket_index < remainder else 0)
        for posterior_row, _expected_quantile in ordered_rows[cursor : cursor + bucket_size]:
            posterior_row.row.difficulty.bucket_index = bucket_index
            posterior_row.row.difficulty.bucket_count = effective_bucket_count
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


def build_output_paths(
    output_root: Path, *, task_name: str, model_name: str | None, dataset_metadata: DatasetMetadata
) -> tuple[Path, Path, Path]:
    task_suffix = task_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_") or "unknown-task"
    model_suffix = (model_name or "").replace(":", "_").replace("/", "_").replace("\\", "_").replace(
        " ", "_"
    ) or "unknown-model"
    difficulty_suffix = f"__{dataset_metadata.difficulty_generation.tag}"
    stem = output_root / f"{task_suffix}__{model_suffix}{difficulty_suffix}"
    return Path(f"{stem}.jsonl"), Path(f"{stem}.schema.json"), Path(f"{stem}.metadata.json")


def write_output_files(
    *, output_jsonl: Path, schema_json: Path, metadata_json: Path, dataset: Dataset, dataset_metadata: DatasetMetadata
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
        json.dump(make_jsonable(dataset_metadata), output_file, indent=2, sort_keys=True)


def build_dataset_metadata(
    *,
    rows: list[DifficultyRow],
    task_name: str,
    model_name: str | None,
    requested_prior_mode: str,
    requested_bucket_count: int,
    lower_quantile: float,
    prior: BetaPrior | None,
    binary_row_count: int,
    score_processing: ScoreProcessingMetadata,
    source_format: SourceFormatMetadata,
) -> DatasetMetadata:
    effective_bucket_count = extract_effective_bucket_count(rows)
    difficulty_generation = DifficultyGenerationMetadata(
        method=DIFFICULTY_GENERATION_METHOD,
        difficulty_value_field="difficulty.value",
        difficulty_value_definition="1 - difficulty.posterior_lower_bound",
        bucket_field="difficulty.bucket_index",
        bucket_count_field="difficulty.bucket_count",
        bucket_ranking_field="difficulty.expected_quantile",
        posterior_lower_quantile=lower_quantile,
        bucket_count_requested=requested_bucket_count,
        bucket_count_effective=effective_bucket_count,
        beta_prior_requested=requested_prior_mode,
        beta_prior_used=BetaPriorUsageMetadata(
            source=prior.source if prior is not None else None,
            alpha=prior.alpha if prior is not None else None,
            beta=prior.beta if prior is not None else None,
        ),
        binary_instance_count=binary_row_count,
        nonbinary_instance_count=max(0, len(rows) - binary_row_count),
    )
    method_value = difficulty_generation.method
    if method_value:
        method_token = DIFFICULTY_METHOD_FILENAME_ALIASES.get(
            method_value, method_value.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        )
    else:
        method_token = "diff"

    prior_source = difficulty_generation.beta_prior_used.source
    if prior_source:
        prior_token = PRIOR_SOURCE_FILENAME_ALIASES.get(
            prior_source, prior_source.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        )
    else:
        prior_token = "none"

    quantile_token = (
        f"q{f'{difficulty_generation.posterior_lower_quantile * 100.0:.8g}'.replace('-', 'm').replace('.', 'p')}"
    )
    if difficulty_generation.bucket_count_requested == difficulty_generation.bucket_count_effective:
        bucket_token = f"k{difficulty_generation.bucket_count_requested}"
    else:
        bucket_token = (
            f"k{difficulty_generation.bucket_count_requested}e{difficulty_generation.bucket_count_effective}"
        )
    difficulty_generation.tag = "-".join([method_token, prior_token, quantile_token, bucket_token])
    return DatasetMetadata(
        task_name=task_name,
        model_name=model_name,
        row_count=len(rows),
        source_format=source_format,
        score_processing=score_processing,
        difficulty_generation=difficulty_generation,
    )


def extract_effective_bucket_count(rows: list[DifficultyRow]) -> int:
    effective_bucket_counts = {row.difficulty.bucket_count for row in rows if row.difficulty.bucket_count is not None}
    if not effective_bucket_counts:
        return 0
    if len(effective_bucket_counts) != 1:
        raise ValueError(f"Expected a single effective bucket count, found {sorted(effective_bucket_counts)}")
    return next(iter(effective_bucket_counts))


def annotate_dataset_metadata(dataset: Dataset, dataset_metadata: DatasetMetadata) -> None:
    if not hasattr(dataset, "info") or dataset.info is None:
        return
    dataset.info.description = json.dumps(make_jsonable(dataset_metadata), indent=2, sort_keys=True)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.posterior_lower_quantile < 1.0:
        raise ValueError("--posterior-lower-quantile must be between 0 and 1.")
    if args.difficulty_buckets < 0:
        raise ValueError("--difficulty-buckets must be non-negative.")
    if args.max_instances is not None and args.max_instances <= 0:
        raise ValueError("--max-instances must be positive when provided.")


def group_rows_by_task_and_model(rows: list[DifficultyRow]) -> dict[tuple[str, str | None], list[DifficultyRow]]:
    rows_by_group: dict[tuple[str, str | None], list[DifficultyRow]] = defaultdict(list)
    for row in rows:
        task_name = row.task_name
        model_name = row.experiment_metadata.model_name
        rows_by_group[(task_name, model_name)].append(row)
    return dict(rows_by_group)


def get_base_task_name(task_name: str) -> str:
    return task_name.split("@", 1)[0].split(":", 1)[0]


def extract_binary_counts(attempt_scores: list[float]) -> tuple[int, int] | None:
    if not attempt_scores:
        return None

    success_count = 0
    for score in attempt_scores:
        if math.isclose(score, 0.0, rel_tol=EPS, abs_tol=EPS):
            continue
        if math.isclose(score, 1.0, rel_tol=EPS, abs_tol=EPS):
            success_count += 1
            continue
        return None

    return success_count, len(attempt_scores)


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {key: make_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            (key if isinstance(key, str) else "" if key is None else str(key)): make_jsonable(item)
            for key, item in value.items()
        }
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
