"""Download eval predictions via the oe_eval datalake and run length statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import OrderedDict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - dependency check is runtime only
    raise SystemExit(
        "The duckdb package is required for datalake queries. Install it with 'pip install duckdb'."
    ) from exc

from fetch_predictions_by_experiment import (
    dataset_id_from_experiment,
    download_dataset as fetch_dataset_artifact,
    get_experiment_details,
)
from length_analysis import (
    calculate_statistics,
    extract_model_responses,
    load_jsonl_file,
    print_statistics,
    tokenize_responses,
)


DEFAULT_ALIASES: Tuple[str, ...] = (
    "mmlu:cot::hamish_zs_reasoning_deepseek",
    "popqa::hamish_zs_reasoning_deepseek",
    "simpleqa::tulu-thinker_deepseek",
    "bbh:cot::hamish_zs_reasoning_deepseek_v2",
    "gpqa:0shot_cot::hamish_zs_reasoning_deepseek",
    "zebralogic::hamish_zs_reasoning_deepseek",
    "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek",
    "minerva_math::hamish_zs_reasoning_deepseek",
    "minerva_math_500::hamish_zs_reasoning_deepseek",
    "gsm8k::zs_cot_latex_deepseek",
    "omega_500:0-shot-chat_deepseek",
    "aime:zs_cot_r1::pass_at_32_2025_deepseek",
    "aime:zs_cot_r1::pass_at_32_2024_deepseek",
    "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek",
    "mbppplus:0-shot-chat::tulu-thinker_deepseek",
    "livecodebench_codegeneration::tulu-thinker_deepseek",
    "alpaca_eval_v3::hamish_zs_reasoning_deepseek",
    "ifeval::hamish_zs_reasoning_deepseek",
    "bfcl_all::std",
)

SCORE_FIELDS: Tuple[str, ...] = (
    "primary_score",
    "pass_at_1",
    "exact_match_simple_macro",
    "exact_match_flex_macro",
    "exact_match_flex",
)

BIGQUERY_METRICS = "ai2-datascience.oe_eval.metrics"
BIGQUERY_DIM_MODEL_HASH = "ai2-datascience.oe_eval.dim_model_hash"
BIGQUERY_EXPERIMENTS = "ai2-datascience.oe_eval.experiments"
BIGQUERY_ATTACH = "project=ai2-datascience dataset=oe_eval"

StatsDict = Dict[str, Union[int, float]]
RowDict = Dict[str, Any]


def sanitize_name(value: str) -> str:
    """Return a filesystem-friendly name derived from value."""

    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "experiment"


def escape_sql_literal(value: str) -> str:
    """Escape single quotes for safe SQL literal usage."""

    return value.replace("'", "''")


def prepare_duckdb_path(path: Path, *, required: bool = False) -> Optional[Path]:
    try:
        expanded = path.expanduser().resolve(strict=False)
        expanded.parent.mkdir(parents=True, exist_ok=True)
        return expanded
    except OSError as exc:
        if required:
            raise SystemExit(f"Unable to prepare DuckDB path {path}: {exc}") from exc
        print(f"Warning: unable to prepare DuckDB path {path}: {exc}")
        return None


def resolve_duckdb_path(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return prepare_duckdb_path(Path(explicit), required=True)

    for env_var in ("DUCKDB_PATH", "DATABASE_PATH"):
        env_value = os.environ.get(env_var)
        if env_value:
            candidate = prepare_duckdb_path(Path(env_value))
            if candidate:
                return candidate

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = [
        repo_root.parent / "adapt-leaderboard" / "pv_duckdb" / "cache.db",
        repo_root / "pv_duckdb" / "cache.db",
    ]
    for candidate in candidates:
        prepared = prepare_duckdb_path(candidate)
        if prepared:
            return prepared

    return None


def initialize_duckdb_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            eval_sha VARCHAR,
            task_name VARCHAR,
            task_alias VARCHAR,
            task_hash VARCHAR,
            task_idx INTEGER,
            model VARCHAR,
            primary_score DOUBLE,
            run_date TIMESTAMP,
            num_instances INTEGER,
            workspace VARCHAR,
            experiment_id VARCHAR,
            model_hash VARCHAR,
            exact_match_simple_macro DOUBLE,
            exact_match_flex_macro DOUBLE,
            exact_match_flex DOUBLE,
            pass_at_1 DOUBLE
        );
        """
    )

    try:
        conn.execute("ALTER TABLE metrics ADD COLUMN pass_at_1 DOUBLE;")
    except duckdb.CatalogException:
        pass

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_model_hash (
            model_hash VARCHAR PRIMARY KEY,
            metadata JSON
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            tags VARCHAR,
            experiment_id VARCHAR PRIMARY KEY,
            author_name VARCHAR
        );
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_metrics_model_hash ON metrics(model_hash);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id ON metrics(experiment_id);
        """
    )


def fetch_max_run_date(conn: duckdb.DuckDBPyConnection) -> Optional[Any]:
    row = conn.execute("SELECT MAX(run_date) FROM metrics;").fetchone()
    if not row:
        return None
    return row[0]


def format_for_bigquery_timestamp(value: Any) -> str:
    if value is None:
        return "1970-01-01 00:00:00 UTC"

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(" UTC"):
            text = text[:-4] + "+00:00"
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return value
    else:
        return str(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def perform_initial_load(conn: duckdb.DuckDBPyConnection) -> None:
    print("Performing initial DuckDB cache load from BigQuery...")
    metrics_count = conn.execute(
        f"SELECT COUNT(*) FROM bigquery_scan('{BIGQUERY_METRICS}')"
    ).fetchone()[0]
    model_hash_count = conn.execute(
        f"SELECT COUNT(*) FROM bigquery_scan('{BIGQUERY_DIM_MODEL_HASH}')"
    ).fetchone()[0]
    experiments_count = conn.execute(
        f"SELECT COUNT(*) FROM bigquery_scan('{BIGQUERY_EXPERIMENTS}')"
    ).fetchone()[0]

    print(
        "  Initial load counts:\n"
        f"    - Metrics: {metrics_count}\n"
        f"    - Model Hashes: {model_hash_count}\n"
        f"    - Experiments: {experiments_count}"
    )

    conn.execute(
        f"""
        INSERT INTO metrics
        SELECT eval_sha, task_name, task_config.metadata.alias, task_hash, task_idx,
               model_config.model, metrics.primary_score,
               run_date, num_instances, workspace, experiment_id, model_hash,
               metrics.exact_match_simple_macro, metrics.exact_match_flex_macro,
               metrics.exact_match_flex, metrics.pass_at_1
        FROM bigquery_scan('{BIGQUERY_METRICS}');
        """
    )

    conn.execute(
        f"""
        INSERT OR IGNORE INTO dim_model_hash
        SELECT DISTINCT model_hash, model_config.metadata
        FROM bigquery_scan('{BIGQUERY_DIM_MODEL_HASH}');
        """
    )

    conn.execute(
        f"""
        INSERT INTO experiments
        SELECT tags, experiment_id, author_name
        FROM bigquery_scan('{BIGQUERY_EXPERIMENTS}');
        """
    )


def perform_incremental_update(conn: duckdb.DuckDBPyConnection, last_run_date: Any) -> None:
    last_run_str = format_for_bigquery_timestamp(last_run_date)
    print(f"Performing incremental DuckDB update from BigQuery (since {last_run_str})...")

    conn.execute(f"ATTACH '{BIGQUERY_ATTACH}' AS bq (TYPE bigquery, READ_ONLY);")
    conn.execute("SET temp_directory = '/tmp';")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE new_metrics AS
        SELECT new.*
        FROM bigquery_query('bq', $$
            SELECT eval_sha, task_name, task_config.metadata.alias AS task_alias, task_hash, task_idx,
                   model_config.model, metrics.primary_score, run_date, num_instances,
                   workspace, experiment_id, model_hash,
                   metrics.exact_match_simple_macro, metrics.exact_match_flex_macro,
                   metrics.exact_match_flex, metrics.pass_at_1
            FROM `oe_eval.metrics`
            WHERE DATE(run_date) > DATE_SUB(DATE('{last_run_str}'), INTERVAL 3 DAY)
        $$) new
        LEFT JOIN metrics existing
          ON new.eval_sha = existing.eval_sha
        WHERE existing.eval_sha IS NULL;
        """
    )

    new_metrics_count = conn.execute(
        "SELECT COUNT(1) FROM new_metrics;"
    ).fetchone()[0]

    if new_metrics_count == 0:
        print("  No new data to insert.")
        conn.execute("DROP TABLE new_metrics;")
        return

    new_model_hash_count = conn.execute(
        """
        WITH new_model_hashes AS (
            SELECT DISTINCT model_hash
            FROM new_metrics
        )
        SELECT COUNT(1)
        FROM new_model_hashes m
        LEFT JOIN dim_model_hash existing ON m.model_hash = existing.model_hash
        WHERE existing.model_hash IS NULL;
        """
    ).fetchone()[0]

    new_experiments_count = conn.execute(
        """
        WITH new_experiments AS (
            SELECT DISTINCT experiment_id
            FROM new_metrics
        )
        SELECT COUNT(1)
        FROM new_experiments e
        LEFT JOIN experiments existing ON e.experiment_id = existing.experiment_id
        WHERE existing.experiment_id IS NULL;
        """
    ).fetchone()[0]

    print(
        "  Incremental update counts:\n"
        f"    - New Metrics: {new_metrics_count}\n"
        f"    - New Model Hashes: {new_model_hash_count}\n"
        f"    - New Experiments: {new_experiments_count}"
    )

    conn.execute(
        """
        INSERT INTO metrics
        SELECT eval_sha, task_name, task_alias, task_hash, task_idx, model, primary_score,
               run_date, num_instances, workspace, experiment_id, model_hash,
               exact_match_simple_macro, exact_match_flex_macro, exact_match_flex, pass_at_1
        FROM new_metrics;
        """
    )

    conn.execute(
        f"""
        INSERT OR IGNORE INTO dim_model_hash
        SELECT DISTINCT m.model_hash, dmh.model_config.metadata
        FROM new_metrics m
        JOIN bigquery_scan('{BIGQUERY_DIM_MODEL_HASH}') dmh
          ON m.model_hash = dmh.model_hash
        WHERE m.model_hash NOT IN (SELECT model_hash FROM dim_model_hash);
        """
    )

    conn.execute(
        f"""
        INSERT INTO experiments
        SELECT DISTINCT e.tags, e.experiment_id, e.author_name
        FROM new_metrics m
        LEFT JOIN experiments existing
          ON m.experiment_id = existing.experiment_id
        JOIN bigquery_scan('{BIGQUERY_EXPERIMENTS}') e
          ON m.experiment_id = e.experiment_id
        WHERE existing.experiment_id IS NULL;
        """
    )

    conn.execute("DROP TABLE new_metrics;")


def refresh_duckdb_cache(duckdb_path: Path) -> None:
    print(f"Refreshing DuckDB cache at {duckdb_path}...")
    conn = duckdb.connect(str(duckdb_path), config={"allow_unsigned_extensions": "true"})
    try:
        initialize_duckdb_schema(conn)
        conn.execute("BEGIN TRANSACTION;")
        conn.execute("INSTALL bigquery FROM community;")
        conn.execute("LOAD bigquery;")

        max_run_date = fetch_max_run_date(conn)
        if max_run_date is None:
            perform_initial_load(conn)
        else:
            perform_incremental_update(conn, max_run_date)

        conn.execute("COMMIT;")
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def query_local_metrics(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
    aliases: Sequence[str],
) -> List[RowDict]:
    if not aliases:
        return []
    if not table_exists(conn, "metrics"):
        return []

    alias_sql = ", ".join(f"'{escape_sql_literal(alias)}'" for alias in aliases)
    model_literal = escape_sql_literal(model_name)
    query = f"""
        SELECT
            task_alias AS alias,
            run_date,
            primary_score,
            pass_at_1,
            exact_match_simple_macro,
            exact_match_flex_macro,
            exact_match_flex,
            num_instances,
            workspace,
            experiment_id,
            model_hash,
            eval_sha
        FROM (
            SELECT
                task_alias,
                run_date,
                primary_score,
                pass_at_1,
                exact_match_simple_macro,
                exact_match_flex_macro,
                exact_match_flex,
                num_instances,
                workspace,
                experiment_id,
                model_hash,
                eval_sha,
                ROW_NUMBER() OVER (
                    PARTITION BY task_alias
                    ORDER BY run_date DESC
                ) AS row_number
            FROM metrics
            WHERE model = '{model_literal}'
              AND task_alias IN ({alias_sql})
        )
        WHERE row_number = 1;
    """

    cursor = conn.execute(query)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def ensure_bigquery_extension(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("INSTALL bigquery FROM community;")
    conn.execute("LOAD bigquery;")
    try:
        conn.execute(
            f"ATTACH '{BIGQUERY_ATTACH}' AS bq "
            "(TYPE bigquery, READ_ONLY);"
        )
    except duckdb.CatalogException as exc:
        message = exc.args[0] if exc.args else ""
        if "already exists" not in message.lower():
            raise


def query_bigquery_metrics(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
    aliases: Sequence[str],
) -> List[RowDict]:
    if not aliases:
        return []

    ensure_bigquery_extension(conn)
    alias_sql = ", ".join(f"'{escape_sql_literal(alias)}'" for alias in aliases)
    model_literal = escape_sql_literal(model_name)
    query = f"""
        SELECT
            task_alias AS alias,
            run_date,
            primary_score,
            pass_at_1,
            exact_match_simple_macro,
            exact_match_flex_macro,
            exact_match_flex,
            num_instances,
            workspace,
            experiment_id,
            model_hash,
            eval_sha
        FROM bigquery_query('bq', $$
            SELECT
                task_config.metadata.alias AS task_alias,
                run_date,
                metrics.primary_score AS primary_score,
                metrics.pass_at_1 AS pass_at_1,
                metrics.exact_match_simple_macro AS exact_match_simple_macro,
                metrics.exact_match_flex_macro AS exact_match_flex_macro,
                metrics.exact_match_flex AS exact_match_flex,
                num_instances,
                workspace,
                experiment_id,
                model_hash,
                eval_sha
            FROM `oe_eval.metrics`
            WHERE model_config.model = '{model_literal}'
              AND task_config.metadata.alias IN ({alias_sql})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY task_config.metadata.alias
                ORDER BY run_date DESC
            ) = 1
        $$);
    """

    cursor = conn.execute(query)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    query = (
        "SELECT 1 FROM information_schema.tables "
        "WHERE lower(table_name) = ? LIMIT 1"
    )
    row = conn.execute(query, [table_name.lower()]).fetchone()
    return row is not None


def fetch_latest_eval_rows(
    model_name: str,
    aliases: Sequence[str],
    duckdb_path: Optional[Path],
) -> Tuple[Dict[str, RowDict], List[str]]:
    """Return the most recent datalake rows per alias along with lookup sources."""

    # Open DuckDB in read-only mode to allow concurrent access from multiple processes
    if duckdb_path:
        conn = duckdb.connect(str(duckdb_path), read_only=True)
    else:
        conn = duckdb.connect(":memory:")
    
    sources: List[str] = []
    results: Dict[str, RowDict] = {}
    try:
        if duckdb_path:
            local_rows = query_local_metrics(conn, model_name, aliases)
            if local_rows:
                sources.append(f"duckdb:{duckdb_path}")
                results.update({row["alias"]: row for row in local_rows})

        missing = [alias for alias in aliases if alias not in results]
        if missing:
            bigquery_rows = query_bigquery_metrics(conn, model_name, missing)
            if bigquery_rows:
                sources.append("bigquery")
                results.update({row["alias"]: row for row in bigquery_rows})

        return results, sources
    finally:
        conn.close()


def to_python_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, Decimal):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    if hasattr(value, "item"):
        return to_python_number(value.item())
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def to_python_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, Decimal):
        return int(value)
    if hasattr(value, "item"):
        return to_python_int(value.item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def convert_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
        return dt.isoformat()
    return str(value)


def extract_scores(row: Mapping[str, Any]) -> OrderedDict[str, float]:
    scores: OrderedDict[str, float] = OrderedDict()
    for field in SCORE_FIELDS:
        if field not in row:
            continue
        value = to_python_number(row[field])
        if value is not None:
            scores[field] = value
    return scores


def format_scores_for_print(scores: Mapping[str, float]) -> str:
    if not scores:
        return "n/a"
    parts: List[str] = []
    for key, value in scores.items():
        parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


def load_tokenizer(identifier: str):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {identifier}")
    return AutoTokenizer.from_pretrained(identifier)


def find_matching_files(root: Path, pattern: str) -> List[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def normalize_stats(stats: StatsDict) -> StatsDict:
    return {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in stats.items()
    }


def analyze_experiment(
    experiment_label: str,
    files: Iterable[Path],
    tokenizer,
) -> Tuple[StatsDict, List[int]]:
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
        print(f"  Accumulated {len(token_counts)} responses for {experiment_label}")

    if not all_token_counts:
        print(f"  No model responses found for {experiment_label}.")
        return {}, []

    stats = normalize_stats(calculate_statistics(all_token_counts))
    print_statistics(f"Experiment {experiment_label}", stats)
    return stats, all_token_counts


def analyze_alias(
    alias: str,
    record: Mapping[str, Any],
    output_root: Path,
    pattern: str,
    tokenizer,
    overwrite: bool,
) -> Dict[str, Any]:
    experiment_id = record["experiment_id"]
    experiment = get_experiment_details(experiment_id)
    dataset_id = dataset_id_from_experiment(experiment)
    experiment_name = experiment.get("name", experiment_id) or experiment_id

    alias_dir = output_root / sanitize_name(alias)
    alias_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = fetch_dataset_artifact(dataset_id, alias_dir, overwrite)

    matched_files = find_matching_files(dataset_dir, pattern)
    if not matched_files:
        raise RuntimeError(
            f"No files matching '{pattern}' found in downloaded dataset {dataset_id}"
        )

    label = f"{alias} :: {experiment_name} ({experiment_id})"
    stats, token_counts = analyze_experiment(label, matched_files, tokenizer)
    return {
        "experiment_name": experiment_name,
        "dataset_id": dataset_id,
        "dataset_dir": dataset_dir,
        "matched_files": matched_files,
        "stats": stats,
        "token_counts": token_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the latest eval runs for a model from the oe_eval datalake and "
            "report response length statistics."
        ),
    )
    parser.add_argument(
        "model_name",
        help="Exact model name as recorded in the datalake (model_config.model).",
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        default=None,
        help="Specific eval aliases to analyze. Defaults to the olmo3 NEW_POST_TRAIN task list.",
    )
    parser.add_argument(
        "--duckdb-path",
        default=None,
        help="Optional path to a DuckDB cache (as generated by the Adapt leaderboard refresh).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where datasets and statistics will be stored. "
            "Defaults to <current directory>/download_evals_analyze_lengths/data/<sanitized model name>_eval_lengths."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*predictions.jsonl",
        help="Glob pattern used to find prediction files inside each dataset (default: *predictions.jsonl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download datasets even if they already exist in the cache.",
    )
    parser.add_argument(
        "--tokenizer",
        default="allenai/dolma2-tokenizer",
        help="Tokenizer identifier to use for response length computation (default: allenai/dolma2-tokenizer).",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip DuckDB cache refresh and use existing cache as-is (useful for parallel execution).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    aliases: List[str] = []
    seen: set[str] = set()
    for alias in (args.aliases or DEFAULT_ALIASES):
        alias = alias.strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)

    if not aliases:
        raise SystemExit("No evaluation aliases provided.")

    duckdb_path = resolve_duckdb_path(args.duckdb_path)
    if duckdb_path:
        if args.no_refresh:
            print(f"Skipping DuckDB cache refresh (--no-refresh specified).")
            print(f"Using existing DuckDB cache at {duckdb_path} for initial lookup.")
        else:
            try:
                refresh_duckdb_cache(duckdb_path)
            except Exception as exc:
                raise SystemExit(f"Failed to refresh DuckDB cache at {duckdb_path}: {exc}") from exc
            print(f"Using DuckDB cache at {duckdb_path} for initial lookup.")
    else:
        print("No DuckDB cache path detected; queries will run directly against BigQuery.")

    rows_by_alias, sources = fetch_latest_eval_rows(args.model_name, aliases, duckdb_path)
    if not rows_by_alias:
        raise SystemExit(
            "No matching eval runs were found in the datalake for the requested model/aliases."
        )

    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        output_root = (
            Path.cwd() / "download_evals_analyze_lengths" / "data" /
            f"{sanitize_name(args.model_name)}_eval_lengths"
        ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)

    per_alias_summary: Dict[str, Dict[str, Any]] = {}
    aggregate_token_counts: List[int] = []
    missing_aliases: List[str] = []
    errors: Dict[str, str] = {}

    for alias in aliases:
        print(f"\n=== {alias} ===")
        record = rows_by_alias.get(alias)
        if not record:
            print("  No matching eval row found in datalake.")
            missing_aliases.append(alias)
            continue

        run_date_str = convert_datetime(record.get("run_date"))
        scores = extract_scores(record)
        print(f"  Run date: {run_date_str or 'unknown'}")
        print(f"  Experiment: {record.get('experiment_id')} (workspace {record.get('workspace')})")
        print(f"  Scores: {format_scores_for_print(scores)}")

        try:
            analysis = analyze_alias(
                alias=alias,
                record=record,
                output_root=output_root,
                pattern=args.pattern,
                tokenizer=tokenizer,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # pylint: disable=broad-except
            message = str(exc)
            errors[alias] = message
            print(f"  ERROR: {message}")
            continue

        stats = normalize_stats(analysis["stats"])
        token_counts = analysis["token_counts"]
        if token_counts:
            aggregate_token_counts.extend(token_counts)

        per_alias_summary[alias] = {
            "status": "analyzed",
            "run_date": run_date_str,
            "workspace": record.get("workspace"),
            "experiment_id": record.get("experiment_id"),
            "experiment_name": analysis["experiment_name"],
            "dataset_id": analysis["dataset_id"],
            "dataset_path": str(analysis["dataset_dir"]),
            "prediction_files": [str(path) for path in analysis["matched_files"]],
            "scores": scores,
            "num_instances": to_python_int(record.get("num_instances")),
            "model_hash": record.get("model_hash"),
            "eval_sha": record.get("eval_sha"),
            "stats": stats,
        }

    overall_stats = (
        normalize_stats(calculate_statistics(aggregate_token_counts))
        if aggregate_token_counts
        else {}
    )

    if overall_stats:
        print_statistics("All Tasks", overall_stats)
    else:
        print("\nNo model responses found across all analyzed datasets.")

    summary = {
        "model_name": args.model_name,
        "lookup_sources": sources,
        "aliases": per_alias_summary,
        "missing_aliases": missing_aliases,
        "errors": errors,
        "overall": overall_stats,
    }

    stats_path = output_root / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSummary written to {stats_path}")


if __name__ == "__main__":
    main()
