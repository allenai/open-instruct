"""Summarize an inference-record store: per-prompt reward evidence and a per-domain token account.

Run with ``python -m open_instruct.miles records summarize STORE --output DIR``. STORE is
the shared records root or one lineage directory beneath it. The command reads only
complete JSON lines and writes:

    prompts.jsonl  one row per (input_key, policy_scope)
    tokens.json    generated tokens by domain and disposition
    summary.json   input snapshot (every file's size and SHA-256), counts and warnings

Reward statistics use only responses whose verifier validity is true. Unknown and
invalid responses are counted, and unknown-validity rewards are summarized
separately, never mixed with valid evidence.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

SUMMARY_SCHEMA_VERSION = 1


def reward_value(reward):
    """Scalar reward, or None for missing or non-numeric rewards."""
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        return None
    return float(reward)


def distribution(values):
    """Count, mean, sample variance and exact-value histogram; fractional rewards stay fractional."""
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else None
    histogram = collections.Counter(f"{value:.6g}" for value in values)
    return {
        "count": len(values),
        "mean": mean,
        "variance": variance,
        "min": min(values),
        "max": max(values),
        "histogram": dict(sorted(histogram.items(), key=lambda item: float(item[0]))),
    }


def group_outcome(row):
    """Classify a group by the rewards of its responses, independent of the filter's decision."""
    rewards = [reward_value(response["reward"]) for response in row["responses"]]
    if any(value is None for value in rewards) or not rewards:
        return "unscored"
    if len(set(rewards)) > 1:
        return "mixed"
    return "all_zero" if rewards[0] == 0 else "constant"


def domain(row):
    return "+".join(sorted(set(row.get("verifiers") or []))) or "unknown"


def _read_rows(path, warnings):
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                # A process that stopped mid-write leaves one partial final line.
                warnings.append(f"{path}: ignored incomplete final line {number}")
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.append(f"{path}: ignored malformed line {number}")
    return rows


def _snapshot(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load(store):
    """Read manifests and rows, attaching each row to its attempt's manifest."""
    store = Path(store)
    warnings = []
    files = []
    attempts = {}
    for path in sorted(store.rglob("manifest-*.json")):
        files.append(_snapshot(path))
        manifest = json.loads(path.read_text())
        attempts[(path.parent, path.stem.removeprefix("manifest-"))] = manifest
    groups, dispositions = [], []
    for path in sorted(store.rglob("records-*.jsonl")):
        files.append(_snapshot(path))
        attempt = attempts.get((path.parent, path.stem.removeprefix("records-")))
        if attempt is None:
            warnings.append(f"{path}: no matching manifest; rows ignored")
            continue
        for row in _read_rows(path, warnings):
            if row.get("schema_version") != 1:
                warnings.append(f"{path}: ignored row with schema_version {row.get('schema_version')!r}")
                continue
            row["_attempt"] = attempt
            (groups if row.get("kind") == "group" else dispositions).append(row)
    return {"files": files, "attempts": attempts, "groups": groups, "dispositions": dispositions, "warnings": warnings}


def summarize(store):
    loaded = load(store)
    groups, warnings = loaded["groups"], loaded["warnings"]
    fate = {}
    for row in loaded["dispositions"]:
        if row["observation_id"] in fate:
            warnings.append(f"observation {row['observation_id']} has more than one disposition")
        fate[row["observation_id"]] = row["disposition"]
    known = {row["observation_id"] for row in groups}
    orphans = sorted(set(fate) - known)
    if orphans:
        warnings.append(f"{len(orphans)} disposition rows have no group row")

    prompts = {}
    tokens = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    for row in groups:
        attempt = row["_attempt"]
        run = attempt["run"]
        unit = f"{run['name']}-{run['id']}:{run['attempt']}"
        if row["filter_decision"] == "passed":
            disposition = fate.get(row["observation_id"], "unused")
        elif row["filter_decision"] == "filtered":
            disposition = f"filtered:{row['filter_reason']}"
        else:
            disposition = row["filter_decision"]
        bucket = tokens[domain(row)][disposition]
        bucket["groups"] += 1
        for response in row["responses"]:
            bucket["responses"] += 1
            bucket["tokens"] += response["response_tokens"] or 0
            if response["truncated"]:
                bucket["truncated_responses"] += 1
                bucket["truncated_tokens"] += response["response_tokens"] or 0
        for scope in sorted({response["policy_scope"] for response in row["responses"]}):
            key = (row["input_key"], scope)
            entry = prompts.setdefault(
                key,
                {
                    "input_key": row["input_key"],
                    "policy_scope": scope,
                    "task_key": row["task_key"],
                    "task_key_basis": row["task_key_basis"],
                    "lineage": attempt["lineage"]["inventory_sha256"],
                    "protocol_sha256": attempt["protocol_sha256"],
                    "domain": domain(row),
                    "source_dataset": row.get("source_dataset"),
                    "source_rows": set(),
                    "prepared_sample_ids": set(),
                    "units": set(),
                    "runs": set(),
                    "groups": 0,
                    "group_outcomes": collections.Counter(),
                    "filter_decisions": collections.Counter(),
                    "validity": collections.Counter(),
                    "valid_rewards": [],
                    "unknown_rewards": [],
                    "truncated": 0,
                },
            )
            entry["units"].add(unit)
            entry["runs"].add(f"{run['name']}-{run['id']}")
            entry["groups"] += 1
            entry["group_outcomes"][group_outcome(row)] += 1
            entry["filter_decisions"][row["filter_decision"]] += 1
            for name in ("source_row", "prepared_sample_id"):
                if row.get(name) is not None:
                    entry["source_rows" if name == "source_row" else "prepared_sample_ids"].add(row[name])
            for response in row["responses"]:
                if response["policy_scope"] != scope:
                    continue
                state = {True: "valid", False: "invalid", None: "unknown"}[response["validity"]["valid"]]
                entry["validity"][state] += 1
                entry["truncated"] += bool(response["truncated"])
                value = reward_value(response["reward"])
                if value is not None and state == "valid":
                    entry["valid_rewards"].append(value)
                elif value is not None and state == "unknown":
                    entry["unknown_rewards"].append(value)

    rows = []
    for key in sorted(prompts):
        entry = prompts[key]
        rows.append(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                **{name: entry[name] for name in ("input_key", "policy_scope", "task_key", "task_key_basis")},
                **{name: entry[name] for name in ("lineage", "protocol_sha256", "domain", "source_dataset")},
                "source_rows": sorted(entry["source_rows"], key=str),
                "prepared_sample_ids": sorted(entry["prepared_sample_ids"], key=str),
                # Observations within one run share a policy trajectory and are correlated;
                # only distinct attempts of start_checkpoint scope are independent draws.
                "independent_units": len(entry["units"]) if key[1] == "start_checkpoint" else len(entry["runs"]),
                "attempts": len(entry["units"]),
                "runs": len(entry["runs"]),
                "groups": entry["groups"],
                "group_outcomes": dict(sorted(entry["group_outcomes"].items())),
                "filter_decisions": dict(sorted(entry["filter_decisions"].items())),
                "validity": dict(sorted(entry["validity"].items())),
                "truncated_responses": entry["truncated"],
                "valid_reward": distribution(entry["valid_rewards"]),
                "unknown_validity_reward": distribution(entry["unknown_rewards"]),
            }
        )
    account = {
        name: {disposition: dict(counts) for disposition, counts in sorted(buckets.items())}
        for name, buckets in sorted(tokens.items())
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "store": str(Path(store)),
        "input_snapshot": loaded["files"],
        "attempts": len(loaded["attempts"]),
        "lineages": sorted({manifest["lineage"]["inventory_sha256"] for manifest in loaded["attempts"].values()}),
        "protocols": sorted({manifest["protocol_sha256"] for manifest in loaded["attempts"].values()}),
        "groups": len(groups),
        "dispositions": dict(collections.Counter(fate.values())),
        "prompt_rows": len(rows),
        "policy_scopes": dict(collections.Counter(row["policy_scope"] for row in rows)),
        "warnings": warnings,
    }
    return rows, account, summary


def write(output, rows, account, summary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prompts.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output / "tokens.json").write_text(json.dumps(account, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m open_instruct.miles records", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize_parser = commands.add_parser("summarize", help="Write per-prompt evidence and a token account")
    summarize_parser.add_argument("store", type=Path, help="Records root or one lineage directory")
    summarize_parser.add_argument("--output", type=Path, required=True, help="Directory for the summary files")
    options = parser.parse_args(argv)
    if not options.store.is_dir():
        parser.error(f"{options.store} is not a directory")
    rows, account, summary = summarize(options.store)
    write(options.output, rows, account, summary)
    for warning in summary["warnings"]:
        logger.warning(warning)
    print(
        json.dumps(
            {name: summary[name] for name in ("groups", "dispositions", "prompt_rows", "policy_scopes", "attempts")}
        )
    )
