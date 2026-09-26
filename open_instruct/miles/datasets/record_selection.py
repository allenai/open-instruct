"""Build and apply frozen prompt-exclusion tables from inference records.

``python -m open_instruct.miles records select STORE --skip all_zero --output table.json``
reads a record store and writes a table of prompts whose evidence shows a constant
reward. A run pins the table by path and SHA-256 in ``[selection]``; its data
source then skips excluded prompts as they stream in, so the prepared data and its
order are unchanged. The table is frozen: randomized readmission is resolved when
the table is built, and a resumed run applies the same table.

Evidence rules are deliberately conservative:

- Only responses with ``validity.valid = true`` count.
- Only ``start_checkpoint`` observations count unless a scope is explicitly
  widened; later versions are specific to one run's trajectory.
- A prompt needs ``min_observations`` valid responses from ``min_units``
  independent attempts. With deterministic inference, attempts sharing a rollout
  seed are one unit.
- A prompt is excluded only when every valid response has the same reward and
  the one-sided upper confidence bound on the rate of any other reward,
  ``1 - (1 - confidence) ** (1 / n)`` for zero deviations in n draws, is below
  ``max_deviation_rate``.
"""

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles.datasets import inference_records, record_summary
from open_instruct.miles.errors import InputError

logger = logger_utils.setup_logger(__name__)

TABLE_SCHEMA_VERSION = 1
SKIP_MODES = ("all_zero", "all_full", "zero_variance")
SCOPES = ("start_checkpoint", "run_version", "mixed")


def upper_bound_zero_deviations(observations, confidence):
    """One-sided Clopper-Pearson upper bound for zero events in ``observations`` draws."""
    return 1.0 - (1.0 - confidence) ** (1.0 / observations)


def readmitted(seed, key, fraction):
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 2**32 < fraction


def _choose(values, name, explicit):
    if explicit is not None:
        if explicit not in values:
            raise InputError(f"--{name} {explicit} does not occur in the store; found {sorted(values)}")
        return explicit
    if len(values) != 1:
        raise InputError(f"The store holds {len(values)} {name}s {sorted(values)}; choose one with --{name}")
    return next(iter(values))


def evidence(loaded, *, lineage, protocol, scopes):
    """Valid rewards per input key, with the independent units that produced them."""
    table = {}
    for row in loaded["groups"]:
        attempt = row["_attempt"]
        if attempt["lineage"]["inventory_sha256"] != lineage or attempt["protocol_sha256"] != protocol:
            continue
        run = attempt["run"]
        deterministic = bool(attempt["protocol"].get("sglang_enable_deterministic_inference"))
        # Deterministic inference seeds siblings from rollout_seed, so equal seeds repeat draws.
        unit = (
            f"seed:{attempt.get('rollout_seed')}" if deterministic else f"{run['name']}-{run['id']}:{run['attempt']}"
        )
        for response in row["responses"]:
            value = record_summary.reward_value(response["reward"])
            if response["policy_scope"] not in scopes or response["validity"]["valid"] is not True or value is None:
                continue
            entry = table.setdefault(
                row["input_key"],
                {"task_key": row["task_key"], "domain": record_summary.domain(row), "rewards": [], "units": set()},
            )
            entry["rewards"].append(value)
            entry["units"].add(unit)
    return table


def build(
    store,
    *,
    skip,
    lineage=None,
    protocol=None,
    scopes=("start_checkpoint",),
    min_observations=16,
    min_units=2,
    confidence=0.95,
    max_deviation_rate=0.2,
    full_reward=1.0,
    readmit_fraction=0.05,
    seed=0,
):
    if not skip or any(mode not in SKIP_MODES for mode in skip):
        raise InputError(f"--skip needs one or more of {', '.join(SKIP_MODES)}")
    if any(scope not in SCOPES for scope in scopes):
        raise InputError(f"scopes must be among {', '.join(SCOPES)}")
    if min_observations < 1 or min_units < 1 or not 0 < confidence < 1 or not 0 < max_deviation_rate <= 1:
        raise InputError("min_observations and min_units must be positive; confidence and rate in (0, 1)")
    if not 0 <= readmit_fraction < 1:
        raise InputError("readmit_fraction must be in [0, 1)")
    loaded = record_summary.load(store)
    manifests = loaded["attempts"].values()
    lineage = _choose({m["lineage"]["inventory_sha256"] for m in manifests}, "lineage", lineage)
    protocols = {m["protocol_sha256"]: m["protocol"] for m in manifests if m["lineage"]["inventory_sha256"] == lineage}
    protocol = _choose(set(protocols), "protocol", protocol)
    candidates = evidence(loaded, lineage=lineage, protocol=protocol, scopes=set(scopes))
    excluded, kept_by_readmission = [], []
    counts = collections.defaultdict(collections.Counter)
    for key in sorted(candidates):
        entry = candidates[key]
        rewards, units = entry["rewards"], len(entry["units"])
        counts[entry["domain"]]["candidates"] += 1
        values = set(rewards)
        if len(rewards) < min_observations or units < min_units or len(values) != 1:
            continue
        value = rewards[0]
        mode = "all_zero" if value == 0 else "all_full" if value == full_reward else "zero_variance"
        if mode not in skip and "zero_variance" not in skip:
            continue
        bound = upper_bound_zero_deviations(len(rewards), confidence)
        if bound >= max_deviation_rate:
            continue
        item = {
            "input_key": key,
            "task_key": entry["task_key"],
            "domain": entry["domain"],
            "mode": mode,
            "reward": value,
            "observations": len(rewards),
            "units": units,
            "deviation_upper_bound": bound,
        }
        if readmitted(seed, key, readmit_fraction):
            kept_by_readmission.append(item)
            counts[entry["domain"]]["readmitted"] += 1
        else:
            excluded.append(item)
            counts[entry["domain"]]["excluded"] += 1
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "created_unix": time.time(),
        "store": str(Path(store)),
        "input_snapshot": loaded["files"],
        "warnings": loaded["warnings"],
        "lineage": lineage,
        "protocol_sha256": protocol,
        "protocol": protocols[protocol],
        "rule": {
            "skip": sorted(skip),
            "scopes": sorted(scopes),
            "min_observations": min_observations,
            "min_units": min_units,
            "confidence": confidence,
            "max_deviation_rate": max_deviation_rate,
            "full_reward": full_reward,
            "readmit_fraction": readmit_fraction,
            "seed": seed,
            "approximate_policy": sorted(scopes) != ["start_checkpoint"],
        },
        "counts": {name: dict(value) for name, value in sorted(counts.items())},
        "excluded": excluded,
        "readmitted": kept_by_readmission,
    }


def write_table(table, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(table, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class Selection:
    """A verified exclusion table applied to prompts as the data source streams them."""

    def __init__(self, args):
        core = args.olmo_core
        path = Path(core.selection_table)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != core.selection_sha256:
            raise RuntimeError(f"Selection table {path} has SHA-256 {digest}, not the pinned {core.selection_sha256}")
        table = json.loads(data)
        if table.get("schema_version") != TABLE_SCHEMA_VERSION:
            raise RuntimeError(f"Selection table {path} has unsupported schema {table.get('schema_version')!r}")
        lineage = inference_records.lineage_identity(args.hf_checkpoint)["inventory_sha256"]
        if lineage != table["lineage"]:
            raise RuntimeError(
                f"Selection table lineage {table['lineage']} does not match this checkpoint ({lineage})"
            )
        self.protocol_sha256 = inference_records.sha256(inference_records.protocol_identity(args))
        if self.protocol_sha256 != table["protocol_sha256"]:
            raise RuntimeError(
                "Selection table evidence used a different sampling/verifier protocol "
                f"({table['protocol_sha256']}) than this run ({self.protocol_sha256})"
            )
        self.sha256 = digest
        self.excluded = {item["input_key"]: item["domain"] for item in table["excluded"]}
        self.skipped = collections.Counter()
        self.passed = 0
        logger.info("Selection table %s excludes %d prompts", path, len(self.excluded))

    def input_key(self, sample):
        metadata = sample.metadata or {}
        task = inference_records.task_identity(metadata, sample.prompt)["task_key"]
        tokens = metadata.get("run_prompt_token_ids_sha256")
        return inference_records.sha256({"task": task, "tokens": tokens, "protocol": self.protocol_sha256})

    def keep(self, group):
        domain = self.excluded.get(self.input_key(group[0]))
        if domain is None:
            self.passed += 1
            return True
        self.skipped[domain] += 1
        total = sum(self.skipped.values())
        if total % 100 == 1:
            logger.info("Selection skipped %d prompts so far: %s", total, dict(self.skipped))
        return False


def take(pull, count, keep, limit):
    """Pull groups until ``count`` are kept, failing after ``limit`` consecutive skips."""
    kept, skipped_in_row = [], 0
    while len(kept) < count:
        groups = pull(count - len(kept))
        if not groups:
            break
        for group in groups:
            if keep(group):
                kept.append(group)
                skipped_in_row = 0
            else:
                skipped_in_row += 1
                if skipped_in_row > limit:
                    raise RuntimeError(
                        f"Selection skipped {skipped_in_row} consecutive prompts; it excludes the dataset"
                    )
    return kept


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m open_instruct.miles records select", description=__doc__)
    parser.add_argument("store", type=Path, help="Records root or one lineage directory")
    parser.add_argument("--output", type=Path, required=True, help="Exclusion table JSON to write")
    parser.add_argument("--skip", action="append", choices=SKIP_MODES, required=True)
    parser.add_argument("--lineage", help="Starting-checkpoint lineage digest; required if the store holds several")
    parser.add_argument("--protocol", help="Protocol digest; required if the lineage holds several")
    parser.add_argument(
        "--scope", action="append", choices=SCOPES, help="Policy scopes to use; default start_checkpoint only"
    )
    parser.add_argument("--min-observations", type=int, default=16)
    parser.add_argument("--min-units", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-deviation-rate", type=float, default=0.2)
    parser.add_argument("--full-reward", type=float, default=1.0)
    parser.add_argument("--readmit-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args(argv)
    if not options.store.is_dir():
        parser.error(f"{options.store} is not a directory")
    try:
        table = build(
            options.store,
            skip=options.skip,
            lineage=options.lineage,
            protocol=options.protocol,
            scopes=tuple(options.scope or ("start_checkpoint",)),
            min_observations=options.min_observations,
            min_units=options.min_units,
            confidence=options.confidence,
            max_deviation_rate=options.max_deviation_rate,
            full_reward=options.full_reward,
            readmit_fraction=options.readmit_fraction,
            seed=options.seed,
        )
    except InputError as error:
        parser.error(str(error))
    digest = write_table(table, options.output)
    for warning in table["warnings"]:
        logger.warning(warning)
    print(
        json.dumps(
            {
                "table": str(options.output),
                "sha256": digest,
                "excluded": len(table["excluded"]),
                "readmitted": len(table["readmitted"]),
                "counts": table["counts"],
            }
        )
    )
