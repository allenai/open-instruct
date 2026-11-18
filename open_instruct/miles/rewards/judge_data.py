"""CPU-only judge binding and target validation shared by preflight and training."""

import gzip
import json
from pathlib import Path

from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import abstention, judging


def validate_data(prepared, value, *, verifiers=None):
    paths = [*prepared.get("train_sources", [prepared["prompt_data"]]), *prepared.get("eval_prompt_data", [])[1::2]]
    if verifiers is None:
        verifiers = json.loads(Path(prepared["reward_config"]).read_text()) if prepared.get("reward_config") else {}
    seen = set()
    rows = 0
    for path in paths:
        for row in data_rows(path):
            rows += 1
            metadata = row.get("metadata", {})
            for verifier in metadata.get("verifiers", []):
                name = verifier["name"]
                judging.require_binding(name, value["bindings"], verifiers.get(name))
                if name not in value["bindings"]:
                    continue
                seen.add(name)
                if not isinstance(metadata.get("judge_query"), str) or not metadata["judge_query"]:
                    raise InputError("Judged data requires the preserved full metadata.judge_query")
                profile = value["rubrics"][value["bindings"][name]["rubric"]]["profile"]
                target = verifier.get("target")
                is_abstention = isinstance(target, dict) and target.get("kind") == abstention.KIND
                if (profile == abstention.PROFILE) != is_abstention:
                    raise InputError("Abstention target and rubric must be bound together")
                if is_abstention:
                    abstention.validate_target(target)
                if profile != "open-instruct/general-quality" and verifier.get("target") in (None, ""):
                    raise InputError(f"{name} requires reference targets")
    if unused := set(value["bindings"]) - seen:
        raise InputError(f"Judge bindings absent from train/eval data: {sorted(unused)}")
    return {"rows": rows, "judge_bindings": sorted(seen)}


def data_rows(path):
    """Stream plain or gzipped JSONL with actionable input errors on CPU hosts."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    metadata = row.get("metadata", {})
                    if not isinstance(metadata, dict) or not isinstance(metadata.get("verifiers", []), list):
                        raise ValueError("metadata.verifiers must be a list")
                    if any(
                        not isinstance(v, dict) or not isinstance(v.get("name"), str)
                        for v in metadata.get("verifiers", [])
                    ):
                        raise ValueError("each verifier requires a string name")
                except (ValueError, AttributeError) as error:
                    raise InputError(f"Invalid judge data at {path}:{index}: {error}") from error
                yield row
    except (OSError, UnicodeError, EOFError) as error:
        raise InputError(f"Cannot read judge data {path}: {error}") from error
