"""Check judge bindings against prepared inputs before a Beaker GPU allocation.

This scans existing JSONL only: it does not prepare data, load a policy/tokenizer,
call a judge, or import the training runtime. Path mappings let a submitting host
check verified local copies of inputs mounted elsewhere in the training job.
"""

import json
from pathlib import Path

from open_instruct.miles.configuration import run_spec
from open_instruct.miles.datasets import reward_registry
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import judge_data, judging


def check(spec, mappings=()):
    if not isinstance(spec, run_spec.RunSpec):
        raise InputError("--check-data requires a schema_version=1 GRPO run file")
    # Fail on configuration/topology errors before reading multi-GB inputs.
    spec.plan()
    sources = spec.data.get("sources", [spec.data])
    if any(not source.get("prompt_data") or not source.get("reward_config") for source in sources):
        raise InputError(
            "--check-data requires prepared prompt_data/reward_config inputs (directly or in data.sources). "
            "Prepare task/recipe/manifest inputs before checking their judge bindings."
        )
    configured = {
        Path(path).expanduser().resolve()
        for source in sources
        for path in [source["prompt_data"], source["reward_config"], *source.get("eval_prompt_data", [])[1::2]]
    }
    replacements = {}
    for entry in mappings:
        original, separator, local = entry.rpartition("=")
        if not separator or not original or not local:
            raise InputError("--data-path requires CONFIGURED_PATH=LOCAL_PATH")
        # Resolve ambiguous '=' delimiters against the configured input paths;
        # both the configured and local filenames may contain '='.
        candidates = []
        for index, character in enumerate(entry):
            if character != "=" or index == 0 or index == len(entry) - 1:
                continue
            prefix = Path(entry[:index]).expanduser().resolve()
            if any(path.is_relative_to(prefix) for path in configured):
                candidates.append((prefix, Path(entry[index + 1 :]).expanduser().resolve()))
        if not candidates:
            raise InputError(f"--data-path mapping {entry!r} does not match any configured input path")
        if len(candidates) != 1:
            raise InputError(f"Ambiguous --data-path mapping {entry!r}; map a more specific path")
        original, local = candidates[0]
        if original in replacements:
            raise InputError(f"Duplicate --data-path mapping: {original}")
        replacements[original] = local

    def local_path(value):
        path = Path(value).expanduser().resolve()
        for original in sorted(replacements, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(original):
                return str(replacements[original] / path.relative_to(original))
        return str(path)

    training, evaluation, verifiers = [], [], {}
    for source in sources:
        training.append(local_path(source["prompt_data"]))
        evaluation.extend(local_path(path) for path in source.get("eval_prompt_data", [])[1::2])
        path = local_path(source["reward_config"])
        try:
            registry = json.loads(Path(path).read_text())
        except (OSError, UnicodeError, ValueError) as error:
            raise InputError(f"Cannot read reward registry {path}: {error}") from error
        try:
            reward_registry.validate(registry)
        except InputError as error:
            raise InputError(f"Reward registry {path}: {error}") from error
        for name, entry in registry.items():
            if name in verifiers and verifiers[name] != entry:
                raise InputError(f"Conflicting reward registry entry {name!r} in {path}")
            verifiers[name] = entry
    prepared = {"prompt_data": training[0], "train_sources": training + evaluation}
    report = judge_data.validate_data(prepared, judging.registry(spec.judges), verifiers=verifiers)
    return {"status": "passed", "check": "judge_data", **report, "paths": training + evaluation}
