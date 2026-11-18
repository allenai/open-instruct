"""Check judge bindings against prepared inputs before a Beaker GPU allocation.

This scans existing JSONL only: it does not prepare data, load a policy/tokenizer,
call a judge, or import the training runtime. Path mappings let a submitting host
check verified local copies of inputs mounted elsewhere in the training job.
"""

import json
from pathlib import Path

from open_instruct.miles.configuration import run_spec
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import judge_data, judging


def check(spec, mappings=()):
    if not isinstance(spec, run_spec.RunSpec):
        raise InputError("--check-data requires a schema_version=1 GRPO run file")
    replacements = {}
    for entry in mappings:
        original, separator, local = entry.partition("=")
        if not separator or not original or not local:
            raise InputError("--data-path requires CONFIGURED_PATH=LOCAL_PATH")
        original = Path(original).expanduser().resolve()
        if original in replacements:
            raise InputError(f"Duplicate --data-path mapping: {original}")
        replacements[original] = Path(local).expanduser().resolve()

    def local_path(value):
        path = Path(value)
        for original in sorted(replacements, key=lambda p: len(p.parts), reverse=True):
            if path.is_relative_to(original):
                return str(replacements[original] / path.relative_to(original))
        return str(path)

    sources = spec.data.get("sources", [spec.data])
    if any(not source.get("prompt_data") or not source.get("reward_config") for source in sources):
        raise InputError(
            "--check-data requires prepared prompt_data/reward_config inputs (directly or in data.sources). "
            "Prepare task/recipe/manifest inputs before checking their judge bindings."
        )
    training, evaluation, verifiers = [], [], {}
    for source in sources:
        training.append(local_path(source["prompt_data"]))
        evaluation.extend(local_path(path) for path in source.get("eval_prompt_data", [])[1::2])
        path = local_path(source["reward_config"])
        try:
            registry = json.loads(Path(path).read_text())
        except (OSError, UnicodeError, ValueError) as error:
            raise InputError(f"Cannot read reward registry {path}: {error}") from error
        if not isinstance(registry, dict) or any(not isinstance(v, dict) for v in registry.values()):
            raise InputError(f"Reward registry {path} must map verifier names to specifications")
        for name, entry in registry.items():
            if name in verifiers and verifiers[name] != entry:
                raise InputError(f"Conflicting reward registry entry {name!r} in {path}")
            verifiers[name] = entry
    prepared = {"prompt_data": training[0], "train_sources": training + evaluation}
    report = judge_data.validate_data(prepared, judging.registry(spec.judges), verifiers=verifiers)
    return {"status": "passed", "check": "judge_data", **report, "paths": training + evaluation}
