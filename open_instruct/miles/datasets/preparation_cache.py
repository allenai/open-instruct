"""Shared immutable preparation for rendered JSONL sources, independent of run recipes.

Cache entries bind source bytes and tokenizer assets. Run manifests bind sampling,
verifier configuration and pool assignments. Legacy imports are explicit trust in
an existing preparation receipt, verified against its input/output hashes; they
never rewrite that receipt or the imported rows.
"""

import fcntl
import hashlib
import json
import math
import tempfile
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles.datasets import indexed_rows, pools, run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import judging
from open_instruct.miles.rollout import pool_sampling

logger = logger_utils.setup_logger(__name__)
VERSION = 1  # Bump when rendered-row tokenization or admission semantics change.


def validate(options, data):
    if not isinstance(options, dict) or set(options) - {"cache_dir", "import_manifests"}:
        raise InputError("data.preparation accepts cache_dir and import_manifests")
    if not isinstance(options.get("cache_dir"), str) or not options["cache_dir"]:
        raise InputError("data.preparation.cache_dir is required")
    imports = options.get("import_manifests", [])
    if not isinstance(imports, list) or any(not isinstance(p, str) or not p for p in imports):
        raise InputError("data.preparation.import_manifests must be a list of manifest paths")
    if not data.get("sources") or any("prompt_data" not in s for s in data["sources"]):
        raise InputError("Shared preparation currently requires data.sources with rendered prompt_data JSONL")


def tokenizer_identity(hf):
    # Model weights and their location do not determine encoding. Include local
    # custom tokenizer code and config fallback selectors, not training metadata.
    root = Path(hf)
    names = ("tokenizer", "special_tokens", "added_tokens", "chat_template", "vocab", "merges")
    files = {
        str(p.relative_to(root)): run_data._file_sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and (p.name.startswith(names) or p.suffix in {".model", ".tiktoken", ".py"})
    }
    config = root / "config.json"
    fallback = json.loads(config.read_text()) if config.exists() else {}
    return {
        "files": files,
        "fallback": {k: fallback[k] for k in ("model_type", "tokenizer_class", "auto_map") if k in fallback},
    }


def _splits(source):
    evaluation = source.get("eval_prompt_data", [])
    return {"train": source["prompt_data"], **dict(zip(evaluation[::2], evaluation[1::2], strict=True))}


def _identity(source, tokenizer):
    return {
        "version": VERSION,
        "tokenizer": tokenizer,
        "sources": {split: run_data._file_sha(path) for split, path in _splits(source).items()},
    }


def _verify_files(root, files):
    for name, expected in files.items():
        if Path(name).name != name or run_data._file_sha(root / name) != expected:
            raise InputError(f"Prepared cache artifact changed: {root / name}")


def _legacy_imports(paths, tokenizer):
    result = []
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema_version") != 1 or "sources" not in manifest.get("contract", {}).get("data", {}):
            raise InputError(f"Not a legacy prepared-source manifest: {path}")
        contract = manifest["contract"]
        for filename, digest in contract["hf_files"].items():
            if run_data._file_sha(filename) != digest:
                raise InputError(f"Legacy tokenizer/model metadata changed: {filename}")
        if tokenizer_identity(contract["hf_checkpoint"]) != tokenizer:
            raise InputError(f"Legacy preparation tokenizer differs: {path}")
        for filename, digest in manifest["inputs"].items():
            if run_data._file_sha(filename) != digest:
                raise InputError(f"Legacy preparation input changed: {filename}")
        _verify_files(path.parent, manifest["outputs"])
        for source in contract["data"]["sources"]:
            if "prompt_data" not in source:
                continue
            files = {}
            for split in _splits(source):
                filename = (
                    f"source-{source['name']}.jsonl" if split == "train" else f"eval-{source['name']}-{split}.jsonl"
                )
                if filename not in manifest["outputs"]:
                    raise InputError(f"Legacy preparation lacks {filename}")
                files[split] = path.parent / filename
            result.append((_identity(source, tokenizer), files, {"manifest": str(path), "sha256": run_data._sha(raw)}))
    return result


def _metadata_check(row, known, overrides, *, training):
    # Tokenization is deliberately separate from verifier admission. New verifier
    # bindings/implementations recheck metadata without re-encoding the prompt.
    if not isinstance(row.get("input"), str) or not row["input"]:
        raise InputError("Prepared cache row needs nonempty input")
    metadata = row.get("metadata", {})
    verifiers = metadata.get("verifiers") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or not isinstance(verifiers, list) or not verifiers:
        raise InputError("Prepared cache row needs verifiers")
    for verifier in verifiers:
        if not isinstance(verifier, dict):
            raise InputError("Prepared verifier must be an object")
        name = verifier.get("name")
        if name not in known or "target" not in verifier:
            raise InputError(f"Prepared source has unknown/malformed verifier: {name}")
        weight = verifier.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise InputError("Verifier weight must be finite")
        judging.require_binding(name, overrides, known[name])
    if training:
        run_data._verify_if_training_row(row, run_data._if_verifier_names(known))


def _source_entry(cache, source, identity, imports, hf, known, overrides):
    key = run_data._sha(run_data._encoded(identity))
    target = cache / key
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / "manifest.json").exists():
            entry = json.loads((target / "manifest.json").read_text())
            if entry["identity"] != identity:
                raise InputError(f"Prepared cache identity mismatch: {target}")
            _verify_files(target, entry["files"])
            return target, entry, "hit"
        staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache))
        imported = next((x for x in imports if x[0] == identity), None)
        partitions, origin = {}, None
        if imported:
            _, partitions, origin = imported
        else:
            partitions, _, _ = run_data._prepared(source, {})
        tokenizer = None if imported else run_data._tokenizer(hf)
        files, splits = {}, {}
        for split, rows in partitions.items():
            filename = f"{split}.jsonl"
            path = staging / filename
            if imported:
                path.symlink_to(rows)
                iterator = run_data._JsonlRows(rows, {})
            else:
                iterator = rows
            count, maximum, labels = 0, 0, {}
            # Imports retain exact old bytes. Fresh entries retain canonical row
            # labels; source-to-pool assignments are attached by the run manifest.
            output = None if imported else path.open("wb")
            try:
                for row in iterator:
                    _metadata_check(row, known, overrides, training=split == "train")
                    if not imported:
                        run_data._verify_row(row, tokenizer, 2**63 - 1, known)
                        row["metadata"].setdefault("sampling_pool", row.get("pool_id", "__source__"))
                        output.write(run_data._encoded(row))
                    meta = row["metadata"]
                    tokens = meta.get("run_prompt_tokens")
                    token_hash = meta.get("run_prompt_token_ids_sha256")
                    if (
                        type(tokens) is not int
                        or tokens < 1
                        or not isinstance(token_hash, str)
                        or len(token_hash) != 64
                    ):
                        raise InputError("Legacy import requires validated token lengths and token-ID hashes")
                    code = meta.get("sampling_pool", row.get("pool_id", "__source__"))
                    label = row.get("pool_id")
                    if code in labels and labels[code] != label:
                        raise InputError("Prepared pool code maps to inconsistent canonical labels")
                    labels[code] = label
                    count += 1
                    maximum = max(maximum, tokens)
            finally:
                if output is not None:
                    output.close()
            if not count:
                raise InputError(f"Prepared source {split} is empty")
            files[filename] = run_data._file_sha(path)
            splits[split] = {"file": filename, "rows": count, "max_prompt_tokens": maximum, "labels": labels}
        # Recheck source bytes to detect mutation during construction/import.
        if _identity(source, identity["tokenizer"]) != identity:
            raise InputError("Source changed during cache preparation")
        entry = {"identity": identity, "files": files, "splits": splits, "imported_from": origin}
        (staging / "manifest.json").write_bytes(run_data._encoded(entry))
        staging.rename(target)
        return target, entry, "imported" if imported else "built"


def prepare(data, hf, output, *, max_prompt_length, seed, registry_overrides):
    """Publish a small run view referencing shared sources; never alter an old run."""
    options = data["preparation"]
    cache = Path(options["cache_dir"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    overrides = registry_overrides or {}
    tokenizer = tokenizer_identity(hf)
    contract = {
        "data": data,
        "tokenizer": tokenizer,
        "max_prompt_length": max_prompt_length,
        "seed": seed,
        "registry_overrides": overrides,
        "cache_version": VERSION,
    }
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest.get("contract") != contract:
                raise InputError("Preparation contract changed; refusing to alter resumed run data")
            _verify_files(output, manifest["outputs"])
            for path, digest in manifest["inputs"].items():
                if run_data._file_sha(path) != digest:
                    raise InputError(f"Preparation source changed: {path}")
            run_data._verify_resumed_if_training_data(manifest["result"])
            logger.info("Prepared source cache: reused run manifest %s", output)
            return manifest["result"]
        # Avoid re-reading imported bundles on ordinary cache hits.
        identities = [_identity(s, tokenizer) for s in data["sources"]]
        missing = any(not (cache / run_data._sha(run_data._encoded(i)) / "manifest.json").exists() for i in identities)
        imports = _legacy_imports(options.get("import_manifests", []), tokenizer) if missing else []
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        registry, sources, inputs, outputs, evaluation, receipts = {}, [], {}, {}, [], []
        resolved = pools.resolve(data)
        seen = {"train": set(), "eval": set()}
        for source, identity in zip(data["sources"], identities, strict=True):
            rewards = run_data._read_json_object(source["reward_config"], inputs)
            known = rewards | overrides
            for name, spec in known.items():
                if name in registry and registry[name] != spec:
                    raise InputError(f"Conflicting reward registry entry {name!r}")
                registry[name] = spec
            root, entry, status = _source_entry(cache, source, identity, imports, hf, known, overrides)
            receipts.append({"source": source["name"], "artifact": str(root), "status": status})
            for path in _splits(source).values():
                inputs[str(Path(path).resolve())] = run_data._file_sha(path)
            inputs[str(root / "manifest.json")] = run_data._file_sha(root / "manifest.json")
            for split, info in entry["splits"].items():
                index, _ = indexed_rows.open_index(root / info["file"], entry["files"][info["file"]], info["rows"])
                index._mmap.close()
                if info["max_prompt_tokens"] > max_prompt_length:
                    raise InputError(f"Source {source['name']} exceeds prompt length limit; no rows dropped")
                filename = (
                    f"source-{source['name']}.jsonl" if split == "train" else f"eval-{source['name']}-{split}.jsonl"
                )
                (staging / filename).symlink_to(root / info["file"])
                outputs[filename] = entry["files"][info["file"]]
                # Preserve overlap and verifier checks across the complete recipe,
                # but do not tokenize or rewrite rows on cache hits.
                for row in run_data._JsonlRows(root / info["file"], {}):
                    _metadata_check(row, known, overrides, training=split == "train")
                    bucket = seen["train" if split == "train" else "eval"]
                    bucket.add(hashlib.sha256(b"prompt:" + row["input"].encode()).digest())
                    if row["metadata"].get("prepared_sample_id"):
                        bucket.add(
                            hashlib.sha256(b"id:" + str(row["metadata"]["prepared_sample_id"]).encode()).digest()
                        )
                if split != "train":
                    evaluation.extend([f"heldout-{source['name']}-{split}", str(output / filename)])
                    continue
                item = {k: source[k] for k in ("name", "pool", "fraction") if k in source}
                item.update(path=filename, sha256=outputs[filename], rows=info["rows"])
                if resolved:
                    label_names = {v["label"]: k for k, v in resolved.items()}
                    mapping = {}
                    for code, label in info["labels"].items():
                        pool = source.get("pool", label_names.get(label))
                        if pool not in resolved:
                            raise InputError(f"Source {source['name']}: missing or unknown pool label {label!r}")
                        mapping[code] = pool
                    item["pool_mapping"] = mapping
                sources.append(item)
        if tokenizer_identity(hf) != tokenizer or any(
            _identity(source, tokenizer) != identity
            for source, identity in zip(data["sources"], identities, strict=True)
        ):
            raise InputError("Preparation inputs changed during validation")
        if seen["train"] & seen["eval"]:
            raise InputError("Training and held-out data overlap by prompt or source identity")
        recipe = {
            "schema_version": 1,
            "seed": data.get("seed", seed),
            "shuffle": data.get("shuffle", True),
            "sources": sources,
        }
        if resolved:
            recipe.update(pools=resolved, sampling=pool_sampling.validate(data.get("sampling", {}), resolved))
        for filename, value in (("mixture.json", recipe), ("verifiers.json", registry)):
            raw = run_data._encoded(value)
            (staging / filename).write_bytes(raw)
            outputs[filename] = run_data._sha(raw)
        result = {
            "prompt_data": str(output / sources[0]["path"]),
            "eval_prompt_data": evaluation,
            "reward_config": str(output / "verifiers.json"),
            "manifest": str(output / "manifest.json"),
            "prompt_mixture": str(output / "mixture.json"),
            "train_sources": [str(output / s["path"]) for s in sources],
        }
        manifest = {
            "schema_version": 1,
            "contract": contract,
            "inputs": inputs,
            "outputs": outputs,
            "provenance": {"cache": receipts},
            "result": result,
        }
        (staging / "manifest.json").write_bytes(run_data._encoded(manifest))
        staging.rename(output)
        logger.info("Prepared source cache: %s", receipts)
        return result
