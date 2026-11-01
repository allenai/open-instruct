"""Prepare the fixed training and held-out inputs consumed by a MILES run. Named
tasks, existing JSONL datasets or an adopted manifest are turned into rendered
prompts with verifier targets, a reward registry and a provenance manifest. The
preparation checks prompt lengths, reward metadata and overlap by rendered prompt
or sample ID, then records input and output hashes so a resumed run reuses the
same data instead of silently resampling or accepting changed files.
"""

import copy
import hashlib
import importlib
import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path

from open_instruct import toolrl_utils
from open_instruct.miles.configuration import validation
from open_instruct.miles.datasets import mixture, synlogic
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import judging, task_verifiers

# Preserve factory paths in existing prepared datasets and reward registries.
RewardConfig = task_verifiers.RewardConfig
RewardResult = task_verifiers.RewardResult
MultiplicationVerifier = task_verifiers.MultiplicationVerifier
R1FormatVerifier = task_verifiers.R1FormatVerifier
ManifestIFVerifier = task_verifiers.ManifestIFVerifier

TASKS = {
    "gsm8k": ("ai2-adapt-dev/rlvr_gsm8k_zs", "93ffaae6cd2acb8f821f6d4712651320a889b1b9", "gsm8k"),
    # Informally cleaned, not rigorously validated: 39 corrected targets and 93 removed rows.
    # TODO: Finish reviewing and cleaning examples observed to be consistently answered
    # incorrectly, then publish the reviewed dataset under a team-owned namespace and repin here.
    "gsm8k-less-noise": ("techarb/gsm8k-cleaner", "3a4e9e3e600ea2854a6d2d0483721b3cd68580ce", "gsm8k"),
    "gsm8k_original": ("ai2-adapt-dev/rlvr_gsm8k_zs", "93ffaae6cd2acb8f821f6d4712651320a889b1b9", "gsm8k"),
    "math": ("ai2-adapt-dev/rlvr_open_reasoner_math", "2cdc4f9e67b426a693d19f11dcc05f1cb8f44793", "math"),
    "ifeval": ("allenai/RLVR-IFeval", "47c03c73621c4aab2b824b7818681117d662770e", "ifeval_old"),
    "synlogic-sudoku": (synlogic.DATASET, synlogic.REVISION, "sudoku"),
    "multiplication": ("generated-multiplication-v1", None, "multiplication"),
}
FACTORIES = {
    "sudoku": "open_instruct.miles.rewards.sudoku.SudokuVerifier",
    "gsm8k": "open_instruct.miles.rewards.verifiers.GSM8KVerifier",
    "math": "open_instruct.ground_truth_utils.MathVerifier",
    "strict_math": "open_instruct.ground_truth_utils.StrictMathVerifier",
    "ifeval_old": "open_instruct.ground_truth_utils.IFEvalVerifierOld",
    "ifeval": "open_instruct.miles.rewards.task_verifiers.ManifestIFVerifier",
    "multiplication": "open_instruct.miles.rewards.task_verifiers.MultiplicationVerifier",
    "r1_format": "open_instruct.miles.rewards.task_verifiers.R1FormatVerifier",
    "toolrl": "open_instruct.miles.rewards.toolrl.ToolRLVerifier",
    "toolrl_think_prefix": "open_instruct.miles.rewards.toolrl.ToolRLThinkPrefixVerifier",
}
ANSWER_PREFIX = (
    "Solve the following problem step by step. The last line of your response should be the answer to "
    "the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)
ANSWER_SUFFIX = 'Remember to put your answer on its own line after "Answer:"'


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, inputs):
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InputError(f"Cannot read input {path}: {error}. Check the path and the job's WEKA mounts.") from error
    inputs[str(path)] = _sha(raw)
    return raw


def _read_json_object(path, inputs):
    raw = _read(path, inputs)
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise InputError(f"Invalid JSON in {path}: {error}. Supply a UTF-8 JSON object.") from error
    return validation.mapping(value, str(path))


def _required_mapping(value, name, fields):
    value = validation.mapping(value, name)
    missing = set(fields) - value.keys()
    if missing:
        raise InputError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    return value


def _rows(raw, source="Data JSONL"):
    try:
        # Unicode paragraph/line separators are valid inside JSON strings.
        # JSONL record boundaries are LF, optionally preceded by CR.
        lines = raw.decode().split("\n")
    except UnicodeError as error:
        raise InputError(f"{source} must be UTF-8 JSONL.") from error
    values = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise InputError(f"{source}: line {line_number} is invalid JSON; use one JSON object per line.") from error
        if not isinstance(row, dict):
            raise InputError(f"{source}: line {line_number}: Data JSONL must contain objects, not arrays or scalars.")
        values.append(row)
    return values


def validate_data(data):
    validation.mapping(data, "[data]")
    """Reject unsupported preparation contracts before downloading any inputs."""
    unknown = set(data) - {
        "seed",
        "shuffle",
        "tasks",
        "sources",
        "recipe",
        "rl_manifest",
        "prompt_data",
        "eval_prompt_data",
        "reward_config",
    }
    if unknown:
        raise InputError(
            f"Unknown data fields: {sorted(unknown)}; use data.tasks, data.rl_manifest, or prepared data.prompt_data."
        )
    modes = [
        name for name in ("tasks", "sources", "recipe", "rl_manifest", "prompt_data") if data.get(name) is not None
    ]
    if len(modes) != 1:
        raise InputError("Choose exactly one of data.tasks, sources, recipe, rl_manifest, or prompt_data")
    if "prompt_data" not in modes and any(data.get(key) is not None for key in ("eval_prompt_data", "reward_config")):
        raise InputError("eval_prompt_data and reward_config belong to prepared prompt_data mode")
    if "recipe" in modes:
        raise InputError("Named data recipes are not yet ported; use tasks or an immutable rl_manifest")
    if "prompt_data" in modes and not data.get("reward_config"):
        raise InputError("Prepared prompt_data requires a trusted reward_config")
    if "sources" in modes:
        mixture.validate_sources(data["sources"])
        for source in data["sources"]:
            if set(source) & {"sources", "seed", "shuffle"}:
                raise InputError("Mixture sources cannot nest sources or override data.seed/shuffle")
            validate_data(mixture.source_data(source, data))
    if "tasks" in modes:
        tasks = data["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise InputError("data.tasks must be a nonempty list")
        names = []
        for task in tasks:
            if not isinstance(task, dict) or set(task) - {"task", "train_count", "eval_count", "prompt_wrapper"}:
                raise InputError("Task entries require task, train_count/eval_count and optional prompt_wrapper")
            name = task.get("task")
            if not isinstance(name, str) or name not in TASKS:
                raise InputError(f"Unsupported task {name!r}; supported tasks: {sorted(TASKS)}")
            names.append(name)
            if name == "synlogic-sudoku" and task.get("prompt_wrapper", "none") != "none":
                raise InputError(
                    "synlogic-sudoku requires prompt_wrapper=none (its answer format is JSON in answer tags)"
                )
            for key in ("train_count", "eval_count"):
                count = task.get(key)
                if count is not None and (type(count) is not int or count < 1):
                    raise InputError(f"{name}.{key} must be a positive integer")
            if not (task.get("train_count") or task.get("eval_count")):
                raise InputError(f"{name} requires a train_count or eval_count")
            if task.get("prompt_wrapper", "none") not in ("none", "auto", "open_instruct_rlzero_answer"):
                raise InputError(
                    f"data.tasks[{len(names) - 1}].prompt_wrapper: Unsupported prompt_wrapper; use 'none', 'auto', or 'open_instruct_rlzero_answer'."
                )
        if len(set(names)) != len(names) or not any(task.get("train_count") for task in tasks):
            raise InputError("Tasks must be unique and include training data")
    validation.boolean(data.get("shuffle", True), "data.shuffle")
    validation.integer(data.get("seed", 17), "data.seed", minimum=0)


def _tokenizer(hf):
    return importlib.import_module("open_instruct.tokenizer_utils").load_tokenizer(hf, trust_remote_code=True)


def _source_rows(name):
    dataset, revision, _ = TASKS[name]
    loader = importlib.import_module("datasets").load_dataset
    if name == "synlogic-sudoku":
        return synlogic.sudoku_rows(loader(dataset, "hard", revision=revision, split="train"))
    return loader(dataset, revision=revision, split="train")


def _messages(row, *, strip_answer, with_tools=False):
    messages = copy.deepcopy(row.get("messages"))
    if messages is None:
        question = row.get("question", row.get("prompt"))
        if not isinstance(question, str) or not question:
            raise InputError("Source requires messages or a question/prompt")
        messages = [{"role": "user", "content": question}]
    if not isinstance(messages, list) or not messages:
        raise InputError("messages must be nonempty")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role")
            not in (("system", "user", "assistant", "tool") if with_tools else ("system", "user", "assistant"))
            or not isinstance(message.get("content"), str)
        ):
            raise InputError("Messages need a supported role and text content; tool turns require structured tools")
    if strip_answer:
        messages = messages[
            : next((i for i, message in enumerate(messages) if message["role"] == "assistant"), len(messages))
        ]
    if not messages or messages[-1]["role"] not in (("user", "tool") if with_tools else ("user",)):
        raise InputError(
            "Prepared prompt must end in a user turn (or tool turn with structured tools); reference answers are not prompts"
        )
    return messages


def _render(messages, tokenizer, template, tools=None):
    if not isinstance(template, str) or not template:
        raise InputError("Supply an explicit HF or manifest chat template")
    options = {"tools": tools} if tools is not None else {}
    return tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, add_generation_prompt=True, **options
    )


def _verify_row(row, tokenizer, limit, known):
    if not isinstance(row.get("input"), str) or not row["input"]:
        raise InputError("Prepared row needs nonempty rendered input")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("verifiers"), list) or not metadata["verifiers"]:
        raise InputError(
            "Prepared row requires a nonempty metadata.verifiers array with name and target for each verifier."
        )
    for index, spec in enumerate(metadata["verifiers"]):
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("name"), str)
            or spec["name"] not in known
            or "target" not in spec
        ):
            raise InputError(
                f"Unsupported or malformed verifier at metadata.verifiers[{index}]; requires a registered name and a target; "
                f"supported names: {', '.join(sorted(known))}."
            )
        weight = spec.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise InputError("Verifier weight must be finite")
        if spec["name"] in ("toolrl", "toolrl_think_prefix"):
            try:
                toolrl_utils.validate_target(spec["target"])
            except ValueError as error:
                raise InputError(str(error)) from error
            expects_prefix = spec["name"] == "toolrl_think_prefix"
            suffix = re.search(r"<think>\s*$", row["input"])
            if suffix and suffix[0] != "<think>":
                raise InputError(
                    "ToolRL does not support whitespace after a prompt's <think>; use an explicit adapter"
                )
            if row["input"].endswith("<think>") != expects_prefix:
                raise InputError(
                    "ToolRL verifier disagrees with the rendered generation prefix. "
                    "Prepare with --think-prefix-in-prompt exactly when the template ends in <think>."
                )
    ids = tokenizer.encode(row["input"], add_special_tokens=False)
    if not 0 < len(ids) <= limit:
        raise InputError(
            f"Prompt has {len(ids)} tokens, outside 1..{limit}; immutable input cannot be truncated. Increase inference.max_context_length/max_prompt_length or prepare shorter prompts."
        )
    expected = metadata.get("prompt_token_ids")
    if expected is not None and expected != ids:
        raise InputError("Prepared prompt token IDs differ under this tokenizer/template")
    metadata["run_prompt_tokens"] = len(ids)
    metadata["run_prompt_token_ids_sha256"] = _sha(_encoded(ids))


def _adopt(data, tokenizer, template, inputs):
    path = Path(data["rl_manifest"]).resolve()
    manifest = _read_json_object(path, inputs)
    if manifest.get("schema_version") != 1:
        raise InputError("Only baseline rl_manifest schema_version=1 is supported")
    options = _required_mapping(manifest.get("miles"), f"{path}: miles", ("input_key", "metadata_key", "label_key"))
    for key in ("input_key", "metadata_key", "label_key"):
        validation.text(options[key], f"{path}: miles.{key}")
    if options.get("custom_rm_path") != "olmo_miles.rl.rewards.registered_reward":
        raise InputError("Manifest reward function is not the supported baseline registered reward contract")
    if options.get("apply_chat_template") is not True:
        raise InputError("Baseline manifest must declare apply_chat_template=true")
    descriptor = options.get("chat_template")
    if descriptor is not None:
        _required_mapping(descriptor, f"{path}: chat_template", ("path", "sha256"))
        validation.text(descriptor["path"], f"{path}: chat_template.path")
        raw = _read(path.parent / descriptor["path"], inputs)
        if _sha(raw) != descriptor["sha256"]:
            raise InputError("Manifest chat template hash mismatch")
        try:
            template = raw.decode()
        except UnicodeError as error:
            raise InputError(f"{path}: chat template must be UTF-8") from error
    if options.get("tool_format") == "olmo-toolrl-v1":
        probe_tools = [
            {
                "type": "function",
                "function": {
                    "name": "format_probe",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                },
            }
        ]
        probe = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "probe"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "format_probe", "arguments": {"value": "sentinel"}}}],
                },
            ],
            tools=probe_tools,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
        )
        if "<function=format_probe>" not in probe or "<parameter=value>" not in probe or "<tools>" not in probe:
            raise InputError("ToolRL requires an inherited Olmo XML tool chat template")
    partitions = {}
    artifacts = validation.mapping(manifest.get("artifacts"), f"{path}: artifacts")
    for split, artifact in artifacts.items():
        if split not in ("train", "eval"):
            raise InputError(f"Unsupported manifest partition: {split}")
        _required_mapping(artifact, f"{path}: artifacts.{split}", ("path", "sha256", "records"))
        validation.text(artifact["path"], f"{path}: artifacts.{split}.path")
        raw = _read(path.parent / artifact["path"], inputs)
        if _sha(raw) != artifact["sha256"]:
            raise InputError(f"Manifest {split} artifact hash mismatch")
        source = _rows(raw, str(path.parent / artifact["path"]))
        if len(source) != artifact.get("records"):
            raise InputError(f"Manifest {split} record count mismatch")
        partitions[split] = []
        for index, row in enumerate(source, 1):
            context = f"{path}: {split} row {index}"
            _required_mapping(row, context, (options["input_key"], options["metadata_key"], options["label_key"]))
            tools = None
            if "tools_key" in options:
                validation.text(options["tools_key"], f"{path}: miles.tools_key")
                tools = row.get(options["tools_key"])
                if not isinstance(tools, list) or not tools:
                    raise InputError(f"{context}: requires nonempty structured tools")
                for tool in tools:
                    function = tool.get("function") if isinstance(tool, dict) else None
                    if (
                        not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("parameters"), dict)
                    ):
                        raise InputError(f"{context}: invalid function schema")
            messages = _messages(
                {"messages": row[options["input_key"]]}, strip_answer=False, with_tools=tools is not None
            )
            metadata = copy.deepcopy(validation.mapping(row[options["metadata_key"]], f"{context}: metadata"))
            metadata.setdefault("query", messages[-1]["content"])
            partitions[split].append(
                {
                    "input": _render(messages, tokenizer, template, tools),
                    "label": row[options["label_key"]],
                    "metadata": metadata,
                }
            )
    return partitions, {"source_manifest": manifest, "template_sha256": _sha(template.encode())}, None


def _prepared(data, inputs):
    registry = _read_json_object(data["reward_config"], inputs)
    if not isinstance(registry, dict) or not registry:
        raise InputError("Trusted reward registry must be a nonempty mapping")
    for name, spec in registry.items():
        if (
            not isinstance(name, str)
            or not isinstance(spec, dict)
            or not isinstance(spec.get("factory"), str)
            or "." not in spec["factory"]
        ):
            raise InputError("Trusted registry entries require explicit factory paths")
    evaluation = data.get("eval_prompt_data", [])
    if not isinstance(evaluation, list) or len(evaluation) % 2:
        raise InputError("eval_prompt_data must alternate dataset names and paths")
    partitions = {"train": _rows(_read(data["prompt_data"], inputs), data["prompt_data"])}
    for name, path in zip(evaluation[::2], evaluation[1::2], strict=True):
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
            or name == "train"
            or name in partitions
        ):
            raise InputError("Evaluation names must be unique safe identifiers distinct from train")
        partitions[name] = _rows(_read(path, inputs), str(path))
    return partitions, {"already_rendered": True}, registry


def _sample_indices(rng, size, train_count, eval_count):
    """Choose training independently of the requested held-out sample count."""
    training = rng.sample(range(size), train_count)
    if not eval_count:
        return training
    selected = set(training)
    remaining = [index for index in range(size) if index not in selected]
    return training + rng.sample(remaining, eval_count)


def _tasks(data, tokenizer, template, seed):
    partitions = {"train": [], "eval": []}
    provenance = []
    for task in data["tasks"]:
        name = task["task"]
        train_count, eval_count = task.get("train_count") or 0, task.get("eval_count") or 0
        total = train_count + eval_count
        source, revision, verifier = TASKS[name]
        # Preserve the cleaned dataset's seed convention under its former task name, gsm8k.
        # This does not align its selected rows with the original, uncleaned dataset.
        sampling_name = "gsm8k" if name == "gsm8k-less-noise" else name
        rng = random.Random(f"{seed}:{sampling_name}")
        if name == "multiplication":
            if total > 8100:
                raise InputError("Generated multiplication supports at most 8100 unique ordered two-digit pairs")
            pairs = _sample_indices(rng, 8100, train_count, eval_count)
            rows = [
                {
                    "question": f"Compute {10 + p // 90} * {10 + p % 90}. Put the result in <answer> tags.",
                    "ground_truth": str((10 + p // 90) * (10 + p % 90)),
                }
                for p in pairs
            ]
            indices = list(range(total))
        else:
            rows = _source_rows(name)
            if len(rows) < total:
                raise InputError(f"{name}: requested {total} rows but source has {len(rows)}")
            indices = _sample_indices(rng, len(rows), train_count, eval_count)
        for position, index in enumerate(indices):
            raw = rows[index]
            messages = _messages(raw, strip_answer=True)
            target = raw.get("ground_truth")
            if target is None and verifier == "gsm8k":
                target = raw.get("answer", "").split("####")[-1]
            target = (
                json.dumps(target, sort_keys=True) if verifier == "ifeval_old" and isinstance(target, dict) else target
            )
            if verifier in ("gsm8k", "multiplication"):
                target = task_verifiers.scalar_target(target)
            elif verifier == "sudoku":
                # The source adapter checks givens against the supplied valid completion.
                if not isinstance(target, str):
                    raise InputError("Sudoku target must be a JSON grid")
            elif verifier == "math":
                if isinstance(target, list) and len(target) == 1:
                    target = target[0]
                if not isinstance(target, str) or not target.strip():
                    raise InputError("Math target must be a nonempty scalar string")
            else:
                try:
                    parsed_target = json.loads(target) if isinstance(target, str) else None
                except ValueError as error:
                    raise InputError(f"{name} source row {index}: target must be valid JSON with func_name") from error
                if not isinstance(parsed_target, dict) or "func_name" not in parsed_target:
                    raise InputError(f"{name} source row {index}: requires legacy func_name target")
            wrapper = task.get("prompt_wrapper", "none")
            if wrapper == "auto":
                wrapper = "none" if name == "ifeval" else "open_instruct_rlzero_answer"
            if wrapper == "open_instruct_rlzero_answer":
                messages[-1]["content"] = f"{ANSWER_PREFIX}\n\n{messages[-1]['content']}\n\n{ANSWER_SUFFIX}"
            metadata = {
                # "train" is the source dataset split, not this run's partition.
                # Keep source identities equal even if a row is assigned to eval.
                "prepared_sample_id": f"{name}:train:{index}",
                "source_dataset": source,
                "source_revision": revision,
                "source_row": index,
                **({"original_row": raw["original_row"]} if "original_row" in raw else {}),
                **raw.get("source_metadata", {}),
                "query": messages[-1]["content"],
                "prompt_wrapper": wrapper,
                "verifiers": [{"name": verifier, "target": target, "weight": 1.0}],
            }
            if name == "multiplication":
                metadata["verifiers"] = [
                    {"name": "multiplication", "target": target, "weight": 10.0},
                    {"name": "r1_format", "target": "", "weight": 1.0},
                ]
            split = "train" if position < train_count else "eval"
            partitions[split].append(
                {"input": _render(messages, tokenizer, template), "label": target, "metadata": metadata}
            )
        provenance.append(
            {
                "task": name,
                "dataset": source,
                "revision": revision,
                "source_split": "train",
                "train_rows": indices[:train_count],
                "eval_rows": indices[train_count:],
            }
        )
    if data.get("shuffle", True):
        for split, rows in partitions.items():
            random.Random(f"{seed}:{split}").shuffle(rows)
    return partitions, {"sources": provenance, "template_sha256": _sha(template.encode())}, None


def _load(data, tokenizer, template, inputs, seed):
    if data.get("rl_manifest") is not None:
        return _adopt(data, tokenizer, template, inputs)
    if data.get("prompt_data") is not None:
        return _prepared(data, inputs)
    return _tasks(data, tokenizer, template, seed)


def _sources(data, tokenizer, template, inputs, seed):
    partitions, provenance, registry = {}, {}, {}
    training = []
    for source in data["sources"]:
        name = source["name"]
        splits, origin, rewards = _load(mixture.source_data(source, data), tokenizer, template, inputs, seed)
        if not splits.get("train"):
            raise InputError(f"Source {name}: preparation requires nonempty training data")
        if rewards is None:
            names = {v["name"] for rows in splits.values() for row in rows for v in row["metadata"]["verifiers"]}
            rewards = {key: {"factory": FACTORIES[key]} for key in names}
        for key, spec in rewards.items():
            if key in registry and registry[key] != spec:
                raise InputError(
                    f"Source {name}: conflicting reward registry entry {key!r}; use distinct verifier names"
                )
            registry[key] = spec
        for split, rows in splits.items():
            key = f"source-{name}" if split == "train" else f"eval-{name}-{split}"
            if key in partitions:
                raise InputError(f"Mixture partition name collision: {key}; rename sources/evaluation datasets")
            partitions[key] = rows
        training.append(f"source-{name}")
        provenance[name] = origin
    return partitions, provenance, registry, training


def prepare_data(
    data: dict,
    hf_checkpoint: Path,
    output: Path,
    *,
    max_prompt_length: int,
    seed: int,
    registry_overrides: dict | None = None,
) -> dict:
    """Create or verify a completed immutable prepared-data directory.

    Adopted manifests retain source ordering, prompt contents and verifier targets;
    data.seed/shuffle never reshuffle an adopted artifact. Resume validates every
    source and output digest before returning the same paths.
    """
    validate_data(data)
    if type(max_prompt_length) is not int or max_prompt_length < 1:
        raise InputError("max_prompt_length must be positive")
    output, hf_checkpoint = Path(output).resolve(), Path(hf_checkpoint).resolve()
    inputs = {}
    try:
        hf_files = sorted(hf_checkpoint.iterdir())
    except OSError as error:
        raise InputError(
            f"Cannot read tokenizer directory {hf_checkpoint}: {error}. Check the path and mounts."
        ) from error
    for path in hf_files:
        if path.is_file() and (
            path.suffix in (".json", ".jinja", ".model", ".txt", ".tiktoken") or path.name.startswith("tokenizer")
        ):
            _read(path, inputs)
    contract = {
        "data": data,
        "hf_checkpoint": str(hf_checkpoint),
        "hf_files": inputs.copy(),
        "max_prompt_length": max_prompt_length,
        "seed": data.get("seed", seed),
        **({"registry_overrides": registry_overrides} if registry_overrides else {}),
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise InputError("Preparation directory is incomplete; use a fresh output directory")
        manifest = _required_mapping(
            _read_json_object(manifest_path, {}), str(manifest_path), ("contract", "inputs", "outputs", "result")
        )
        if manifest["contract"] != contract:
            raise InputError("Preparation contract changed; refusing to alter resumed run data")
        for path, digest in validation.mapping(manifest["inputs"], f"{manifest_path}: inputs").items():
            if _sha(_read(path, {})) != digest:
                raise InputError(f"Preparation source changed: {path}")
        for filename, digest in validation.mapping(manifest["outputs"], f"{manifest_path}: outputs").items():
            if _sha(_read(output / filename, {})) != digest:
                raise InputError(f"Prepared artifact changed: {filename}")
        return _required_mapping(
            manifest["result"],
            f"{manifest_path}: result",
            ("prompt_data", "eval_prompt_data", "reward_config", "manifest"),
        )
    tokenizer = _tokenizer(hf_checkpoint)
    template = tokenizer.chat_template
    training = ["train"]
    if "sources" in data:
        partitions, provenance, registry, training = _sources(data, tokenizer, template, inputs, contract["seed"])
    else:
        partitions, provenance, registry = _load(data, tokenizer, template, inputs, contract["seed"])
    if registry_overrides:
        registry = (
            {name: {"factory": factory} for name, factory in FACTORIES.items()} if registry is None else registry
        ) | registry_overrides
    known = registry if registry is not None else FACTORIES
    if not all(partitions.get(split) for split in training):
        raise InputError("Preparation requires nonempty training data")
    train_prompts, eval_prompts = set(), set()
    train_ids, eval_ids = set(), set()
    for split, rows in partitions.items():
        for index, row in enumerate(rows, 1):
            try:
                if registry_overrides is not None:
                    metadata = row.get("metadata", {})
                    verifiers = metadata.get("verifiers", []) if isinstance(metadata, dict) else []
                    if isinstance(verifiers, list):
                        for verifier in verifiers:
                            if isinstance(verifier, dict) and isinstance(name := verifier.get("name"), str):
                                judging.require_binding(name, registry_overrides, known.get(name))
                _verify_row(row, tokenizer, max_prompt_length, known)
            except InputError as error:
                raise InputError(f"{split} row {index}: {error}") from error
            (train_prompts if split in training else eval_prompts).add(row["input"])
            identity = row["metadata"].get("prepared_sample_id")
            if identity:
                (train_ids if split in training else eval_ids).add(identity)
    if train_prompts & eval_prompts or train_ids & eval_ids:
        raise InputError("Training and held-out data overlap by prompt or source identity")
    if registry is None:
        names = {spec["name"] for rows in partitions.values() for row in rows for spec in row["metadata"]["verifiers"]}
        registry = {name: {"factory": FACTORIES[name]} for name in sorted(names)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        outputs = {}
        evaluation = []
        for split, rows in partitions.items():
            if not rows:
                continue
            raw = b"".join(_encoded(row) for row in rows)
            filename = f"{split}.jsonl"
            (staging / filename).write_bytes(raw)
            outputs[filename] = _sha(raw)
            if split not in training:
                evaluation.extend(["heldout" if split == "eval" else split, str(output / filename)])
        raw = _encoded(registry)
        (staging / "verifiers.json").write_bytes(raw)
        outputs["verifiers.json"] = _sha(raw)
        result = {
            "prompt_data": str(output / f"{training[0]}.jsonl"),
            "eval_prompt_data": evaluation,
            "reward_config": str(output / "verifiers.json"),
            "manifest": str(manifest_path),
        }
        if "sources" in data:
            sources = [
                {
                    "name": source["name"],
                    "fraction": source["fraction"],
                    "path": f"{split}.jsonl",
                    "sha256": outputs[f"{split}.jsonl"],
                    "rows": len(partitions[split]),
                }
                for source, split in zip(data["sources"], training, strict=True)
            ]
            raw = _encoded(
                {
                    "schema_version": 1,
                    "seed": contract["seed"],
                    "shuffle": data.get("shuffle", True),
                    "sources": sources,
                }
            )
            (staging / "mixture.json").write_bytes(raw)
            outputs["mixture.json"] = _sha(raw)
            result["prompt_mixture"] = str(output / "mixture.json")
            result["train_sources"] = [str(output / f"{split}.jsonl") for split in training]
        manifest = {
            "schema_version": 1,
            "contract": contract,
            "inputs": inputs,
            "outputs": outputs,
            "provenance": provenance,
            "result": result,
        }
        (staging / "manifest.json").write_bytes(_encoded(manifest))
        for path, digest in inputs.items():
            if _sha(_read(path, {})) != digest:
                raise InputError(f"Preparation source changed while preparing: {path}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result
