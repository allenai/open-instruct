"""Immutable researcher-run data preparation, independent of sibling repositories."""

import ast
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError

TASKS = {
    "gsm8k": ("ai2-adapt-dev/rlvr_gsm8k_zs", "93ffaae6cd2acb8f821f6d4712651320a889b1b9", "gsm8k"),
    "math": ("ai2-adapt-dev/rlvr_open_reasoner_math", "2cdc4f9e67b426a693d19f11dcc05f1cb8f44793", "math"),
    "ifeval": ("allenai/RLVR-IFeval", "47c03c73621c4aab2b824b7818681117d662770e", "ifeval_old"),
    "multiplication": ("generated-multiplication-v1", None, "multiplication"),
}
FACTORIES = {
    "gsm8k": "open_instruct.ground_truth_utils.GSM8KVerifier",
    "math": "open_instruct.ground_truth_utils.MathVerifier",
    "strict_math": "open_instruct.ground_truth_utils.StrictMathVerifier",
    "ifeval_old": "open_instruct.ground_truth_utils.IFEvalVerifierOld",
    "ifeval": "open_instruct.miles.run_data.ManifestIFVerifier",
    "multiplication": "open_instruct.miles.run_data.MultiplicationVerifier",
    "r1_format": "open_instruct.miles.run_data.R1FormatVerifier",
}
ANSWER_PREFIX = (
    "Solve the following problem step by step. The last line of your response should be the answer to "
    "the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
)
ANSWER_SUFFIX = 'Remember to put your answer on its own line after "Answer:"'


@dataclasses.dataclass
class RewardConfig:
    pass


@dataclasses.dataclass
class RewardResult:
    score: float
    cost: float = 0.0


class MultiplicationVerifier:
    """Preserve the baseline's answer-tag numeric scorer."""

    def __init__(self, verifier_config=None):
        pass

    @classmethod
    def get_config_class(cls):
        return RewardConfig

    async def async_call(self, tokens, prediction, label, **kwargs):
        try:
            answer = prediction[prediction.find("<answer>") + len("<answer>") : prediction.find("</answer>")]
            score = float(float(answer.replace(",", "").strip()) == float(_scalar(label)))
        except (TypeError, ValueError):
            score = 0.0
        return RewardResult(score)


class R1FormatVerifier(MultiplicationVerifier):
    async def async_call(self, tokens, prediction, label, **kwargs):
        return RewardResult(float(re.match(r".*?</think>\s*<answer>.*?</answer>", prediction, re.DOTALL) is not None))


class ManifestIFVerifier(MultiplicationVerifier):
    """Translate canonical baseline constraint targets to OI's legacy list wrapper."""

    async def async_call(self, tokens, prediction, label, **kwargs):
        target = label
        if isinstance(target, str):
            try:
                target = json.loads(target)
            except ValueError:
                target = ast.literal_eval(target)
        if isinstance(target, list):
            if len(target) != 1:
                raise InputError("IF target requires exactly one constraint bundle")
            target = target[0]
            if isinstance(target, str):
                target = json.loads(target)
        if not isinstance(target, dict) or "instruction_id" not in target or "kwargs" not in target:
            raise InputError("IF target requires instruction_id and kwargs")
        factory = importlib.import_module("open_instruct.ground_truth_utils").IFEvalVerifier
        return await factory().async_call(tokens, prediction, repr([target]), **kwargs)


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


def _scalar(target):
    if isinstance(target, list):
        if len(target) != 1:
            raise InputError("Expected a singleton answer")
        target = target[0]
    if target is None or not str(target).strip() or isinstance(target, dict):
        raise InputError("Expected a nonempty scalar answer")
    normalized = str(target).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        raise InputError("Scalar answer must not be a stringified list")
    return normalized.replace(",", "")


def validate_data(data):
    validation.mapping(data, "[data]")
    """Reject unsupported preparation contracts before downloading any inputs."""
    unknown = set(data) - {
        "seed",
        "shuffle",
        "tasks",
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
    modes = [name for name in ("tasks", "recipe", "rl_manifest", "prompt_data") if data.get(name) is not None]
    if len(modes) != 1:
        raise InputError("Choose exactly one of data.tasks, recipe, rl_manifest, or prompt_data")
    if "prompt_data" not in modes and any(data.get(key) is not None for key in ("eval_prompt_data", "reward_config")):
        raise InputError("eval_prompt_data and reward_config belong to prepared prompt_data mode")
    if "recipe" in modes:
        raise InputError("Named data recipes are not yet ported; use tasks or an immutable rl_manifest")
    if "prompt_data" in modes and not data.get("reward_config"):
        raise InputError("Prepared prompt_data requires a trusted reward_config")
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
    return importlib.import_module("transformers").AutoTokenizer.from_pretrained(hf, trust_remote_code=True)


def _source_rows(name):
    dataset, revision, _ = TASKS[name]
    return importlib.import_module("datasets").load_dataset(dataset, revision=revision, split="train")


def _messages(row, *, strip_answer):
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
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
        ):
            raise InputError("Only text system/user/assistant messages are supported")
    if strip_answer:
        messages = messages[
            : next((i for i, message in enumerate(messages) if message["role"] == "assistant"), len(messages))
        ]
    if not messages or messages[-1]["role"] != "user":
        raise InputError("Prepared prompt must end in a user turn; reference answers are not prompts")
    return messages


def _render(messages, tokenizer, template):
    if not isinstance(template, str) or not template:
        raise InputError("Supply an explicit HF or manifest chat template")
    return tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, add_generation_prompt=True)


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
    options = manifest["miles"]
    if options.get("custom_rm_path") != "olmo_miles.rl.rewards.registered_reward":
        raise InputError("Manifest reward function is not the supported baseline registered reward contract")
    if options.get("apply_chat_template") is not True:
        raise InputError("Baseline manifest must declare apply_chat_template=true")
    descriptor = options.get("chat_template")
    if descriptor is not None:
        raw = _read(path.parent / descriptor["path"], inputs)
        if _sha(raw) != descriptor["sha256"]:
            raise InputError("Manifest chat template hash mismatch")
        template = raw.decode()
    partitions = {}
    for split, artifact in manifest["artifacts"].items():
        if split not in ("train", "eval"):
            raise InputError(f"Unsupported manifest partition: {split}")
        raw = _read(path.parent / artifact["path"], inputs)
        if _sha(raw) != artifact["sha256"]:
            raise InputError(f"Manifest {split} artifact hash mismatch")
        source = _rows(raw, str(path.parent / artifact["path"]))
        if len(source) != artifact.get("records"):
            raise InputError(f"Manifest {split} record count mismatch")
        partitions[split] = []
        for row in source:
            messages = _messages({"messages": row[options["input_key"]]}, strip_answer=False)
            metadata = copy.deepcopy(row[options["metadata_key"]])
            metadata.setdefault("query", messages[-1]["content"])
            partitions[split].append(
                {
                    "input": _render(messages, tokenizer, template),
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


def _tasks(data, tokenizer, template, seed):
    partitions = {"train": [], "eval": []}
    provenance = []
    for task in data["tasks"]:
        name = task["task"]
        train_count, eval_count = task.get("train_count") or 0, task.get("eval_count") or 0
        total = train_count + eval_count
        source, revision, verifier = TASKS[name]
        rng = random.Random(f"{seed}:{name}")
        if name == "multiplication":
            if total > 8100:
                raise InputError("Generated multiplication supports at most 8100 unique ordered two-digit pairs")
            pairs = rng.sample(range(8100), total)
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
            indices = rng.sample(range(len(rows)), total)
        for position, index in enumerate(indices):
            raw = rows[index]
            messages = _messages(raw, strip_answer=True)
            target = raw.get("ground_truth")
            if target is None and name == "gsm8k":
                target = raw.get("answer", "").split("####")[-1]
            target = (
                json.dumps(target, sort_keys=True) if verifier == "ifeval_old" and isinstance(target, dict) else target
            )
            if verifier in ("gsm8k", "multiplication"):
                target = _scalar(target)
            elif verifier == "math":
                if isinstance(target, list) and len(target) == 1:
                    target = target[0]
                if not isinstance(target, str) or not target.strip():
                    raise InputError("Math target must be a nonempty scalar string")
            elif not isinstance(target, str) or "func_name" not in json.loads(target):
                raise InputError("Named ifeval source requires legacy func_name target")
            wrapper = task.get("prompt_wrapper", "none")
            if wrapper == "auto":
                wrapper = "none" if name == "ifeval" else "open_instruct_rlzero_answer"
            if wrapper == "open_instruct_rlzero_answer":
                messages[-1]["content"] = f"{ANSWER_PREFIX}\n\n{messages[-1]['content']}\n\n{ANSWER_SUFFIX}"
            metadata = {
                "prepared_sample_id": f"{name}:train:{index}",
                "source_dataset": source,
                "source_revision": revision,
                "source_row": index,
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
    for path in sorted(hf_checkpoint.iterdir()):
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
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract"] != contract:
            raise InputError("Preparation contract changed; refusing to alter resumed run data")
        for path, digest in manifest["inputs"].items():
            if _sha(Path(path).read_bytes()) != digest:
                raise InputError(f"Preparation source changed: {path}")
        for filename, digest in manifest["outputs"].items():
            if _sha((output / filename).read_bytes()) != digest:
                raise InputError(f"Prepared artifact changed: {filename}")
        return manifest["result"]
    tokenizer = _tokenizer(hf_checkpoint)
    template = tokenizer.chat_template
    if data.get("rl_manifest") is not None:
        partitions, provenance, registry = _adopt(data, tokenizer, template, inputs)
    elif data.get("prompt_data") is not None:
        partitions, provenance, registry = _prepared(data, inputs)
    else:
        partitions, provenance, registry = _tasks(data, tokenizer, template, contract["seed"])
    if registry_overrides:
        registry = (
            {name: {"factory": factory} for name, factory in FACTORIES.items()} if registry is None else registry
        ) | registry_overrides
    known = registry if registry is not None else FACTORIES
    if not partitions.get("train"):
        raise InputError("Preparation requires nonempty training data")
    train_prompts, eval_prompts = set(), set()
    train_ids, eval_ids = set(), set()
    for split, rows in partitions.items():
        for index, row in enumerate(rows, 1):
            try:
                _verify_row(row, tokenizer, max_prompt_length, known)
            except InputError as error:
                raise InputError(f"{split} row {index}: {error}") from error
            (train_prompts if split == "train" else eval_prompts).add(row["input"])
            identity = row["metadata"].get("prepared_sample_id")
            if identity:
                (train_ids if split == "train" else eval_ids).add(identity)
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
            if split != "train":
                evaluation.extend(["heldout" if split == "eval" else split, str(output / filename)])
        raw = _encoded(registry)
        (staging / "verifiers.json").write_bytes(raw)
        outputs["verifiers.json"] = _sha(raw)
        result = {
            "prompt_data": str(output / "train.jsonl"),
            "eval_prompt_data": evaluation,
            "reward_config": str(output / "verifiers.json"),
            "manifest": str(manifest_path),
        }
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
            if _sha(Path(path).read_bytes()) != digest:
                raise InputError(f"Preparation source changed while preparing: {path}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result
