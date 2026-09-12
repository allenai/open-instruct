"""Execute the researcher run file without importing the GPU runtime on submission."""

import asyncio
import copy
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from open_instruct.miles.errors import InputError


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def model_identity(source):
    """Record checkpoint metadata and shard identity without hashing tens of GB."""
    source = Path(source)
    if not source.is_dir():
        raise InputError(
            f"Model directory does not exist: {source}. Check model.source and its mount in launch.weka_mounts."
        )
    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "workflow-model.json":
            continue
        stat = path.stat()
        item = dict(path=str(path.relative_to(source)), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if path.suffix in (".json", ".jinja") or path.name == ".metadata":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(item)
    if not files:
        raise InputError(f"Empty model directory: {source}")
    return dict(path=str(source), files=files)


def prepare_model(spec):
    source = Path(spec.model["source"])
    target = Path(spec.conversion["hf_output"])
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise InputError("Prepared model output must not be inside its read-only source")
    template = spec.model.get("hf_template")
    identity = dict(source=model_identity(source), format=spec.model["format"], conversion=spec.conversion)
    if template:
        identity["template"] = (
            model_identity(template)
            if Path(template).is_dir()
            else {"path": template, "sha256": hashlib.sha256(Path(template).read_bytes()).hexdigest()}
        )
    marker = target / "workflow-model.json"
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if recorded["identity"] != identity:
            raise InputError("Prepared model identity changed; choose a new output.root")
        if recorded.get("prepared_files") != model_identity(target)["files"]:
            raise InputError("Prepared model files changed or lack an integrity inventory")
        return str(target)
    if target.exists():
        raise InputError(
            f"Refusing to adopt an incomplete or unrelated prepared model: {target}. Choose a new output.root."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.preparing-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        if spec.model["format"] == "hf":
            if not (source / "config.json").is_file() or not list(source.glob("*.safetensors")):
                raise InputError(f"Expected an HF config and safetensors weights in {source}")
            for path in source.iterdir():
                if path.name == "workflow-model.json":
                    continue
                destination = staging / path.name
                if path.is_file() and path.suffix == ".safetensors":
                    destination.symlink_to(path.resolve())
                elif path.is_dir():
                    shutil.copytree(path, destination, symlinks=False)
                else:
                    # Tokenizer.save_pretrained overwrites these files. Never
                    # allow those writes to follow a link into the source.
                    shutil.copy2(path, destination)
        else:
            _convert_native(spec, staging)
        if template:
            template_path = Path(template)
            if template_path.is_dir():
                transformers = importlib.import_module("transformers")
                tokenizer = transformers.AutoTokenizer.from_pretrained(template_path, trust_remote_code=True)
                # A stale standalone template overrides tokenizer_config in
                # Transformers. Remove only our copied templates before saving.
                (staging / "chat_template.jinja").unlink(missing_ok=True)
                if (staging / "chat_templates").exists():
                    shutil.rmtree(staging / "chat_templates")
                tokenizer.save_pretrained(staging)
            else:
                shutil.copyfile(template_path, staging / "chat_template.jinja")
        if model_identity(source) != identity["source"]:
            raise InputError("Source model changed during preparation")
        write_json(staging / marker.name, {"identity": identity, "prepared_files": model_identity(staging)["files"]})
        staging.rename(target)
    except BaseException:
        # Retain a failed conversion for inspection; never present it as complete.
        raise
    return str(target)


def _convert_native(spec, target):
    converter = importlib.import_module("olmo_core.nn.hf.convert_checkpoint")
    core_config = importlib.import_module("olmo_core.config")
    torch = importlib.import_module("torch")
    saved = converter.load_config(spec.model["source"])
    if not saved or "model" not in saved or "dataset" not in saved:
        raise InputError("Native input must include its saved model and tokenizer configuration")
    converter.convert_checkpoint_to_hf(
        spec.model["source"],
        target,
        copy.deepcopy(saved["model"]),
        copy.deepcopy(saved["dataset"]["tokenizer"]),
        dtype=core_config.DType(spec.conversion.get("dtype", "bfloat16")),
        max_sequence_length=spec.compile().core.max_sequence_length,
        device=torch.device(spec.conversion.get("device", "cpu")),
        validate=False,
    )


def parse_runtime(config):
    try:
        native = importlib.import_module("miles.utils.arguments")
    except ModuleNotFoundError as error:
        if error.name not in {"miles", "miles.utils", "miles.utils.arguments"}:
            raise
        raise InputError(
            "MILES is not installed in this environment. Use plan for local config checks; validate/train require the pinned MILES/Core runtime image."
        ) from error
    original = sys.argv
    try:
        sys.argv = [original[0], *config.arguments()]
        args = native.parse_args()
    finally:
        sys.argv = original
    if args.train_backend != "olmo_core":
        raise InputError(
            "This MILES installation does not select the olmo_core backend; use the pinned MILES/Core runtime image built by scripts/train/build_image_and_launch.sh --miles."
        )
    if args.load:
        checkpoint = importlib.import_module("open_instruct.miles.checkpoint")
        _, manifest = checkpoint.resume_manifest(args.load)
        args.start_rollout_id = manifest["clock"]["next_rollout_id"]
    return args


def train_config(config, *, export_hf=None):
    args = parse_runtime(config)
    os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "olmo_sglang.models")
    driver = importlib.import_module("open_instruct.miles.driver")
    return asyncio.run(driver.train(args, export_hf=export_hf))


@contextmanager
def run_directory(spec):
    root = Path(spec.output["root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".workflow.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InputError(
                f"Another process owns this run: {root}. Wait for it to finish or choose a different output.root."
            ) from error
        document = spec.to_dict()
        identity = fingerprint(document)
        path = root / "workflow.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["spec_sha256"] != identity:
                raise InputError("Run configuration changed; choose a new output.root")
            if previous["status"] == "complete":
                raise InputError(f"Run already completed: {root}. Choose a new output.root for another run.")
            if not spec.launch["auto_resume"]:
                raise InputError(
                    f"Run already exists and launch.auto_resume is false: {root}. "
                    "Set launch.auto_resume=true to resume, or choose a new output.root."
                )
        state = dict(spec_sha256=identity, status="preparing", started_unix=time.time())
        write_json(root / "run-spec.json", document)
        write_json(root / "plan.json", spec.plan())
        write_json(path, state)
        try:
            yield root, state
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_unix=time.time())
            write_json(path, state)
            raise


def execute(spec):
    with run_directory(spec) as (root, state):
        hf = prepare_model(spec)
        data = importlib.import_module("open_instruct.miles.run_data")
        planned = spec.compile()
        prepared = data.prepare_data(
            spec.data,
            Path(hf),
            root / "prepared" / "data",
            max_prompt_length=planned.miles.get("rollout_max_prompt_len", 2048),
            seed=planned.miles.get("seed", 17),
            **(
                {
                    "registry_overrides": {
                        name: {
                            "factory": "open_instruct.miles.judge_registry.NamedJudgeVerifier",
                            "config": {"name": name},
                        }
                        for name in spec.judges["judging"]["bindings"]
                    }
                }
                if spec.judges["judging"]["bindings"]
                else {}
            ),
        )
        if spec.judges["judging"]["bindings"]:
            registry_module = importlib.import_module("open_instruct.miles.judge_registry")
            resolved = registry_module.registry()
            if resolved is None:
                raise InputError("Named judges require the managed launcher (or a resolved registry)")
            registry_module.validate_data(prepared, resolved)
        config = spec.compile({**prepared, "hf_checkpoint": hf})
        checkpoint_root = Path(config.miles["save"])
        if spec.launch["auto_resume"] and (checkpoint_root / "core-latest.json").exists():
            config.miles["load"] = str(checkpoint_root)
        write_json(root / "resolved-plan.json", config.plan())
        state.update(status="training", training_started_unix=time.time())
        write_json(root / "workflow.json", state)
        result = train_config(config, export_hf=spec.output["hf_dir"] if spec.output["export_hf"] else None)
        # A deliberate debug exit is successful execution, not a completed run.
        completed = result.get("completed_rollout_ids") if result else None
        complete = completed is None or (
            (completed[-1] + 1 if completed else result["start_rollout_id"]) == config.miles.get("num_rollout")
        )
        state.update(status="complete" if complete else "stopped", finished_unix=time.time(), result=result)
        write_json(root / "workflow.json", state)
        return state
