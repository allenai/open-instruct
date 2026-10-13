"""Compile shared TOML fields against a schema captured from an upstream image.

No Core RunSpec defaults or fork option snapshots apply here. Native names are
validated against the selected backend, including names added upstream later.
"""

import copy
import json
from pathlib import Path

from open_instruct.miles.configuration import options, validation
from open_instruct.miles.errors import InputError

BACKENDS = ("fsdp", "megatron", "torchtitan")
SECTIONS = ("training", "trainer", "inference", "optimizer", "async", "tracking", "miles")
ALIASES = {
    "trainer_num_nodes": "actor_num_nodes",
    "num_gpus": "actor_num_gpus_per_node",
    "rollout_gpus_per_node": "num_gpus_per_node",
    "rollout_tensor_parallel_size": "rollout_num_gpus_per_engine",
    "rollout_expert_parallel_size": "sglang_ep_size",
    "num_rollouts": "num_rollout",
    "samples_per_prompt": "n_samples_per_prompt",
    "max_response_length": "rollout_max_response_len",
    "eval_max_response_length": "eval_max_response_len",
    "learning_rate": "lr",
    "comparison_id": "wandb_group",
    "dynamic_batching": "use_dynamic_batch_size",
}
PATH_OPTIONS = {
    "hf_checkpoint",
    "load",
    "ref_load",
    "save",
    "prompt_data",
    "wandb_dir",
    "save_debug_rollout_data",
    "load_debug_rollout_data",
}
# A second config file could override checked TOML values inside the native parser.
FORBIDDEN_NATIVE = {"config", "yaml_cfg", "olmo_core_config"}


def table(document, name, allowed=None):
    value = document.get(name, {})
    validation.mapping(value, f"[{name}]")
    if allowed is not None:
        validation.fields(value, f"[{name}]", allowed)
    return value


def read(path, overrides=()):
    path = Path(path).expanduser().resolve()
    document = validation.read_document(path)
    for override in overrides:
        key, separator, raw = override.partition("=")
        parts = key.split(".")
        if not separator or len(parts) < 2 or any(not part.isidentifier() for part in parts):
            raise InputError("Overrides must be SECTION.KEY=TOML_VALUE")
        current = document
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise InputError(f"Override {key} traverses a non-table")
        current[parts[-1]] = validation.override_value(key, raw)
    return document, path


def backend(document):
    trainer = table(document, "trainer")
    value = trainer.get("backend")
    if value not in BACKENDS:
        raise InputError(f"trainer.backend must explicitly select one of {BACKENDS}; olmo_core uses --miles")
    return value


def compile_config(document, config_path, schema_path):
    document = copy.deepcopy(document)
    validation.fields(
        document, "upstream run", {"schema_version", "name", "model", "data", "output", "launch", *SECTIONS}
    )
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise InputError("schema_version must be 1")
    selected = backend(document)
    schema = json.loads(Path(schema_path).read_text())
    if (
        schema.get("backend") != selected
        or schema.get("source", {}).get("repository") != "https://github.com/radixark/miles.git"
    ):
        raise InputError(f"Expected an upstream {selected} schema exported from the selected image")
    base = Path(config_path).resolve().parent
    native, origins = {}, {}

    def path(value, origin):
        resolved = Path(validation.text(value, origin)).expanduser()
        return str(resolved if resolved.is_absolute() else base / resolved)

    def put(key, value, origin):
        if key in FORBIDDEN_NATIVE:
            raise InputError(f"{origin} cannot be used in upstream TOML; put native settings in [miles]")
        try:
            record, value = options.resolve_option(key, value, schema_path=schema_path)
            key = record["dest"]
            if key in FORBIDDEN_NATIVE:
                raise InputError(f"{key} cannot override the TOML configuration")
            if key in PATH_OPTIONS:
                value = path(value, origin)
            if key == "eval_prompt_data":
                if not isinstance(value, list) or not value or len(value) % 2:
                    raise InputError("eval_prompt_data must alternate dataset names and paths")
                value = [path(v, origin) if i % 2 else validation.text(v, origin) for i, v in enumerate(value)]
            options.encode_options({key: value}, schema_path=schema_path)
        except InputError as error:
            raise InputError(f"{origin} is unavailable or invalid for upstream {selected}: {error}") from error
        if key in native and (type(native[key]) is not type(value) or native[key] != value):
            raise InputError(f"Conflicting settings for {key}: {origins[key]} and {origin}")
        native[key], origins[key] = value, origin

    put("train_backend", selected, "trainer.backend")
    model = table(document, "model", {"source", "format"})
    if model.get("format", "hf") != "hf":
        raise InputError("model.format must be hf; Megatron converted weights belong in miles.load/ref_load")
    put("hf_checkpoint", model.get("source"), "model.source")
    output = table(document, "output", {"root"})
    if "root" in output:
        put("save", str(Path(path(output["root"], "output.root")) / "checkpoints"), "output.root")
    for key, value in table(document, "data").items():
        # Dataset preparation is application behavior, not an upstream argument.
        if key in {"tasks", "train", "eval", "seed", "shuffle", "train_count", "eval_count"}:
            raise InputError(f"data.{key} needs the Core preparation workflow; upstream expects prepared prompt_data")
        put(key, value, f"data.{key}")
    for section in SECTIONS:
        for key, value in table(document, section).items():
            origin = f"{section}.{key}"
            if section == "trainer" and key == "backend":
                continue
            if section != "miles":
                if key == "gpus" and section in ("trainer", "inference"):
                    key = "actor_num_gpus_per_node" if section == "trainer" else "rollout_num_gpus"
                elif key == "placement_mode" and section == "inference":
                    validation.choice(value, origin, ("colocated", "disaggregated"))
                    key, value = "colocate", value == "colocated"
                elif key == "radix_cache":
                    key, value = "sglang_disable_radix_cache", not validation.boolean(value, origin)
                elif key == "activation_recompute":
                    validation.boolean(value, origin)
                    if selected == "megatron":
                        raise InputError(f"{origin}: use native recompute_granularity/method/num_layers for Megatron")
                    key = "gradient_checkpointing"
                else:
                    key = ALIASES.get(key, key)
            put(key, value, origin)
    # Explicit defaults make allocation accounting independent of upstream defaults.
    for key, default in {"actor_num_nodes": 1, "actor_num_gpus_per_node": 1, "num_gpus_per_node": 1}.items():
        if key not in native:
            put(key, default, "upstream adapter default")
    return {
        "name": validation.text(document.get("name"), "name"),
        "backend": selected,
        "entrypoint": "train_async.py" if native.get("fully_async", False) else "train.py",
        "argv": options.encode_options(native, schema_path=schema_path),
        "native": native,
        "launch": copy.deepcopy(table(document, "launch")),
        "source": schema["source"],
        "build_id": schema["build_id"],
    }
