"""CPU-safe researcher run specifications around the MILES/Core runtime facade.

The section names follow olmo-miles. Preparation, launch and export are explicit
workflow stages; this module only validates and compiles their desired state.
"""

import copy
import dataclasses
import math
import re
from pathlib import Path
from typing import Any

from open_instruct.miles import judging, options, run_data, topology, validation
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.errors import InputError

RUN_SECTIONS = ("training", "trainer", "inference", "optimizer", "async", "tracking", "runtime")
WORKFLOW_SECTIONS = {"model", "conversion", "data", "output", "launch", "compiler_cache"}
CORE_FIELDS = {field.name for field in dataclasses.fields(CoreConfig)}
# Names that change when the trainer is replaced, rather than native switches.
FIELD_MAP = {
    "trainer_num_nodes": "miles.actor_num_nodes",
    "num_gpus": "miles.actor_num_gpus_per_node",
    "rollout_gpus_per_node": "miles.num_gpus_per_node",
    "rollout_tensor_parallel_size": "miles.rollout_num_gpus_per_engine",
    "rollout_expert_parallel_size": "miles.sglang_ep_size",
    "expert_parallel_size": "core.expert_parallel_size",
    "activation_recompute": "core.activation_checkpointing",
    "replay_rollout_data": "miles.load_debug_rollout_data",
    "replay_rollout_data_subsample": "miles.load_debug_rollout_data_subsample",
    "num_rollouts": "miles.num_rollout",
    "collect_dashboard": "miles.use_miles_dashboard",
    "samples_per_prompt": "miles.n_samples_per_prompt",
    "eval_max_response_length": "miles.eval_max_response_len",
    "max_response_length": "miles.rollout_max_response_len",
    "learning_rate": "miles.lr",
    "max_weight_staleness": "core.max_policy_lag",
    "max_train_rollout_logprob_abs_diff": "core.max_train_rollout_logprob_abs_diff",
    "comparison_id": "miles.wandb_group",
    # Serving prefix cache and request routing, named as in olmo-miles.
    "mamba_radix_cache_strategy": "miles.sglang_mamba_radix_cache_strategy",
    "router_policy": "miles.sglang_router_policy",
    "router_cache_threshold": "miles.router_cache_threshold",
    "router_balance_abs_threshold": "miles.router_balance_abs_threshold",
    "router_balance_rel_threshold": "miles.router_balance_rel_threshold",
    "enable_mixed_chunk": "miles.sglang_enable_mixed_chunk",
}
UNSUPPORTED_FIELDS = {
    "megatron_checkpoint": "use model.source/model.format or miles.load for a native Core RL resume",
    "output_dir": "use output.root",
    "hf_checkpoint": "use model.source; preparation supplies miles.hf_checkpoint",
    "trainer_backend": "Core selects its native model backend; omit the Megatron optimized/compatibility switch",
    "recompute_modules": "Core supports block activation_checkpointing, not Megatron selective modules",
    "accumulate_allreduce_grads_in_fp32": "Core owns reduction precision; this Megatron switch has no Core equivalent",
    "save_retain_interval": "native Core checkpoint retention is not implemented",
    "save_tokens_per_expert_interval": "tokens-per-expert checkpoint capture is not implemented",
    "capture_generation_samples": "use save_debug_rollout_data or collect_dashboard; bounded sampling is not implemented",
    "generation_samples_per_rollout": "use save_debug_rollout_data or collect_dashboard; bounded sampling is not implemented",
    "rollout_recovery_max_attempts": "Core does not yet implement the baseline driver retry budget",
    "rollout_recovery_mem_fraction_static": "Core does not implement recovery-time memory overrides",
    "rollout_stage_timeout": "Core does not yet implement the baseline per-stage deadline",
    "rollout_health_diagnostics": "use dedicated recovery probes; the baseline diagnostic wrapper is not installed",
    "rollout_test_fault": "use a dedicated fault-injection qualification, not an ordinary run",
    "inference_ep_diagnostics": "use the separate inference-EP diagnostics",
    "determinism_probe_samples": "use retained-input diagnostic scripts",
    "determinism_probe_forward_trace": "use retained-input diagnostic scripts",
    "determinism_probe_cross_gpu": "use retained-input diagnostic scripts",
    "determinism_probe_retune_kda": "use retained-input diagnostic scripts",
    "determinism_probe_l2norm_inputs": "use retained-input diagnostic scripts",
    "weight_export_mode": "Core exports native HF tensors; use core.stream_moe_export",
    "colocated_live_weight_export": "Core already owns live IPC export; there is no Megatron patch selector",
    "hardware_profile": "choose explicit Core/serving settings and launch.cluster; automatic hardware policy is not implemented",
    "code_service_mode": "provision the verifier service externally and pass its environment",
    "code_service_workers": "provision the verifier service externally",
    "code_service_source_revision": "record externally provisioned service provenance",
    "start_code_service": "per-run code-service provisioning is not implemented",
    "code_service_source_root": "per-run code-service provisioning is not implemented",
    "code_service_python": "per-run code-service provisioning is not implemented",
    "code_service_host": "per-run code-service provisioning is not implemented",
    "code_service_port": "per-run code-service provisioning is not implemented",
    "code_service_log": "per-run code-service provisioning is not implemented",
    "fla_prewarm": "use compiler_cache.enabled; generic FLA prewarming is not implemented",
    "fla_prewarm_sequence_length": "generic FLA prewarming is not implemented",
    "miles_train_script": "this workflow owns the Core driver",
    "python_path": "install code in the pinned runtime image; the launcher owns PYTHONPATH",
    "skip_cuda_check": "plan is CPU-safe; validate checks the installed runtime",
    "validate_miles_args": "use the validate command",
    "no_start_ray": "the launcher owns Ray startup",
    "dataset_profile": "choose data.tasks, data.recipe or data.rl_manifest",
    "rl_manifest": "use data.rl_manifest",
}
PATH_OPTIONS = {
    "load",
    "ref_load",
    "save",
    "save_debug_rollout_data",
    "load_debug_rollout_data",
    "wandb_dir",
    "eval_hf_dir",
}


def _table(document, name, allowed=None, *, required=False):
    value = document.get(name, {})
    if not isinstance(value, dict) or (required and not value):
        raise InputError(f"[{name}] must be a {'nonempty ' if required else ''}table")
    validation.mapping(value, f"[{name}]")
    if allowed is not None:
        validation.fields(value, f"[{name}]", allowed)
    return copy.deepcopy(value)


def _text(value, name):
    return validation.text(value, name)


def _boolean(value, name):
    return validation.boolean(value, name)


def _positive(value, name):
    return validation.integer(value, name)


def _path(value, base, name):
    path = Path(_text(value, name)).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _eval_paths(value, base):
    if not isinstance(value, list) or len(value) % 2:
        raise InputError("data.eval_prompt_data must alternate dataset names and JSONL paths")
    return [
        _path(item, base, "data.eval_prompt_data") if index % 2 else _text(item, "eval dataset name")
        for index, item in enumerate(value)
    ]


def _apply_overrides(document, overrides):
    for override in overrides or []:
        key, separator, raw = validation.text(override, "Override").partition("=")
        parts = key.split(".")
        if not separator or not all(re.fullmatch(r"[a-z][a-z0-9_]*", part) for part in parts):
            raise InputError("Overrides must be SECTION.KEY=TOML_VALUE")
        value = validation.override_value(key, raw)
        current = document
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise InputError(f"Override {key} traverses a non-table value")
        current[parts[-1]] = value


@dataclasses.dataclass(frozen=True)
class RunSpec:
    name: str
    config_path: Path
    model: dict[str, Any]
    data: dict[str, Any]
    output: dict[str, Any]
    launch: dict[str, Any]
    conversion: dict[str, Any]
    compiler_cache: dict[str, Any]
    sections: dict[str, dict[str, Any]]
    core: dict[str, Any]
    miles: dict[str, Any]
    judges: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "RunSpec":
        path = Path(path).expanduser().resolve()
        document = validation.read_document(path)
        return cls.from_dict(document, config_path=path, overrides=overrides)

    @classmethod
    def from_dict(
        cls, payload: dict, config_path: str | Path = Path("run.toml"), overrides: list[str] | None = None
    ) -> "RunSpec":
        """Load a serialized specification; relative paths use the original config directory."""
        if not isinstance(payload, dict):
            raise InputError("Run specification must be a mapping")
        document = copy.deepcopy(payload)
        path = Path(config_path).expanduser().resolve()
        _apply_overrides(document, overrides)
        allowed = {
            "schema_version",
            "name",
            "core",
            "miles",
            *WORKFLOW_SECTIONS,
            *RUN_SECTIONS,
            "judges",
            "rubrics",
            "judging",
        }
        if set(document) & {"validation", "conversion_validation"}:
            raise InputError(
                "Megatron conversion/parity thresholds do not apply to Core; use separate Core parity probes"
            )
        validation.fields(document, "run sections", allowed)
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise InputError("schema_version must be 1")
        name = _text(document.get("name"), "name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise InputError("name must contain only letters, digits, dots, underscores and hyphens")
        base = path.parent
        model = _table(document, "model", {"source", "format", "hf_template", "reference_hf"}, required=True)
        model["source"] = _path(model.get("source"), base, "model.source")
        if model.get("reference_hf") is not None:
            raise InputError(
                "model.reference_hf describes baseline conversion validation, which is not implemented here; "
                "use miles.ref_load for a frozen KL reference"
            )
        model.setdefault("format", "hf")
        if model["format"] not in ("hf", "olmo_core"):
            raise InputError("model.format must be hf or olmo_core; Megatron checkpoints are not Core inputs")
        if model["format"] == "olmo_core" and not model.get("hf_template"):
            raise InputError("model.hf_template is required for an olmo_core input")
        if model["format"] == "hf" and "hf_template" in model:
            raise InputError("model.hf_template applies only to olmo_core inputs")
        for key in ("hf_template",):
            if key in model:
                model[key] = _path(model[key], base, f"model.{key}")
        output = _table(document, "output", {"root", "export_hf", "hf_dir"}, required=True)
        output["root"] = _path(output.get("root"), base, "output.root")
        output["export_hf"] = _boolean(output.get("export_hf", False), "output.export_hf")
        output["hf_dir"] = _path(output.get("hf_dir", str(Path(output["root"]) / "export-hf")), base, "output.hf_dir")
        conversion = _table(document, "conversion", {"hf_output"})
        conversion["hf_output"] = _path(
            conversion.get("hf_output", str(Path(output["root"]) / "prepared" / "hf")), base, "conversion.hf_output"
        )
        data = cls._data(_table(document, "data", required=True), base)
        run_data.validate_data(data)
        launch = cls._launch(_table(document, "launch"), base)
        cache = _table(document, "compiler_cache", {"enabled", "shared_root", "restore", "diagnostics"})
        for key in ("enabled", "restore", "diagnostics"):
            if key in cache:
                _boolean(cache[key], f"compiler_cache.{key}")
        if "shared_root" in cache:
            cache["shared_root"] = _path(cache["shared_root"], base, "compiler_cache.shared_root")
        result = cls(
            name,
            path,
            model,
            data,
            output,
            launch,
            conversion,
            cache,
            {section: _table(document, section) for section in RUN_SECTIONS},
            _table(document, "core", CORE_FIELDS),
            _table(document, "miles"),
            judging.parse(document),
        )
        result.compile()
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run contract, with filesystem paths independent of submitter cwd."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "name": self.name,
            "model": self.model,
            "data": self.data,
            "output": self.output,
            "launch": self.launch,
            "conversion": self.conversion,
            "compiler_cache": self.compiler_cache,
            **self.sections,
            "core": self.core,
            "miles": self.miles,
            **self.judges,
        }
        payload = copy.deepcopy(payload)
        for section in (*RUN_SECTIONS, "core", "miles"):
            for key, value in payload[section].items():
                if key in PATH_OPTIONS | {
                    "replay_rollout_data",
                    "model_config",
                    "reward_config",
                    "compiler_cache_root",
                }:
                    payload[section][key] = _path(value, self.config_path.parent, f"{section}.{key}")
        return payload

    @staticmethod
    def _data(data, base):
        allowed = {
            "seed",
            "shuffle",
            "tasks",
            "recipe",
            "rl_manifest",
            "prompt_data",
            "eval_prompt_data",
            "reward_config",
        }
        if unknown := set(data) - allowed:
            raise InputError(f"Unknown [data] fields: {sorted(unknown)}")
        selectors = [key for key in ("tasks", "recipe", "rl_manifest", "prompt_data") if key in data]
        if len(selectors) != 1:
            raise InputError(
                f"data must select exactly one of tasks, recipe, rl_manifest or prompt_data; found {selectors or 'none'}. Remove competing selectors or add data.tasks."
            )
        data.setdefault("seed", 17)
        if type(data["seed"]) is not int or data["seed"] < 0:
            raise InputError("data.seed must be a nonnegative integer")
        data["shuffle"] = _boolean(data.get("shuffle", True), "data.shuffle")
        for key in ("rl_manifest", "prompt_data", "reward_config"):
            if key in data:
                data[key] = _path(data[key], base, f"data.{key}")
        if "eval_prompt_data" in data:
            data["eval_prompt_data"] = _eval_paths(data["eval_prompt_data"], base)
        if "recipe" in data:
            _text(data["recipe"], "data.recipe")
        if "tasks" in data:
            if not isinstance(data["tasks"], list) or not data["tasks"]:
                raise InputError("data.tasks must be a nonempty array of task tables")
            names = []
            for index, task in enumerate(data["tasks"]):
                if not isinstance(task, dict) or set(task) - {"task", "train_count", "eval_count", "prompt_wrapper"}:
                    raise InputError("data.tasks entries accept task, train_count, eval_count and prompt_wrapper")
                names.append(_text(task.get("task"), f"data.tasks[{index}].task"))
                if "train_count" not in task and "eval_count" not in task:
                    raise InputError("Each task must set train_count or eval_count")
                for key in ("train_count", "eval_count"):
                    if key in task:
                        _positive(task[key], f"data.tasks[{index}].{key}")
                if "prompt_wrapper" in task:
                    _text(task["prompt_wrapper"], "data.tasks.prompt_wrapper")
            if len(names) != len(set(names)):
                raise InputError("data.tasks must not repeat task names")
            if not any("train_count" in task for task in data["tasks"]):
                raise InputError("data.tasks must select training data")
        return data

    @staticmethod
    def _launch(launch, base):
        allowed = {
            "workspace",
            "budget",
            "cluster",
            "priority",
            "min_runtime",
            "auto_resume",
            "shared_memory",
            "gpus_per_replica",
            "weka_mounts",
            "env",
            "secrets",
            "timeout",
            "coordination",
        }
        if unknown := set(launch) - allowed:
            raise InputError(f"Unknown [launch] fields: {sorted(unknown)}")
        defaults = dict(
            workspace="ai2/open-instruct-dev",
            budget="ai2/oe-other",
            cluster="ai2/holmes",
            priority="urgent",
            min_runtime="1h",
            auto_resume=True,
            shared_memory="200 GiB",
            timeout="3h",
        )
        launch = defaults | launch
        coordination = launch.setdefault("coordination", {})
        validation.mapping(coordination, "launch.coordination")
        validation.fields(coordination, "launch.coordination", {"startup_timeout", "heartbeat_timeout"})
        for key, default in (("startup_timeout", 1200), ("heartbeat_timeout", 120)):
            coordination.setdefault(key, default)
            _positive(coordination[key], f"launch.coordination.{key}")
        for key in ("workspace", "budget", "cluster", "priority", "min_runtime", "shared_memory", "timeout"):
            _text(launch[key], f"launch.{key}")
        if launch["priority"] not in ("low", "normal", "high", "urgent"):
            raise InputError("launch.priority must be low, normal, high or urgent")
        _boolean(launch["auto_resume"], "launch.auto_resume")
        if "gpus_per_replica" in launch:
            _positive(launch["gpus_per_replica"], "launch.gpus_per_replica")
        for key in ("env", "secrets"):
            launch.setdefault(key, {})
            if not isinstance(launch[key], dict) or any(not isinstance(value, str) for value in launch[key].values()):
                raise InputError(f"launch.{key} must map names to strings")
            validation.mapping(launch[key], f"launch.{key}")
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in launch[key]):
                raise InputError(f"launch.{key} names must be valid environment-variable identifiers")
        if duplicate := set(launch["env"]) & set(launch["secrets"]):
            raise InputError(f"launch.env and launch.secrets overlap: {sorted(duplicate)}")
        reserved = {
            "OI_MILES_REPLICA_RANK",
            "OI_MILES_REPLICA_COUNT",
            "OI_MILES_LAUNCH_ID",
            "OI_MILES_JUDGE_REGISTRY",
            "RAY_ADDRESS",
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        }
        if managed := reserved & (set(launch["env"]) | set(launch["secrets"])):
            raise InputError(f"The launcher/runtime owns these environment variables: {sorted(managed)}")
        launch.setdefault("weka_mounts", [{"weka": "oe-training-default", "mount_path": "/weka/oe-training-default"}])
        if not isinstance(launch["weka_mounts"], list):
            raise InputError("launch.weka_mounts must be an array of tables")
        seen_filesystems = set()
        seen_paths = []
        for mount in launch["weka_mounts"]:
            if not isinstance(mount, dict) or set(mount) != {"weka", "mount_path"}:
                raise InputError("Each launch.weka_mounts entry requires weka and mount_path")
            _text(mount["weka"], "launch.weka_mounts.weka")
            mount["mount_path"] = _path(mount["mount_path"], base, "launch.weka_mounts.mount_path")
            mount_path = Path(mount["mount_path"])
            if mount_path == Path("/") or any(
                mount_path == previous or mount_path in previous.parents or previous in mount_path.parents
                for previous in seen_paths
            ):
                raise InputError("launch.weka_mounts paths must be distinct, nonoverlapping and below filesystem root")
            if mount["weka"] in seen_filesystems:
                raise InputError("launch.weka_mounts must not repeat a filesystem")
            seen_filesystems.add(mount["weka"])
            seen_paths.append(mount_path)
        return launch

    def compile(self, prepared: dict | None = None) -> RunConfig:
        """Resolve settings without opening checkpoints, datasets or GPU libraries."""
        prepared = prepared or {}
        if unknown := set(prepared) - {
            "hf_checkpoint",
            "prompt_data",
            "eval_prompt_data",
            "reward_config",
            "manifest",
        }:
            raise InputError(f"Unknown prepared fields: {sorted(unknown)}")
        root = Path(self.output["root"])
        base = self.config_path.parent
        values = {"core": {}, "miles": {}}
        origins = {}
        controls = {}

        def put(target, value, origin):
            section, key = target.split(".", 1)
            if section == "miles":
                record, value = options.resolve_option(key, value)
                key = record["dest"]
                target = f"miles.{key}"
            if (section == "miles" and key in PATH_OPTIONS) or (
                section == "core" and key in ("model_config", "reward_config", "compiler_cache_root")
            ):
                value = _path(value, base, target)
            try:
                if section == "miles":
                    options.encode_options({key: value})
                    validation.runtime_values({key: value})
                else:
                    CoreConfig(**{key: value})
            except InputError as error:
                raise InputError(f"{origin}: {error}") from error
            if target in origins and values[section][key] != value:
                raise InputError(f"Conflicting settings for {target}: {origins[target]} and {origin}")
            origins[target] = origin
            values[section][key] = value

        for section in RUN_SECTIONS:
            for key, value in self.sections[section].items():
                origin = f"{section}.{key}"
                if key == "gpus" and section in ("trainer", "inference"):
                    put(
                        "miles.actor_num_gpus_per_node" if section == "trainer" else "miles.rollout_num_gpus",
                        value,
                        origin,
                    )
                elif key in UNSUPPORTED_FIELDS:
                    raise InputError(f"{origin} is unsupported: {UNSUPPORTED_FIELDS[key]}")
                elif key in (
                    "placement_mode",
                    "max_context_length",
                    "save_checkpoints",
                    "off_policy_correction",
                    "policy_drift_action",
                ):
                    if key in controls and controls[key][0] != value:
                        raise InputError(f"Conflicting settings for {key}")
                    controls[key] = (value, origin)
                elif key in ("radix_cache", "disable_radix_cache"):
                    value = _boolean(value, origin)
                    put("miles.sglang_disable_radix_cache", not value if key == "radix_cache" else value, origin)
                elif key == "trainer_diagnostics":
                    put("core.diagnostic_interval", int(_boolean(value, origin)), origin)
                elif key == "recompute_mode":
                    if value not in ("full", "off"):
                        raise InputError(f"{origin}: Core supports full/off block recomputation, not selective mode")
                    put("core.activation_checkpointing", value == "full", origin)
                elif key == "trainer_flash_attention_version":
                    if type(value) is not int or value not in (2, 3, 4):
                        raise InputError(f"{origin} must be 2, 3 or 4")
                    put("core.attention_backend", f"flash_{value}", origin)
                elif key == "dynamic_batching":
                    put("miles.use_dynamic_batch_size", _boolean(value, origin), origin)
                elif key in FIELD_MAP:
                    put(FIELD_MAP[key], value, origin)
                elif key in CORE_FIELDS:
                    put(f"core.{key}", value, origin)
                else:
                    put(f"miles.{key}", value, origin)
        for key, value in self.core.items():
            if key in ("model_config", "reward_config", "compiler_cache_root"):
                value = _path(value, base, f"core.{key}")
            put(f"core.{key}", value, f"core.{key}")
        for key, value in options.normalize_options(self.miles).items():
            put(f"miles.{key}", value, f"miles.{key}")
        for key, field in {
            "enabled": "compiler_cache",
            "shared_root": "compiler_cache_root",
            "restore": "compiler_cache_restore",
            "diagnostics": "compiler_cache_diagnostics",
        }.items():
            if key in self.compiler_cache:
                put(f"core.{field}", self.compiler_cache[key], f"compiler_cache.{key}")
        core, miles = values["core"], values["miles"]
        # Check explicit scalar types before doing any batch/topology arithmetic.
        options.encode_options(miles)
        validation.runtime_values(miles)
        validation.inference_capacity(miles)
        CoreConfig(**core)
        if "placement_mode" in controls:
            placement, origin = controls["placement_mode"]
            if placement not in ("colocated", "disaggregated"):
                raise InputError("inference.placement_mode must be colocated or disaggregated")
            put("miles.colocate", placement == "colocated", origin)
        asynchronous = _boolean(miles.get("fully_async", False), "async.fully_async")
        miles.setdefault("colocate", not asynchronous)
        miles.setdefault("actor_num_nodes", 1)
        miles.setdefault("actor_num_gpus_per_node", 2)
        world = _positive(miles["actor_num_nodes"], "trainer_num_nodes") * _positive(
            miles["actor_num_gpus_per_node"], "trainer.gpus"
        )
        miles.setdefault("rollout_num_gpus", world if miles["colocate"] else 1)
        miles.setdefault("rollout_num_gpus_per_engine", 1)
        _positive(miles["rollout_num_gpus"], "inference.gpus")
        tp = _positive(miles["rollout_num_gpus_per_engine"], "inference.rollout_tensor_parallel_size")
        if miles["rollout_num_gpus"] % tp:
            raise InputError(
                f"inference.gpus={miles['rollout_num_gpus']} must be divisible by "
                f"rollout_tensor_parallel_size={tp}; allocate a multiple of {tp} GPUs or lower tensor parallelism."
            )
        if miles["colocate"] and miles["rollout_num_gpus"] != world:
            raise InputError(
                f"Colocated inference.gpus must equal the total trainer GPUs ({world}); got {miles['rollout_num_gpus']}."
            )
        allocated = world if miles["colocate"] else world + miles["rollout_num_gpus"]
        per_node = self.launch.get("gpus_per_replica", min(8, allocated))
        miles.setdefault("num_gpus_per_node", per_node)
        if miles["actor_num_gpus_per_node"] > miles["num_gpus_per_node"]:
            raise InputError("trainer.gpus exceeds the declared physical num_gpus_per_node")
        core.setdefault("expert_parallel_size", min(2, world) if world % 2 == 0 else 1)
        core.setdefault("attention_backend", "flash_4")
        core.setdefault("row_specialization", "dynamic")
        core.setdefault("max_policy_lag", 1 if asynchronous else 0)
        if "max_context_length" in controls:
            length, origin = controls["max_context_length"]
            _positive(length, origin)
            for target in ("core.max_sequence_length", "miles.sglang_context_length", "miles.rollout_max_context_len"):
                put(target, length, origin)
        length = core.setdefault("max_sequence_length", miles.get("rollout_max_context_len", 6144))
        miles.setdefault("rollout_max_context_len", length)
        miles.setdefault("sglang_context_length", length)
        miles.setdefault("rollout_max_response_len", min(4096, length // 2 if length <= 4096 else 4096))
        response = _positive(miles["rollout_max_response_len"], "inference.max_response_length")
        if response >= miles["rollout_max_context_len"]:
            raise InputError(
                f"max_response_length={response} must be smaller than max_context_length={miles['rollout_max_context_len']}; "
                "leave room for the prompt or increase the context limit."
            )
        miles.setdefault("rollout_max_prompt_len", miles["rollout_max_context_len"] - response)
        if (
            miles["rollout_max_context_len"] > length
            or miles["rollout_max_context_len"] > miles["sglang_context_length"]
        ):
            raise InputError(
                f"Rollout context {miles['rollout_max_context_len']} exceeds Core ({length}) or SGLang "
                f"({miles['sglang_context_length']}) context capacity; use inference.max_context_length to set all three."
            )
        defaults = {
            "fully_async": asynchronous,
            "offload_train": False,
            "offload_rollout": False,
            "micro_batch_size": 1,
            "rollout_batch_size": 8,
            "n_samples_per_prompt": 8,
            "num_rollout": 100,
            "rollout_temperature": 1.0,
            "rollout_seed": self.data["seed"],
            "seed": self.data["seed"],
            "input_key": "input",
            "label_key": "label",
            "metadata_key": "metadata",
            "custom_rm_path": "open_instruct.miles.rewards.registered_reward",
            "custom_rollout_log_function_path": "open_instruct.miles.rollout_metrics.log_rollout_data",
            "rollout_global_dataset": True,
            "rollout_shuffle": self.data["shuffle"],
            "loss_type": "policy_loss",
            "advantage_estimator": "grpo",
            "use_rollout_logprobs": False,
            "grpo_std_normalization": False,
            "eps_clip": 0.2,
            "eps_clip_high": 0.28,
            "kl_loss_coef": 0.0,
            "entropy_coef": 0.0,
            "lr": 1e-6,
            "lr_decay_style": "constant",
            "lr_warmup_iters": 0,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1e-8,
            "clip_grad": 1.0,
            "update_weight_buffer_size": 1073741824,
            "update_weights_interval": 1,
            "sglang_server_concurrency": 64,
            "sglang_max_running_requests": 64,
            "sglang_mem_fraction_static": 0.4 if miles["colocate"] else 0.6,
            "sglang_disable_radix_cache": True,
            "sglang_max_mamba_cache_size": 128,
            "sglang_cuda_graph_backend_decode": "full",
            "sglang_cuda_graph_max_bs_decode": 64,
            "sglang_cuda_graph_backend_prefill": "disabled",
            "sglang_sampling_backend": "pytorch",
            "sglang_attention_backend": "triton",
            "use_wandb": False,
            "wandb_run_name": self.name,
            "save": str(root / "checkpoints"),
            "save_debug_rollout_data": str(root / "rollouts" / "{rollout_id}.pt"),
            "wandb_dir": str(root / "wandb"),
            "async_save": False,
        }
        for key, value in defaults.items():
            miles.setdefault(key, value)
        miles.setdefault("global_batch_size", miles["rollout_batch_size"] * miles["n_samples_per_prompt"])
        miles.setdefault("sglang_max_total_tokens", max(524288, miles["sglang_max_running_requests"] * length))
        if miles.get("use_rollout_routing_replay"):
            miles.setdefault("use_miles_router", True)
        if controls.get("policy_drift_action", ("fail", ""))[0] != "fail":
            raise InputError("policy_drift_action=warn is not implemented by Core; select fail")
        correction = controls.get("off_policy_correction", ("tis", ""))[0]
        if correction != "tis":
            raise InputError(
                "off_policy_correction supports tis; other corrections require an explicitly qualified custom_tis_function_path"
            )
        if "off_policy_correction" in controls:
            # An explicit algorithm choice must take effect in either schedule,
            # and must agree with any native switch supplied alongside it.
            put("miles.use_tis", True, controls["off_policy_correction"][1])
        miles.setdefault("use_tis", asynchronous and not miles["use_rollout_logprobs"])
        if miles["use_tis"] and miles["use_rollout_logprobs"]:
            raise InputError(
                "use_tis and use_rollout_logprobs cannot both be enabled; use use_tis=true with use_rollout_logprobs=false for trainer-scored correction."
            )
        if asynchronous:
            if not (miles["use_tis"] or miles["use_rollout_logprobs"]):
                raise InputError("Async training requires TIS or an explicit rollout-logprob policy anchor")
            miles.setdefault("async_data_buffer_capacity_factor", 2.0)
            miles.setdefault("async_unused_samples_handler", "retry")
            miles.setdefault("rollout_submission_granularity", "group")
        if "kl_loss_coef" in miles and miles["kl_loss_coef"] > 0:
            miles.setdefault("use_kl_loss", True)
        if miles.get("use_kl_loss"):
            miles.setdefault("ref_load", prepared.get("hf_checkpoint", self.conversion["hf_output"]))
        save_enabled = _boolean(controls.get("save_checkpoints", (True, ""))[0], "training.save_checkpoints")
        if save_enabled:
            miles.setdefault("save_interval", miles["num_rollout"])
        elif "miles.save_interval" in origins:
            raise InputError("save_checkpoints=false conflicts with an explicit save_interval")
        generated = {
            "hf_checkpoint": self.conversion["hf_output"],
            "prompt_data": self.data.get("prompt_data", str(root / "prepared" / "data" / "train.jsonl")),
            "reward_config": self.data.get("reward_config", str(root / "prepared" / "data" / "verifiers.json")),
        }
        for key in generated:
            value = str(prepared.get(key, generated[key]))
            target = f"core.{key}" if key == "reward_config" else f"miles.{key}"
            if target in origins and values[target.split(".")[0]][key] != value:
                raise InputError(f"{target} conflicts with workflow preparation; set model/data fields instead")
            values[target.split(".")[0]][key] = value
        eval_data = prepared.get("eval_prompt_data", self.data.get("eval_prompt_data"))
        if eval_data is None and (
            "tasks" not in self.data or any(task.get("eval_count") for task in self.data["tasks"])
        ):
            eval_data = ["heldout", str(root / "prepared" / "data" / "eval.jsonl")]
        if eval_data:
            put("miles.eval_prompt_data", eval_data, "data.eval_prompt_data")
            miles.setdefault("eval_interval", 20)
            miles.setdefault("skip_eval_before_train", False)
            miles.setdefault("eval_temperature", 0.0)
            miles.setdefault("n_samples_per_eval_prompt", 1)
            miles.setdefault("eval_max_response_len", response)
        elif miles.get("eval_interval") is not None:
            raise InputError(
                "eval_interval requires prepared held-out data; set data.tasks[].eval_count or data.eval_prompt_data, or remove eval_interval."
            )
        if "miles.wandb_mode" in origins and "miles.use_wandb" not in origins:
            miles["use_wandb"] = miles.get("wandb_mode") != "disabled"
        for key in PATH_OPTIONS:
            if key in miles:
                miles[key] = _path(miles[key], base, f"miles.{key}")
        for key in ("save_interval", "eval_interval"):
            if key in miles:
                _positive(miles[key], f"miles.{key}")
        if "async_data_buffer_capacity_factor" in miles:
            capacity = miles["async_data_buffer_capacity_factor"]
            if type(capacity) not in (int, float) or not math.isfinite(capacity) or capacity <= 0:
                raise InputError("async_data_buffer_capacity_factor must be finite and positive")
        result = RunConfig(CoreConfig(**core), miles)
        result.validate()
        return result

    def plan(self) -> dict[str, Any]:
        runtime = self.compile().plan()
        return {
            "schema_version": 1,
            "name": self.name,
            "config_path": str(self.config_path),
            "model": self.model,
            "conversion": self.conversion,
            "data": self.data,
            "output": self.output,
            "launch": self.launch,
            "compiler_cache": self.compiler_cache,
            "runtime": runtime,
            **self.judges,
            "allocation": topology.plan(self),
            "runtime_validated": False,
            "stages": ["prepare_model", "prepare_data", "validate", "train"]
            + (["export_hf"] if self.output["export_hf"] else []),
        }
