"""Register hybrid serving capabilities before SGLang resolves its arguments."""

import importlib
import os
import runpy
import shlex
import sys
from functools import partial

from miles.utils.compiler_cache import startup_cache

from open_instruct.miles.configuration import inference_profiles

CORE_COMPAT_ENV = "OLMO_SGLANG_CORE_COMPAT"


def configure_specs(args, specs):
    configured = startup_cache.configure_specs(args, specs)
    rounding_default = _default_tp2_rounding(args)
    for index, spec in enumerate(configured):
        if not spec.name.startswith("inference-engine-"):
            continue
        updates = {"launch_command": partial(_serving_command, spec)}
        if rounding_default:
            updates["env_var"] = partial(_rounding_environment, spec)
        configured[index] = spec.model_copy(update=updates)
    return configured


def _default_tp2_rounding(args):
    if (
        args.olmo_core.publication_mode != "refresh"
        or getattr(args, "rollout_num_gpus_per_engine", 1) != 2
        or CORE_COMPAT_ENV in os.environ
        or getattr(args, "sglang_attn_cp_size", None) not in (None, 1)
        or getattr(args, "sglang_moe_dp_size", None) not in (None, 1)
        or getattr(args, "sglang_quantization", None)
        or getattr(args, "sglang_moe_runner_backend", None) not in (None, "auto", "triton")
    ):
        return False
    model, _ = inference_profiles.model_metadata({}, {"hf_checkpoint": args.hf_checkpoint})
    if not model or model.get("model_type") != "olmo3moe":
        return False
    dtype = getattr(args, "sglang_dtype", None)
    if dtype in (None, "auto"):
        dtype = model.get("dtype") or model.get("torch_dtype")
    layers = model.get("layer_types") or ()
    return (
        dtype == "bfloat16"
        and not model.get("quantization_config")
        and not model.get("attention_bias")
        and model.get("use_rope") is False
        and "linear_attention" in layers
        and set(layers) <= {"linear_attention", "full_attention"}
    )


def _rounding_environment(spec, context):
    env = dict(spec.env_var(context))
    env.setdefault(CORE_COMPAT_ENV, "rounding")
    return env


def _serving_command(spec, context):
    command = shlex.split(spec.launch_command(context))
    module_index = command.index("-m") + 1
    if command[module_index] != "miles.utils.compiler_cache.serving":
        raise ValueError("Expected the MILES serving startup hook")
    command[module_index] = "open_instruct.miles.execution.serving"
    return shlex.join(command)


def main():
    startup_cache.setup_worker()
    # Restore caches before importing Transformers or serving kernels, as in
    # MILES' serving entrypoint. Register KDA before argument resolution: lazy
    # model discovery otherwise happens after the radix capability check.
    importlib.import_module("miles.utils.compiler_cache.hf_module_cache").prime_serving_modules(sys.argv[1:])
    importlib.import_module("olmo_sglang.kda.backend").register_olmo_kda_backend()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
