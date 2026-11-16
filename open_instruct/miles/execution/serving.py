"""Register hybrid serving capabilities before SGLang resolves its arguments."""

import importlib
import runpy
import shlex
import sys
from functools import partial

from miles.utils.compiler_cache import startup_cache

from open_instruct.miles.execution import fp8_publication, kda_dispatch, quantized_publication


def configure_specs(args, specs):
    configured = startup_cache.configure_specs(args, specs)
    if getattr(getattr(args, "olmo_core", None), "freeze_router", False):
        configured = [
            spec.model_copy(
                update={"worker_class": "open_instruct.miles.execution.frozen_router_actor.FrozenRouterTrainActor"}
            )
            if spec.worker_class == "miles.backends.core_utils.actor.OLMoCoreTrainRayActor"
            else spec
            for spec in configured
        ]
    return [
        spec.model_copy(update={"launch_command": partial(_serving_command, spec)})
        if spec.name.startswith("inference-engine-")
        else spec
        for spec in configured
    ]


def _serving_command(spec, context):
    command = shlex.split(spec.launch_command(context))
    module_index = command.index("-m") + 1
    if command[module_index] != "miles.utils.compiler_cache.serving":
        raise ValueError("Expected the MILES serving startup hook")
    command[module_index] = "open_instruct.miles.execution.serving"
    return shlex.join(command)


def run_scheduler_process(*args, **kwargs):
    """Install publication hooks inside the spawned scheduler, before construction."""
    fp8_publication.install_fp8_publication()
    return kda_dispatch.run_scheduler_process(*args, **kwargs)


def main():
    startup_cache.setup_worker()
    # Restore caches before importing Transformers or serving kernels, as in
    # MILES' serving entrypoint. Register KDA before argument resolution: lazy
    # model discovery otherwise happens after the radix capability check.
    importlib.import_module("miles.utils.compiler_cache.hf_module_cache").prime_serving_modules(sys.argv[1:])
    importlib.import_module("olmo_sglang.kda.backend").register_olmo_kda_backend()
    quantized_publication.install_nvfp4_publication_context()
    fp8_publication.install_fp8_publication()
    http_server = importlib.import_module("sglang.srt.entrypoints.http_server")
    original_launch = http_server.launch_server

    def launch(server_args, *args, **kwargs):
        if args or "run_scheduler_process_func" in kwargs:
            raise ValueError("Publication compatibility requires the application scheduler entrypoint")
        return original_launch(server_args, run_scheduler_process_func=run_scheduler_process, **kwargs)

    http_server.launch_server = launch
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
