"""Register hybrid serving capabilities before SGLang resolves its arguments."""

import importlib
import runpy
import shlex
import sys
from functools import partial

from miles.utils.compiler_cache import startup_cache


def configure_specs(args, specs):
    configured = startup_cache.configure_specs(args, specs)
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
