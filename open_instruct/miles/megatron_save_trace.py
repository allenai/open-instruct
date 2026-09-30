"""Internal save-only diagnostic for the exact inspected native converter.

Run under torchrun with this module and the native converter filename followed
by its original arguments. No save format, synchronization or fsync is changed.
"""

import faulthandler
import functools
import hashlib
import importlib
import inspect
import json
import os
import runpy
import sys
import time
from pathlib import Path

from open_instruct import logger_utils
from open_instruct.miles import megatron_grpo_convert

logger = logger_utils.setup_logger(__name__)

SOURCE_HASHES = {
    "filesystem_async": "d6222ed7627d84b53226604f2694a26e6b07e12d4a6dce360c76d622366ca9b6",
    "state_dict_saver": "3b35f1e9cc1dd3a48827675c60508303bdd4bb285d78fb7d6fb56ec57ef2e69c",
    "async_utils": "f4dc1559b13ecfa91ac49d11d9372dcff12b95e1d6b3517b90970fcf8d83186e",
    "torch": "c8f7511363ddbf58ce3ae978ca6efa73751daa3d7e9b5f8d8e8956acae776876",
}
CONVERTER_HASH = "0c2541d30073777a30344273a3773844a70ca1961287520c0496a1cec18d43f6"


def verify_source(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Diagnostic source mismatch for {path}: {actual}")
    return actual


def record(stage, event, **values):
    logger.info(
        "[ConversionSaveTrace] %s",
        json.dumps(
            {"stage": stage, "event": event, "rank": os.getenv("RANK"), "unix": time.time(), **values}, sort_keys=True
        ),
    )


def wrap(owner, name, stage, emit=record):
    """Trace entry/return/exception while preserving instance/static binding."""
    descriptor = inspect.getattr_static(owner, name)
    original = getattr(owner, name)

    @functools.wraps(original)
    def traced(*args, **kwargs):
        emit(stage, "enter")
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            emit(stage, "error", error_type=type(error).__name__)
            raise
        emit(stage, "exit")
        return result

    setattr(owner, name, staticmethod(traced) if isinstance(descriptor, staticmethod) else traced)


def install():
    modules = {}
    for name, expected in SOURCE_HASHES.items():
        module = importlib.import_module("megatron.core.dist_checkpointing.strategies." + name)
        verify_source(module.__file__, expected)
        modules[name] = module
    writer = modules["filesystem_async"].FileSystemWriterAsync
    for name in [
        "preload_tensors",
        "write_preloaded_data",
        "write_preloaded_data_multithread",
        "retrieve_write_results",
        "finish",
    ]:
        wrap(writer, name, name)
    wrap(modules["async_utils"].AsyncRequest, "execute_sync", "execute_sync")
    wrap(modules["state_dict_saver"], "save_state_dict_async_finalize", "metadata_finalize")
    wrap(importlib.import_module("torch.distributed"), "barrier", "distributed_barrier")
    wrap(os, "fsync", "fsync")
    record("source_verification", "passed", hashes=SOURCE_HASHES)


def main():
    native = Path(sys.argv[1])
    verify_source(native, CONVERTER_HASH)
    megatron_grpo_convert.verify_keep_pp1(native.read_text(), str(native))
    if os.getenv("CONVERT_KEEP_PP1") != "1":
        raise ValueError("Save diagnostic requires native CONVERT_KEEP_PP1=1")
    output = Path(os.environ["OI_CONVERSION_TRACE_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ("stacks-rank" + os.environ["RANK"] + ".log")).open("w") as stack:
        faulthandler.enable(file=stack, all_threads=True)
        faulthandler.dump_traceback_later(120, repeat=True, file=stack)
        try:
            install()
            sys.argv = [str(native), *sys.argv[2:]]
            record("native_converter", "enter", converter_sha256=CONVERTER_HASH)
            runpy.run_path(str(native), run_name="__main__")
            record("native_converter", "exit")
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()


if __name__ == "__main__":
    main()
