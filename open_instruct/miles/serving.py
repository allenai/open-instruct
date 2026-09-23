"""Restore the serving compiler cache before importing SGLang or its models."""

import runpy

from open_instruct.miles import startup_cache

if __name__ == "__main__":
    startup_cache.setup_worker()
    runpy.run_module("sglang.launch_server", run_name="__main__")
