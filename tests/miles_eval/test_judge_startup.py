"""Check lifecycle timeout routing against the installed evaluator, without a GPU."""

import importlib
import inspect
import time
from unittest.mock import Mock
from urllib import error, request

import pytest

pytest.importorskip("olmo_eval", exc_type=ModuleNotFoundError)
manager = importlib.import_module("olmo_eval.inference.manager")
config = importlib.import_module("olmo_eval.inference.providers.config")
server_utils = importlib.import_module("olmo_eval.inference.providers.vllm_server_utils")


@pytest.mark.parametrize("model", ["allenai/wildguard", "google/gemma-2b"])
def test_startup_timeout_reaches_lifecycle_not_vllm_cli(model, monkeypatch):
    monkeypatch.setattr(server_utils, "_find_free_internal_port", lambda **kwargs: 20001)
    monkeypatch.setattr(server_utils, "_find_free_port", lambda: 20000)
    provider = config.ProviderConfig(kind="vllm_server", model=model, kwargs={"startup_timeout": 1200})
    server = manager._create_server(provider, gpu_ids=[1])
    assert server.startup_timeout == 1200
    assert "startup_timeout" not in server.server_kwargs
    argv = server_utils._build_server_command(model, server.port, **server.server_kwargs)
    assert not any("startup-timeout" in arg or "startup_timeout" in arg for arg in argv)


@pytest.mark.parametrize("mode", ["slow", "dead", "timeout"])
def test_readiness_retains_slow_start_dead_process_detection_and_bound(mode, monkeypatch):
    started = time.monotonic()
    process = Mock()
    process.poll.return_value = 1 if mode == "dead" else None
    process.communicate.return_value = (b"judge startup failed", None)
    response = Mock()
    response.__enter__ = Mock(return_value=Mock(status=200))
    response.__exit__ = Mock(return_value=False)

    def health(*args, **kwargs):
        if mode == "slow" and time.monotonic() - started >= 0.03:
            return response
        raise error.URLError("not ready")

    monkeypatch.setattr(request, "urlopen", health)
    progress = Mock()
    success, last_error, output = server_utils._wait_for_server(
        "http://localhost:20000/v1", timeout=0.3, interval=0.01, process=process, progress_callback=progress
    )
    elapsed = time.monotonic() - started
    if mode == "slow":
        assert success and 0.03 <= elapsed < 5
    elif mode == "dead":
        assert not success and elapsed < 5
        if "log_path" not in inspect.signature(server_utils._wait_for_server).parameters:
            # The original lifecycle must retain the crashed child's pipe output.
            assert "judge startup failed" in output
        assert "Process exited" in str(last_error)
    else:
        assert not success and 0.3 <= elapsed < 5
    assert progress.called
