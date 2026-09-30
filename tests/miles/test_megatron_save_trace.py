"""Trace wrappers preserve callable behavior; no CUDA dependency required."""

import hashlib
from types import SimpleNamespace

import pytest

from open_instruct.miles import megatron_save_trace


def test_instance_and_static_bindings():
    events = []

    class Writer:
        def finish(self, value, *, extra=0):
            return value + extra

        @staticmethod
        def preload(value):
            return value * 2

    def emit(*args, **kw):
        events.append((args, kw))

    megatron_save_trace.wrap(Writer, "finish", "finish", emit)
    megatron_save_trace.wrap(Writer, "preload", "preload", emit)
    assert Writer().finish(4, extra=3) == 7
    assert Writer.preload(3) == Writer().preload(3) == 6
    assert [event[0][1] for event in events] == ["enter", "exit"] * 3


def test_function_exception_and_arguments_unchanged():
    events = []
    error = RuntimeError("failure")

    def original(value):
        assert value is error
        raise value

    owner = SimpleNamespace(run=original)
    megatron_save_trace.wrap(owner, "run", "writer", lambda *a, **kw: events.append((a, kw)))
    with pytest.raises(RuntimeError) as caught:
        owner.run(error)
    assert caught.value is error
    assert events == [(("writer", "enter"), {}), (("writer", "error"), {"error_type": "RuntimeError"})]


def test_source_hash_fails_closed(tmp_path):
    source = tmp_path / "native.py"
    source.write_bytes(b"native")
    digest = hashlib.sha256(b"native").hexdigest()
    assert megatron_save_trace.verify_source(source, digest) == digest
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Diagnostic source mismatch"):
        megatron_save_trace.verify_source(source, digest)
