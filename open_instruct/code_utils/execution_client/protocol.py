"""Client-side wire limits, test decoding and validation; no execution implementation."""

import base64
import io
import json
import math
import pickle
import zlib

COMPARISONS = ("exact", "nemo_v1")

MAX_TESTS = 2000
MIN_PER_TEST_SECONDS = 0.05
MAX_PER_TEST_SECONDS = 60.0
MAX_DECODED_TEST_BYTES = 128 << 20


class BadRequest(ValueError):
    pass


class _RestrictedUnpickler(pickle.Unpickler):
    """Only plain data may come out of a pickled test payload."""

    _ALLOWED = {("builtins", "set"), ("builtins", "frozenset"), ("builtins", "bytearray")}

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"pickled tests may not reference {module}.{name}")


def _decompress(data):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(data, validate=True), MAX_DECODED_TEST_BYTES + 1)
    if len(raw) > MAX_DECODED_TEST_BYTES or decoder.unconsumed_tail:
        raise BadRequest("decoded tests exceed size limit")
    if not decoder.eof or decoder.unused_data:
        raise BadRequest("invalid compressed tests")
    return raw


def decode_tests(tests):
    """Accept the three encodings clients have historically sent: a list, a JSON string, or a
    base64(zlib(pickle(list))) string. Returns a list."""
    if tests is None or tests == "":
        return []
    if isinstance(tests, list):
        return tests
    if isinstance(tests, dict) and tests.get("encoding") == "zlib+json":
        try:
            decoded = json.loads(_decompress(tests["data"]))
        except Exception as exc:
            raise BadRequest(f"tests could not be decoded: {type(exc).__name__}") from exc
        if not isinstance(decoded, list):
            raise BadRequest("compressed tests must decode to a list")
        return decoded
    if not isinstance(tests, str):
        raise BadRequest("tests must be a list or an encoded string")
    try:
        decoded = json.loads(tests)
    except (ValueError, RecursionError):
        decoded = None
    if decoded is not None:
        if not isinstance(decoded, list):
            raise BadRequest("tests JSON must decode to a list")
        return decoded
    try:
        raw = _decompress(tests)
        decoded = _RestrictedUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:  # noqa: BLE001
        raise BadRequest(f"tests could not be decoded: {type(exc).__name__}") from exc
    if not isinstance(decoded, list):
        raise BadRequest("pickled tests must decode to a list")
    return decoded


def validate_tests(mode, tests):
    if len(tests) > MAX_TESTS:
        raise BadRequest(f"too many tests ({len(tests)} > {MAX_TESTS})")
    if mode == "assert":
        if not all(isinstance(test, str) for test in tests):
            raise BadRequest("assert-mode tests must be strings")
    elif mode == "stdio":
        for test in tests:
            if not (isinstance(test, dict) and "input" in test and "output" in test):
                raise BadRequest("stdio-mode tests must be objects with 'input' and 'output'")
            inputs = test["input"]
            if isinstance(inputs, list):
                if not all(isinstance(line, str) for line in inputs):
                    raise BadRequest("stdio-mode 'input' list must contain only strings")
            elif not isinstance(inputs, str):
                raise BadRequest("stdio-mode 'input' must be a string or list of strings")
            if not isinstance(test["output"], str):
                raise BadRequest("stdio-mode 'output' must be a string")
            if test.get("comparison", "exact") not in COMPARISONS:
                raise BadRequest("unknown stdio output comparison")
    else:
        raise BadRequest(f"unknown mode {mode!r}")


def clamp_per_test_seconds(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise BadRequest("per-test time limit must be a number")
    return min(MAX_PER_TEST_SECONDS, max(MIN_PER_TEST_SECONDS, float(value)))
